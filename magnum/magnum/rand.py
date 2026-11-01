"""Случайные величины модели. Все используют сид запуска: один и тот же сид — одни и те же значения.

Аргументы распределений могут быть числами или тензорами torch. Если аргумент — тензор с requires_grad
(обучаемый параметр или выражение от него), логарифм вероятности выпавшего значения добавляется в
``log_prob`` запуска: по нему считается градиент ожидания показателя (см. magnum.train).
Сами выпавшие значения — обычные числа Python.
"""
from __future__ import annotations

import math
import random as _random

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def _grad(x) -> bool:
    return isinstance(x, torch.Tensor) and x.requires_grad


def _f(x) -> float:
    return x.detach().item() if isinstance(x, torch.Tensor) else float(x)


def _t(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)


def _view_of(t):
    """(база, смещение) если t — элемент/строка непрерывного тензора (обучаемого параметра или таблицы,
    посчитанной один раз), иначе None. Такие исходы копятся счётчиками и дают один узел графа на ячейку."""
    base = t._base
    if base is None:
        if not t.is_leaf:
            return None
        base = t
    if not base.is_contiguous() or base.storage_offset() != 0:
        return None
    return base


class RandomSource:
    """Генератор случайных величин, доступный в модели как ``self.rng``."""

    __slots__ = ("_r", "_tape", "_groups", "_counts")

    def __init__(self, seed=None, tape=None):
        self._r = _random.Random(seed)
        self._tape = tape   # список, куда пишутся log p выпавших значений (тензоры с градиентом), или None
        self._groups: dict = {}   # (id базы, версия) -> (база, копия значений на момент первого исхода)
        self._counts: dict = {}   # (id базы, версия, вид, форма, шаги, смещение) -> {исход: число}

    def _log(self, lp):
        if self._tape is not None:
            self._tape.append(lp)

    def _count(self, t, kind, outcome) -> bool:
        """Учесть исход счётчиком (вместо отдельного узла графа). False — если t не подходит."""
        base = _view_of(t)
        if base is None:
            return False
        g = (id(base), base._version)
        if g not in self._groups:
            # копия фиксирует значения: если таблицу потом изменят на месте, новые исходы пойдут в новую группу
            self._groups[g] = (base, base.clone())
        key = (g, kind, tuple(t.shape), t.stride(), t.storage_offset())
        c = self._counts.get(key)
        if c is None:
            c = self._counts[key] = {}
        c[outcome] = c.get(outcome, 0) + 1
        return True

    def flush(self):
        """Перенести накопленные счётчики в log_prob (вызывается движком в конце запуска)."""
        if self._tape is None:
            return
        for (g, kind, shape, stride, offset), c in self._counts.items():
            snap = self._groups[g][1]
            v = snap.as_strided(shape, stride, offset)
            if kind == "bernoulli":
                lp = 0
                if c.get(True):
                    lp = lp + c[True] * torch.log(v)
                if c.get(False):
                    lp = lp + c[False] * torch.log1p(-v)
            else:  # categorical
                idx = torch.tensor(list(c.keys()))
                cnt = torch.tensor(list(c.values()), dtype=v.dtype)
                lp = (cnt * torch.log(v[idx])).sum() - cnt.sum() * torch.log(v.sum())
            self._tape.append(lp)
        self._counts.clear()
        self._groups.clear()

    def random(self) -> float:
        """Равномерно на [0, 1)."""
        return self._r.random()

    def uniform(self, a=0.0, b=1.0) -> float:
        """Равномерно на [a, b]."""
        af, bf = _f(a), _f(b)
        if af > bf:
            raise ValueError(f"uniform: должно быть a <= b, получено a={af:g}, b={bf:g}")
        x = self._r.uniform(af, bf)
        if _grad(a) or _grad(b):
            self._log(-torch.log(_t(b) - _t(a)))
        return x

    def normal(self, mu=0.0, sigma=1.0) -> float:
        """Нормальное N(mu, sigma²)."""
        mf, sf = _f(mu), _f(sigma)
        if sf < 0:
            raise ValueError("normal: sigma должна быть >= 0")
        x = self._r.gauss(mf, sf)
        if _grad(mu) or _grad(sigma):
            s = _t(sigma)
            self._log(-0.5 * ((x - _t(mu)) / s) ** 2 - torch.log(s) - _LOG_SQRT_2PI)
        return x

    def exponential(self, mean) -> float:
        """Экспоненциальное со средним mean (интенсивность 1/mean)."""
        mf = _f(mean)
        if not mf > 0:
            raise ValueError(f"exponential: mean должно быть > 0, получено {mf!r}")
        x = self._r.expovariate(1.0 / mf)
        if _grad(mean):
            self._log(-torch.log(mean) - x / mean)
        return x

    def bernoulli(self, p) -> bool:
        """True с вероятностью p."""
        pf = _f(p)
        if not 0 <= pf <= 1:
            raise ValueError(f"bernoulli: p должна быть в [0, 1], получено {pf!r}")
        x = self._r.random() < pf
        if _grad(p) and self._tape is not None and not self._count(p, "bernoulli", x):
            self._log(torch.log(p) if x else torch.log1p(-p))
        return x

    def poisson(self, lam) -> int:
        """Число Пуассона со средним lam."""
        lf = _f(lam)
        if lf < 0:
            raise ValueError("poisson: lam должна быть >= 0")
        if lf > 500:  # нормальное приближение
            k = max(0, round(self._r.gauss(lf, math.sqrt(lf))))
        else:
            L, k, p = math.exp(-lf), 0, 1.0
            while True:
                p *= self._r.random()
                if p <= L:
                    break
                k += 1
        if _grad(lam):
            self._log(k * torch.log(lam) - lam - math.lgamma(k + 1))
        return k

    def categorical(self, weights) -> int:
        """Индекс 0..n-1 с вероятностью weights[i] / sum(weights); веса >= 0, не обязательно нормированы.
        weights — список чисел или одномерный тензор."""
        if isinstance(weights, torch.Tensor):
            if weights.dim() != 1:
                raise ValueError("categorical: ожидается одномерный массив весов")
            ws = weights.detach().tolist()
            grad = weights.requires_grad
        else:
            ws = []
            grad = False
            for w in weights:
                if isinstance(w, (list, tuple)) or (isinstance(w, torch.Tensor) and w.dim() > 0):
                    raise ValueError("categorical: ожидается одномерный список весов")
                grad = grad or _grad(w)
                ws.append(_f(w))
        total = 0.0
        for w in ws:
            if not (w >= 0) or w == math.inf:
                raise ValueError(f"categorical: веса должны быть конечными и >= 0, получено {w!r}")
            total += w
        if total <= 0:
            raise ValueError("categorical: сумма весов должна быть > 0")
        u = self._r.random() * total
        acc, last, k = 0.0, 0, None
        for i, w in enumerate(ws):
            if w > 0:
                last = i
                acc += w
                if u < acc:
                    k = i
                    break
        if k is None:
            k = last
        if grad and self._tape is not None:
            if isinstance(weights, torch.Tensor):
                if not self._count(weights, "categorical", k):
                    self._tape.append(torch.log(weights[k]) - torch.log(weights.sum()))
            else:
                wt = torch.stack([_t(w) for w in weights])
                self._tape.append(torch.log(wt[k]) - torch.log(wt.sum()))
        return k

    def choice(self, items, weights=None):
        """Случайный элемент списка (равновероятно или по весам)."""
        items = list(items)
        if not items:
            raise ValueError("choice: пустой список")
        if weights is None:
            return items[int(self._r.random() * len(items))]
        return items[self.categorical(weights)]
