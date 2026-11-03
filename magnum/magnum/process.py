"""Прототип: модель как случайный процесс, где ЧТО происходит (эффект события) отделено от того,
КОГДА оно наступает (критерий). Тогда движок сам знает вероятностную структуру модели и
считает правдоподобие наблюдённой истории без ручного вывода.

Четыре вида критериев:

* ``@rate(lambda m: ...)`` — природа: событие наступает с интенсивностью λ(состояние, параметры)
  (в единицах «раз в день»); пока состояние не меняется, λ постоянна;
* ``@when(lambda m: ...)`` — решение: событие наступает сразу, как только условие истинно;
* ``timer(lambda m: распределение, fires="событие")`` — задержка для конкретной сущности
  (клиент, заказ): запускается и отменяется эффектами (``self.start`` / ``self.cancel``),
  по истечении вызывает событие с этой сущностью;
* ``@daily(lambda m: время суток)`` — расписание: событие каждый день в заданное время (открытие,
  закрытие). Оно детерминировано, в журнале не нужно и в правдоподобие ничего не добавляет, но меняет
  состояние — а через него интенсивности (например, ``λ = 15 в час, если открыто, иначе 0``).
  Время расписания берётся из целых (структурных) параметров, как ``open_hour``.

Эффекты детерминированы, кроме одного явного источника случайности — ``self.choose(lambda m: веса)``:
выбор варианта с вероятностями, заданными параметрами (например, сколько позиций в чеке). Выбор пишется
в журнал, а его вероятность входит в правдоподобие слагаемым log p(выбранного варианта).
Остальная случайность — в критериях. Поэтому, зная
моменты природных событий и срабатываний таймеров, движок однозначно восстанавливает всю историю
(решения ``when`` выводятся сами) и пишет правдоподобие:

    log L = Σ_rate [ log λ(s) в моменты событий − ∫ λ(s(t)) dt ]
          + Σ_timer [ log f(длительность) для сработавших + log S(возраст) для отменённых и незавершённых ]
          + Σ_choose log p(выбранный вариант | s)

Ограничение прототипа: интенсивности зависят от состояния и параметров; время суток входит в них
только через состояние, которое меняют события расписания (кусочно-постоянные интенсивности).
В состоянии — числа, флаги и очереди (для очереди в интенсивность попадает её длина).
"""
from __future__ import annotations

import heapq
from collections import deque
import itertools
import math
import random as _random
import statistics

import torch

from .core import param
from .queues import Queue

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_NORMAL = statistics.NormalDist()


# ------------------------------------------------------------ распределения длительностей
class LogNormal:
    """Логнормальное распределение, заданное средним и стандартным отклонением (а не μ, σ логарифма)."""

    def __init__(self, mean, std):
        mean, std = torch.as_tensor(mean, dtype=torch.float64), torch.as_tensor(std, dtype=torch.float64)
        self.s2 = torch.log1p((std / mean) ** 2)
        self.mu = torch.log(mean) - self.s2 / 2
        self.s = torch.sqrt(self.s2)

    def sample(self, rng: _random.Random) -> float:
        return rng.lognormvariate(float(self.mu), float(self.s))

    def log_prob(self, x):
        lx = torch.log(x)
        return -lx - torch.log(self.s) - _LOG_SQRT_2PI - (lx - self.mu) ** 2 / (2 * self.s2)

    def log_survival(self, x):
        z = (torch.log(x) - self.mu) / (self.s * math.sqrt(2))
        return torch.log(0.5 * torch.erfc(z).clamp_min(1e-300))

    # быстрые версии на числах (для фильтра частиц)
    def floats(self):
        mu, s = float(self.mu), float(self.s)

        def lpf(x):
            if x <= 0:
                return -math.inf
            lx = math.log(x)
            return -lx - math.log(s) - _LOG_SQRT_2PI - (lx - mu) ** 2 / (2 * s * s)

        def lsf(x):
            if x <= 0:
                return 0.0
            return math.log(max(0.5 * math.erfc((math.log(x) - mu) / (s * math.sqrt(2))), 1e-300))

        def cdf(x):
            return 0.0 if x <= 0 else 0.5 * math.erfc(-(math.log(x) - mu) / (s * math.sqrt(2)))

        def ppf(u):
            return math.exp(mu + s * _NORMAL.inv_cdf(min(max(u, 1e-300), 1 - 1e-16)))
        return lpf, lsf, cdf, ppf


class Exponential:
    def __init__(self, mean):
        self.mean = torch.as_tensor(mean, dtype=torch.float64)

    def sample(self, rng):
        return rng.expovariate(1.0 / float(self.mean))

    def log_prob(self, x):
        return -torch.log(self.mean) - x / self.mean

    def log_survival(self, x):
        return -x / self.mean

    def floats(self):
        m = float(self.mean)
        return ((lambda x: -math.log(m) - x / m if x >= 0 else -math.inf), (lambda x: -max(x, 0.0) / m),
                (lambda x: -math.expm1(-max(x, 0.0) / m)), (lambda u: -m * math.log1p(-u)))


# ------------------------------------------------------------ объявления
def rate(fn):
    """Природное событие с интенсивностью fn(m) раз в день."""
    def deco(effect):
        effect.__criterion__ = ("rate", fn)
        return effect
    return deco


def when(fn):
    """Решение: событие наступает сразу, как только fn(m) истинно."""
    def deco(effect):
        effect.__criterion__ = ("when", fn)
        return effect
    return deco


def daily(at):
    """Событие расписания: каждый день в момент at(m) (доля суток, например m.open_hour * HOUR)
    или в каждый из моментов, если at(m) — список."""
    def deco(effect):
        effect.__criterion__ = ("daily", at)
        return effect
    return deco


class timer:  # noqa: N801 — читается как объявление
    """Задержка для сущности: dist(m) — распределение длительности, fires — событие по истечении."""

    def __init__(self, dist, fires):
        self.dist, self.fires, self.name = dist, fires, None

    def __set_name__(self, owner, name):
        self.name = name


class _View:
    """То, что видят критерии: параметры + состояние (для очереди — только её длина)."""

    def __init__(self, params, state):
        self.__dict__.update(params)
        self.__dict__.update(state)


class _Len:
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _weights(fn, view):
    """Веса вариантов choose как числа, нормированные к сумме 1."""
    raw = fn(view)
    if isinstance(raw, torch.Tensor):
        w = [float(x) for x in raw.detach().reshape(-1)]
    else:
        w = [float(x.detach()) if isinstance(x, torch.Tensor) else float(x) for x in raw]
    if not w or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"choose: веса должны быть неотрицательны и не все нулевые, получено {w}")
    tot = sum(w)
    return [x / tot for x in w]


def _pick(w, rng):
    u, acc = rng.random(), 0.0
    for i, wi in enumerate(w):
        acc += wi
        if u < acc:
            return i
    return len(w) - 1


# ------------------------------------------------------------ процесс
class Process:
    """Базовый класс. Наследник объявляет параметры (mg.param), таймеры (timer), метод init()
    (начальное состояние, без случайности) и события-эффекты с критериями @rate / @when / timer."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.params, cls.timers, cls.rates, cls.whens, cls.effects = {}, {}, {}, [], {}
        cls.dailies = {}
        for klass in reversed(cls.__mro__):
            for name, v in vars(klass).items():
                if isinstance(v, param):
                    cls.params[name] = v
                elif isinstance(v, timer):
                    cls.timers[name] = v
                elif callable(v) and hasattr(v, "__criterion__"):
                    kind, fn = v.__criterion__
                    if kind == "rate":
                        cls.rates[name] = fn
                    elif kind == "daily":
                        cls.dailies[name] = fn
                    else:
                        cls.whens.append((name, fn))
                    cls.effects[name] = v
                elif callable(v) and not name.startswith("_") and name != "init":
                    cls.effects.setdefault(name, v)
        cls.fired_by = {t.fires: name for name, t in cls.timers.items()}

    # --- параметры
    @classmethod
    def resolve(cls, controls=None):
        out = {n: p.default.clone() for n, p in cls.params.items()}
        for n, v in (controls or {}).items():
            out[n] = v if isinstance(v, torch.Tensor) and v.requires_grad else cls.params[n].validate(v)
        return out

    # --- служебное для эффектов
    def _new(self, params):
        obj = object.__new__(type(self) if isinstance(self, Process) else self)
        obj.__dict__["_params"] = params
        obj.__dict__["_state"] = {}
        obj.__dict__["_next_id"] = 0
        obj.__dict__["_hooks"] = None
        return obj

    def __getattribute__(self, name):
        d = object.__getattribute__(self, "__dict__")
        st = d.get("_state")
        if st is not None and name in st:
            return st[name]
        pr = d.get("_params")
        if pr is not None and name in pr:
            return pr[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "__dict__")["_state"][name] = value

    def new_id(self):
        self.__dict__["_next_id"] += 1
        return self.__dict__["_next_id"]

    def start(self, timer_name, entity):
        self.__dict__["_hooks"][0](timer_name, entity)

    def cancel(self, timer_name, entity):
        self.__dict__["_hooks"][1](timer_name, entity)

    def choose(self, probs, values=None):
        """Случайный выбор варианта внутри эффекта.

        probs — функция модели (как в критериях): probs(m) возвращает веса вариантов (список или
        тензор; нормируются, сумма не обязана быть 1). Веса могут зависеть от параметров и состояния.
        Возвращает индекс выбранного варианта, а если передан values — values[индекс]:

            n_items = self.choose(lambda m: m.p_items, values=[1, 2, 3, 4])

        Выбор пишется в журнал записью (t, "choose", индекс, "choice") и входит в правдоподобие.
        """
        hooks = self.__dict__["_hooks"]
        if hooks is None or len(hooks) < 3:
            raise RuntimeError("choose можно вызывать только из эффектов во время simulate / replay")
        i = hooks[2](probs)
        return i if values is None else values[i]

    # --- вид состояния для критериев
    def _signature(self, keys):
        """Часть состояния, которую читают интенсивности (для очереди — её длина)."""
        st = object.__getattribute__(self, "__dict__")["_state"]
        sig = []
        for k in keys:
            v = st[k]
            sig.append((k, ("len", len(v)) if isinstance(v, Queue) else v))
        return tuple(sig)

    def _rate_keys(self):
        """Какие переменные состояния читают интенсивности (по пробному вызову на текущем состоянии)."""
        return self._probe_keys(type(self).rates.values())

    def _probe_keys(self, fns):
        """Какие переменные состояния читают функции fns (по пробному вызову на текущем состоянии)."""
        st = object.__getattribute__(self, "__dict__")["_state"]
        read = set()

        class Probe:
            def __getattr__(_, name):
                if name in st:
                    read.add(name)
                    return st[name]
                return 1.0                     # параметр: значение для пробы неважно

        for fn in fns:
            try:
                fn(Probe())
            except Exception:                  # noqa: BLE001 — проба может не пройти арифметику
                pass
        return read

    @staticmethod
    def _view(params, sig):
        st = {k: (_Len(v[1]) if isinstance(v, tuple) and v and v[0] == "len" else v) for k, v in sig}
        return _View(params, st)

    def _resolve_whens(self, on_fire=None):
        fired = []
        again = True
        while again:
            again = False
            for name, cond in type(self).whens:
                if cond(self):
                    if on_fire is not None:      # запись в журнал — до эффекта, чтобы выборы choose шли после неё
                        on_fire(name)
                    type(self).effects[name](self)
                    fired.append(name)
                    again = True
                    break
        return fired

    @classmethod
    def _schedule(cls, structural, horizon):
        """Моменты событий расписания на [0, horizon]: [(t, событие)] по возрастанию."""
        view = _View(structural, {})
        out = []
        for name, at in cls.dailies.items():
            fr = at(view)
            for frac in (fr if isinstance(fr, (list, tuple)) else [fr]):
                frac = float(frac)
                if not 0 <= frac < 1:
                    raise ValueError(f"расписание {name}: время {frac} вне суток [0, 1)")
                out += [(d + frac, name) for d in range(int(math.floor(horizon)) + 1) if d + frac <= horizon]
        return sorted(out)

    # ------------------------------------------------------------ симуляция
    @classmethod
    def simulate(cls, controls=None, seed=0, horizon=1.0):
        """Прогон: возвращает журнал [(t, событие, сущность, вид)] и итоговое состояние.
        Вид: "rate" / "timer" (природа), "when" / "daily" (выводятся сами), "choice" (выбор choose:
        событие "choose", сущность — индекс варианта; идёт сразу после записи события, где сделан)."""
        rng = _random.Random(seed)
        p = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in cls.resolve(controls).items()}
        m = cls._new(cls, p)
        heap, active, seq = [], {}, itertools.count()
        t = 0.0
        log = []

        def start(name, ent):
            d = cls.timers[name].dist(m).sample(rng)
            key = (name, ent)
            tok = next(seq)
            active[key] = tok
            heapq.heappush(heap, (t + d, tok, name, ent))

        def cancel(name, ent):
            active.pop((name, ent), None)

        def choose(fn):
            i = _pick(_weights(fn, m), rng)
            log.append((t, "choose", i, "choice"))
            return i

        def log_when(w):
            log.append((t, w, None, "when"))

        m.__dict__["_hooks"] = (start, cancel, choose)
        sched = cls._schedule({k: int(v) for k, v in p.items() if cls.params[k].integer}, horizon)
        si = 0
        m.init()
        m._resolve_whens(log_when)
        while True:
            view = m
            lam = {e: float(fn(view)) for e, fn in cls.rates.items()}
            total = sum(lam.values())
            t_rate = t + rng.expovariate(total) if total > 0 else math.inf
            while heap and active.get((heap[0][2], heap[0][3])) != heap[0][1]:
                heapq.heappop(heap)
            t_timer = heap[0][0] if heap else math.inf
            t_sched = sched[si][0] if si < len(sched) else math.inf
            t_next = min(t_rate, t_timer, t_sched)
            if t_next > horizon:
                break
            t = t_next
            if t_sched <= min(t_rate, t_timer):      # расписание: интенсивности после него другие —
                ev = sched[si][1]                    # розыгрыш прихода начинается заново (без памяти)
                si += 1
                log.append((t, ev, None, "daily"))
                cls.effects[ev](m)
            elif t_rate <= t_timer:
                u, acc = rng.random() * total, 0.0
                for e, l in lam.items():
                    acc += l
                    if u < acc:
                        break
                log.append((t, e, None, "rate"))
                cls.effects[e](m)
            else:
                _, tok, name, ent = heapq.heappop(heap)
                del active[(name, ent)]
                ev = cls.timers[name].fires
                log.append((t, ev, ent, "timer"))
                cls.effects[ev](m, ent)
            m._resolve_whens(log_when)
        return log, dict(m._state)

    # ------------------------------------------------------------ правдоподобие
    @classmethod
    def replay(cls, log, horizon, controls=None, keys=None):
        """Проигрывает журнал природных событий и таймеров (решения выводятся сами) и собирает всё,
        что нужно правдоподобию: сколько раз и сколько времени каждая интенсивность была при каждом
        состоянии, длительности сработавших таймеров и возрасты отменённых.

        Эффекты и init видят только целые (структурные) параметры, например число мастеров:
        вещественные параметры живут в критериях, поэтому проигрывание от них не зависит и
        делается один раз."""
        structural = {k: int(v) for k, v in cls.resolve(controls).items() if cls.params[k].integer}
        m = cls._new(cls, structural)
        t = 0.0
        active = {}
        done = {n: [] for n in cls.timers}
        cens = {n: [] for n in cls.timers}
        count, exposure = {}, {}

        def start(name, ent):
            active[(name, ent)] = t

        def cancel(name, ent):
            t0 = active.pop((name, ent), None)
            if t0 is not None:
                cens[name].append(t - t0)

        choices = deque(r[2] for r in log if r[3] == "choice")
        picks, choice_keys = {}, {}

        def choose(fn):
            if not choices:
                raise ValueError("в журнале меньше выборов choose, чем их делают эффекты")
            i = choices.popleft()
            if fn not in choice_keys:
                choice_keys[fn] = tuple(sorted(m._probe_keys([fn])))
            key = (fn, m._signature(choice_keys[fn]), i)
            picks[key] = picks.get(key, 0) + 1
            return i

        m.__dict__["_hooks"] = (start, cancel, choose)
        m.init()
        m._resolve_whens()
        keys = tuple(sorted(keys)) if keys is not None else tuple(sorted(m._rate_keys()))
        seen = set(keys)
        # расписание движок знает сам: из журнала его записи не берём, а вставляем свои
        sched = [(ts, ev, None, "daily") for ts, ev in cls._schedule(structural, horizon)]
        recs = [r for r in log if r[3] not in ("when", "daily", "choice")]   # решения выводятся из истории
        merged = sorted(sched + recs, key=lambda r: (r[0], r[3] != "daily"))
        for rec in merged:
            tr, ev, ent, kind = rec
            seen |= m._rate_keys()
            sig = m._signature(keys)
            exposure[sig] = exposure.get(sig, 0.0) + (tr - t)
            t = tr
            if kind == "daily":
                cls.effects[ev](m)
            elif kind == "rate":
                count[(ev, sig)] = count.get((ev, sig), 0) + 1
                cls.effects[ev](m)
            else:
                name = cls.fired_by[ev]
                t0 = active.pop((name, ent), None)
                if t0 is None:
                    raise ValueError(f"в журнале {ev} для {ent}, но таймер {name} для него не запущен")
                done[name].append(t - t0)
                cls.effects[ev](m, ent)
            m._resolve_whens()
        if seen != set(keys):              # интенсивности читают больше, чем показала первая проба — заново
            return cls.replay(log, horizon, controls, keys=seen)
        if choices:
            raise ValueError(f"в журнале на {len(choices)} выборов choose больше, чем их делают эффекты")
        sig = m._signature(keys)
        exposure[sig] = exposure.get(sig, 0.0) + (horizon - t)
        for (name, ent), t0 in active.items():
            cens[name].append(horizon - t0)
        return {"count": count, "exposure": exposure, "choices": picks,
                "done": {n: torch.tensor(v, dtype=torch.float64) for n, v in done.items()},
                "cens": {n: torch.tensor(v, dtype=torch.float64) for n, v in cens.items()}}

    @classmethod
    def log_likelihood(cls, stats, controls):
        """log L журнала при параметрах controls (тензоры с градиентом — градиент идёт в них)."""
        p = cls.resolve(controls)
        ll = torch.zeros((), dtype=torch.float64)
        for e, fn in cls.rates.items():
            for sig, T in stats["exposure"].items():
                n = stats["count"].get((e, sig), 0)
                lam = torch.as_tensor(fn(cls._view(p, sig)), dtype=torch.float64)
                if n:
                    ll = ll + n * torch.log(lam)     # λ = 0 при n > 0 -> −inf: журнал невозможен
                if T:
                    ll = ll - lam * T
        for name, tm in cls.timers.items():
            d = tm.dist(_View(p, {}))
            if len(stats["done"][name]):
                ll = ll + d.log_prob(stats["done"][name]).sum()
            c = stats["cens"][name]
            c = c[c > 0]
            if len(c):
                ll = ll + d.log_survival(c).sum()
        for (fn, sig, i), n in stats.get("choices", {}).items():
            w = torch.as_tensor(fn(cls._view(p, sig)), dtype=torch.float64).reshape(-1)
            ll = ll + n * (torch.log(w[i]) - torch.log(w.sum()))
        return ll
