"""Встроенные функции выражений и распределения для `arrival`.

Значения языка: float или (вложенный) list из float.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Builtin:
    func: Callable[..., Any]
    random: bool = False            # первым аргументом получает генератор random.Random
    aliases: tuple[tuple[str, str], ...] = ()  # имя в языке -> имя параметра Python

    def signature(self) -> inspect.Signature:
        sig = inspect.signature(self.func)
        if self.random:
            params = list(sig.parameters.values())[1:]
            sig = sig.replace(parameters=params)
        return sig

    def check_call(self, n_args: int, kw_names: list[str]) -> str | None:
        """Проверка числа/имён аргументов на этапе загрузки. Возвращает текст ошибки или None."""
        alias = dict(self.aliases)
        try:
            self.signature().bind(*([0.0] * n_args), **{alias.get(k, k): 0.0 for k in kw_names})
        except TypeError as e:
            return str(e)
        return None

    def call(self, rng, args, kwargs):
        if self.aliases:
            alias = dict(self.aliases)
            kwargs = {alias.get(k, k): v for k, v in kwargs.items()}
        if self.random:
            return self.func(rng, *args, **kwargs)
        return self.func(*args, **kwargs)


def _scalar(x, fname):
    if isinstance(x, list):
        raise ValueError(f"{fname}: ожидалось число, получен массив")
    return x


def _int_arg(x, fname, what="размер"):
    x = _scalar(x, fname)
    if x != int(x) or x < 0:
        raise ValueError(f"{fname}: {what} должен быть целым неотрицательным числом, получено {x}")
    return int(x)


def _unary(fn, name):
    def f(x):
        return float(fn(_scalar(x, name)))
    return f


def _log(x, base=None):
    x = _scalar(x, "log")
    if x <= 0:
        raise ValueError(f"log: аргумент должен быть > 0, получено {x}")
    return math.log(x) if base is None else math.log(x, base)


def _sqrt(x):
    x = _scalar(x, "sqrt")
    if x < 0:
        raise ValueError(f"sqrt: аргумент должен быть >= 0, получено {x}")
    return math.sqrt(x)


def _flatten(v):
    if isinstance(v, list):
        for item in v:
            yield from _flatten(item)
    else:
        yield v


def _min(*xs):
    vals = [y for x in xs for y in _flatten(x)]
    if not vals:
        raise ValueError("min: нет аргументов")
    return float(min(vals))


def _max(*xs):
    vals = [y for x in xs for y in _flatten(x)]
    if not vals:
        raise ValueError("max: нет аргументов")
    return float(max(vals))


def _sum(arr):
    if not isinstance(arr, list):
        raise ValueError("sum: ожидался массив")
    return float(sum(_flatten(arr)))


def _len(arr):
    if not isinstance(arr, list):
        raise ValueError("len: ожидался массив")
    return float(len(arr))


def _create_array(*dims, value=0.0):
    if not dims:
        raise ValueError("create_array: нужно указать хотя бы одну размерность")
    sizes = [_int_arg(d, "create_array") for d in dims]
    value = _scalar(value, "create_array")

    def build(level):
        if level == len(sizes):
            return float(value)
        return [build(level + 1) for _ in range(sizes[level])]

    return build(0)


# --- случайные функции (используют сид модели)
def _uniform(rng, a=0.0, b=1.0):
    """Равномерное распределение на [a, b]."""
    a, b = _scalar(a, "uniform"), _scalar(b, "uniform")
    if a > b:
        raise ValueError(f"uniform: должно быть a <= b, получено a={a:g}, b={b:g}")
    return rng.uniform(a, b)


def _normal(rng, mu=0.0, sigma=1.0):
    """Нормальное распределение N(mu, sigma^2)."""
    sigma = _scalar(sigma, "normal")
    if sigma < 0:
        raise ValueError("normal: sigma должна быть >= 0")
    return rng.gauss(_scalar(mu, "normal"), sigma)


def _bernoulli(rng, p):
    p = _scalar(p, "bernoulli")
    if not 0 <= p <= 1:
        raise ValueError(f"bernoulli: p должна быть в [0, 1], получено {p}")
    return 1.0 if rng.random() < p else 0.0


def _exponential(rng, mean):
    """Экспоненциальное распределение со средним mean."""
    mean = _scalar(mean, "exponential")
    if mean <= 0:
        raise ValueError("exponential: mean должно быть > 0")
    return rng.expovariate(1.0 / mean)


def _poisson_value(rng, lam):
    lam = _scalar(lam, "poisson")
    if lam < 0:
        raise ValueError("poisson: lambda должна быть >= 0")
    # для больших lambda — нормальное приближение, иначе алгоритм Кнута
    if lam > 500:
        return float(max(0, round(rng.gauss(lam, math.sqrt(lam)))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return float(k)
        k += 1


BUILTINS: dict[str, Builtin] = {
    "floor": Builtin(_unary(math.floor, "floor")),
    "ceil": Builtin(_unary(math.ceil, "ceil")),
    "round": Builtin(_unary(round, "round")),
    "abs": Builtin(_unary(abs, "abs")),
    "sqrt": Builtin(_sqrt),
    "exp": Builtin(_unary(math.exp, "exp")),
    "log": Builtin(_log),
    "min": Builtin(_min),
    "max": Builtin(_max),
    "sum": Builtin(_sum),
    "len": Builtin(_len),
    "create_array": Builtin(_create_array),
    "uniform": Builtin(_uniform, random=True),
    "normal": Builtin(_normal, random=True),
    "bernoulli": Builtin(_bernoulli, random=True),
    "exponential": Builtin(_exponential, random=True),
    "poisson": Builtin(_poisson_value, random=True, aliases=(("lambda", "lam"),)),
}


# ------------------------------------------------- распределения для arrival
# Функция возвращает интервал (в днях) до следующего прихода события.
def _poisson_arrival(rng, lam, per=1.0):
    lam, per = _scalar(lam, "poisson"), _scalar(per, "poisson")
    if per <= 0:
        raise ValueError("poisson: per должно быть > 0")
    if lam < 0:
        raise ValueError("poisson: lambda должна быть >= 0")
    rate = lam / per  # событий в день
    if rate == 0:
        return math.inf
    return rng.expovariate(rate)


ARRIVALS: dict[str, Builtin] = {
    "poisson": Builtin(_poisson_arrival, random=True, aliases=(("lambda", "lam"),)),
}

# имена, которые нельзя использовать как переменные
# TIME — текущее моделируемое время (в днях), inf — бесконечность
RESERVED_NAMES = {"TIME", "inf"}
