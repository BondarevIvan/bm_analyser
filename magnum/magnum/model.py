"""Загрузка (валидация + компиляция) модели и её запуск.

Выражения компилируются в замыкания Python: `fn(frame) -> value`.
Значение — float или вложенный list из float (массив).
"""
from __future__ import annotations

import copy
import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import nodes as N
from .builtins import ARRIVALS, BUILTINS, RESERVED_NAMES
from .errors import MagnumControlError, MagnumRuntimeError, MagnumValidationError
from .parser import parse
from .sampling import make_recorder

EPS = 1e-9  # допуск сравнения моментов времени (в днях)
_NO_RETURN = object()

Value = Any  # float | list


# =============================================================== утилиты
def _copy(v):
    return copy.deepcopy(v) if isinstance(v, list) else v


def _rt_error(node, msg):
    return MagnumRuntimeError(msg, node.pos.line, node.pos.column)


def _val_error(node, msg):
    return MagnumValidationError(msg, node.pos.line, node.pos.column)


def _num(v, node, what="операнд"):
    if isinstance(v, list):
        raise _rt_error(node, f"{what}: арифметика над массивами не поддерживается (используйте индексацию)")
    return v


def _fmt(v):
    """Числа с целым значением печатаются без .0: 3 вместо 3.0."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_fmt(x) for x in v]
    return v


def _truth(v, node):
    if isinstance(v, list):
        raise _rt_error(node, "условие должно быть числом, получен массив")
    return v != 0


def _get_index(container, i, node):
    if not isinstance(container, list):
        raise _rt_error(node, "индексация применяется только к массивам")
    if isinstance(i, list) or i != int(i):
        raise _rt_error(node, f"индекс должен быть целым числом, получено {i!r}")
    i = int(i)
    if not 0 <= i < len(container):
        raise _rt_error(node, f"индекс {i} вне диапазона [0, {len(container) - 1}]")
    return i


def _shape(v):
    if not isinstance(v, list):
        return ()
    if not v:
        return (0,)
    sub = {_shape(x) for x in v}
    return (len(v), *next(iter(sub))) if len(sub) == 1 else (len(v), None)


def _flatten(v):
    if isinstance(v, list):
        for x in v:
            yield from _flatten(x)
    else:
        yield v


def _normalize_value(v, name):
    """Значение контрола от пользователя -> float / вложенный list из float."""
    if hasattr(v, "tolist"):  # numpy
        v = v.tolist()
    if isinstance(v, bool):
        raise MagnumControlError(f"контрол '{name}': bool не поддерживается, ожидалось число")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_normalize_value(x, name) for x in v]
    raise MagnumControlError(f"контрол '{name}': неподдерживаемый тип {type(v).__name__}")


# ================================================================ кадры
class Frame:
    """Контекст выполнения: локальные переменные (параметры события / функции),
    мгновенные переменные instant[...] и ссылка на симуляцию (глобальные переменные, время, rng)."""

    __slots__ = ("locals", "instant", "sim")

    def __init__(self, locals_, instant, sim):
        self.locals = locals_
        self.instant = instant
        self.sim = sim


@dataclass
class _Scope:
    kind: str                      # "event" | "function" | "const" | "trigger"
    owner: str                     # имя события/функции (для сообщений)
    locals: frozenset = frozenset()
    readonly_locals: bool = False  # параметры события нельзя переприсваивать
    allow_instant: bool = False
    allow_random: bool = True


# ============================================================ компилятор
class _Compiler:
    def __init__(self, model: "Model"):
        self.model = model

    # ---------------------------------------------------------- выражения
    def expr(self, node, scope: _Scope) -> Callable[[Frame], Value]:
        m = getattr(self, "_e_" + type(node).__name__)
        return m(node, scope)

    def _e_Number(self, node, scope):
        v = node.value
        return lambda f: v

    def _e_Var(self, node, scope):
        name = node.name
        if name in scope.locals:
            def get_local(f):
                try:
                    return f.locals[name]
                except KeyError:
                    raise _rt_error(node, f"локальная переменная '{name}' используется до присваивания") from None
            return get_local
        if name == "TIME":  # текущее моделируемое время, в днях
            return lambda f: f.sim.time
        if name == "inf":
            return lambda f: math.inf

        def get_global(f):
            try:
                return f.sim.globals[name]
            except KeyError:
                raise _rt_error(node, f"переменная '{name}' не определена") from None
        return get_global

    def _e_Instant(self, node, scope):
        if not scope.allow_instant:
            raise _val_error(node, f"instant[{node.name}] допустим только в секциях do/choose/emit события")
        name = node.name

        def get_instant(f):
            try:
                return f.instant[name]
            except KeyError:
                raise _rt_error(node, f"мгновенная переменная instant[{name}] используется до присваивания") from None
        return get_instant

    def _e_Index(self, node, scope):
        target, index = self.expr(node.target, scope), self.expr(node.index, scope)

        def get_item(f):
            c = target(f)
            return c[_get_index(c, index(f), node)]
        return get_item

    def _e_Neg(self, node, scope):
        x = self.expr(node.operand, scope)
        return lambda f: -_num(x(f), node)

    def _e_Compare(self, node, scope):
        l, r = self.expr(node.left, scope), self.expr(node.right, scope)
        op = node.op
        if op in ("==", "!="):  # равенство допустимо и для массивов
            eq = op == "=="
            return lambda f: 1.0 if (l(f) == r(f)) == eq else 0.0
        cmp = {"<": float.__lt__, "<=": float.__le__, ">": float.__gt__, ">=": float.__ge__}[op]

        def compare(f):
            a, b = _num(l(f), node), _num(r(f), node)
            return 1.0 if cmp(float(a), float(b)) else 0.0
        return compare

    def _e_Logic(self, node, scope):
        l, r = self.expr(node.left, scope), self.expr(node.right, scope)
        if node.op == "and":  # ленивое вычисление: правая часть считается только при необходимости
            return lambda f: 1.0 if (_truth(l(f), node) and _truth(r(f), node)) else 0.0
        return lambda f: 1.0 if (_truth(l(f), node) or _truth(r(f), node)) else 0.0

    def _e_Not(self, node, scope):
        x = self.expr(node.operand, scope)
        return lambda f: 0.0 if _truth(x(f), node) else 1.0

    def _if(self, node, scope):
        """if(условие, если_истина, если_ложь) — вычисляется только выбранная ветка."""
        if node.kwargs or len(node.args) != 3:
            raise _val_error(node, "if ожидает ровно 3 позиционных аргумента: if(условие, если_истина, если_ложь)")
        c, a, b = (self.expr(x, scope) for x in node.args)
        return lambda f: a(f) if _truth(c(f), node) else b(f)

    def _e_ArrayLit(self, node, scope):
        items = [self.expr(i, scope) for i in node.items]
        return lambda f: [_copy(i(f)) for i in items]

    def _e_BinOp(self, node, scope):
        l, r, op = self.expr(node.left, scope), self.expr(node.right, scope), node.op
        if op == "+":
            return lambda f: _num(l(f), node) + _num(r(f), node)
        if op == "-":
            return lambda f: _num(l(f), node) - _num(r(f), node)
        if op == "*":
            return lambda f: _num(l(f), node) * _num(r(f), node)
        if op == "/":
            def div(f):
                a, b = _num(l(f), node), _num(r(f), node)
                if b == 0:
                    raise _rt_error(node, "деление на ноль")
                return a / b
            return div
        if op == "^":
            def power(f):
                a, b = _num(l(f), node), _num(r(f), node)
                try:
                    res = a ** b
                except ZeroDivisionError:
                    raise _rt_error(node, "0 в отрицательной степени") from None
                except OverflowError:
                    raise _rt_error(node, "переполнение при возведении в степень") from None
                if isinstance(res, complex):
                    raise _rt_error(node, f"{a}^{b} не является вещественным числом")
                return float(res)
            return power
        raise AssertionError(op)  # pragma: no cover

    def _e_Call(self, node, scope):
        if node.name == "if" and not node.user:
            return self._if(node, scope)
        args = [self.expr(a, scope) for a in node.args]
        kwargs = [(k, self.expr(v, scope)) for k, v in node.kwargs]
        if node.user:
            fdef = self.model._function_defs.get(node.name)
            if fdef is None:
                hint = " (это встроенная функция — вызывайте без call)" if node.name in BUILTINS else ""
                raise _val_error(node, f"функция '{node.name}' не объявлена в модели{hint}")
            if kwargs:
                raise _val_error(node, f"функции модели '{node.name}' передаются только позиционные аргументы")
            if len(args) != len(fdef.params):
                raise _val_error(node, f"функция '{node.name}' ожидает {len(fdef.params)} аргумент(ов), передано {len(args)}")
            functions = self.model._functions  # заполняется после компиляции всех функций

            def call_user(f):
                return functions[node.name](f.sim, [a(f) for a in args], node)
            return call_user

        b = BUILTINS.get(node.name)
        if b is None:
            hint = " (функцию модели вызывайте как `call имя(...)`)" if node.name in self.model._function_defs else ""
            raise _val_error(node, f"неизвестная встроенная функция '{node.name}'{hint}")
        if b.random and not scope.allow_random:
            raise _val_error(node, f"случайная функция '{node.name}' недопустима в controls")
        err = b.check_call(len(args), [k for k, _ in kwargs])
        if err:
            raise _val_error(node, f"{node.name}: неверные аргументы ({err})")

        def call_builtin(f):
            a = [x(f) for x in args]
            kw = {k: v(f) for k, v in kwargs}
            try:
                return b.call(f.sim.rng, a, kw)
            except (ValueError, TypeError, OverflowError) as e:
                raise _rt_error(node, str(e)) from None
        return call_builtin

    # --------------------------------------------------------- присваивание
    def statement(self, node, scope: _Scope) -> Callable[[Frame], None]:
        if isinstance(node, N.Print):
            return self.print_(node, scope)
        return self.assign(node, scope)

    def print_(self, node: N.Print, scope: _Scope) -> Callable[[Frame], None]:
        """print(...) — отладочный вывод, как print в Python. Отключается run(debug=False)."""
        parts = [(lambda f, s=a: s) if isinstance(a, str) else self.expr(a, scope) for a in node.args]

        def do_print(f):
            if f.sim.debug:
                print(*(_fmt(p(f)) for p in parts))
        return do_print

    def assign(self, node: N.Assign, scope: _Scope) -> Callable[[Frame], None]:
        value = self.expr(node.value, scope)
        # разворачиваем цепочку a[i][j]
        indices, base = [], node.target
        while isinstance(base, N.Index):
            indices.append(base.index)
            base = base.target
        indices.reverse()

        if isinstance(base, N.Instant) and not scope.allow_instant:
            raise _val_error(base, "instant[...] допустим только в событиях")
        if isinstance(base, N.Var):
            if base.name in RESERVED_NAMES:
                raise _val_error(base, f"'{base.name}' — зарезервированное имя, присваивать нельзя")
            if base.name in scope.locals and scope.readonly_locals:
                raise _val_error(base, f"параметр события '{base.name}' нельзя изменять (используйте instant[...])")

        if not indices:
            if isinstance(base, N.Instant):
                name = base.name

                def set_instant(f):
                    f.instant[name] = _copy(value(f))
                return set_instant
            name = base.name
            if scope.kind == "function":
                def set_local(f):
                    f.locals[name] = _copy(value(f))
                return set_local

            def set_global(f):
                f.sim.globals[name] = _copy(value(f))
                f.sim.changed.add(name)
            return set_global

        container = self.expr(base, scope)
        idx = [self.expr(i, scope) for i in indices]
        global_name = base.name if (isinstance(base, N.Var) and scope.kind == "event"
                                    and base.name not in scope.locals) else None

        def set_item(f):
            c = container(f)
            for i in idx[:-1]:
                c = c[_get_index(c, i(f), node)]
            c[_get_index(c, idx[-1](f), node)] = _copy(value(f))
            if global_name is not None:
                f.sim.changed.add(global_name)
        return set_item


# ======================================================= скомпилированное
@dataclass
class _CEvent:
    name: str
    params: tuple[str, ...]
    do: list
    triggers: list           # [(kind, payload)]
    choose: list             # [[(prob_fn, emit_fn, node)]]
    emits: list
    node: N.Event = field(repr=False)


@dataclass
class _Limit:
    lower: float
    lower_closed: bool
    upper: float
    upper_closed: bool

    def contains(self, x: float) -> bool:
        lo_ok = x >= self.lower if self.lower_closed else x > self.lower
        hi_ok = x <= self.upper if self.upper_closed else x < self.upper
        return lo_ok and hi_ok

    def __str__(self):
        def fmt(v):
            return "inf" if v == math.inf else "-inf" if v == -math.inf else f"{v:g}"
        return f"{'[' if self.lower_closed else '('}{fmt(self.lower)}, {fmt(self.upper)}{']' if self.upper_closed else ')'}"


class _ConstSim:
    """Псевдо-симуляция для вычисления значений по умолчанию и границ."""

    def __init__(self, globals_):
        self.globals = globals_
        self.time = 0.0
        self.rng = None


# ================================================================= модель
class Model:
    """Загруженная (разобранная, проверенная и скомпилированная) модель."""

    def __init__(self, definition: N.ModelDef, source: str | None = None):
        self.definition = definition
        self.source = source
        self.name = definition.name
        self._function_defs = {}
        self._functions: dict[str, Callable] = {}
        self.events: dict[str, _CEvent] = {}
        self.defaults: dict[str, Value] = {}
        self.limits: dict[str, _Limit] = {}
        self._build()

    # -------------------------------------------------------- построение
    def _build(self):
        d = self.definition
        comp = _Compiler(self)

        # 1. controls
        if len(d.controls) > 1:
            raise _val_error(d.controls[1], "секция controls должна быть одна")
        ctrl = d.controls[0] if d.controls else N.Controls((), ())
        const_sim = _ConstSim(self.defaults)
        const_scope = _Scope("const", "controls", allow_random=False)
        for item in ctrl.defaults:
            if item.name in RESERVED_NAMES:
                raise _val_error(item, f"'{item.name}' — зарезервированное имя")
            if item.name in self.defaults:
                raise _val_error(item, f"контрол '{item.name}' объявлен дважды")
            try:
                self.defaults[item.name] = comp.expr(item.value, const_scope)(Frame({}, {}, const_sim))
            except MagnumRuntimeError as e:
                raise MagnumValidationError(f"значение контрола '{item.name}': {e.message}", e.line, e.column) from None
        for lim in ctrl.limits:
            if lim.name not in self.defaults:
                raise _val_error(lim, f"limits: контрол '{lim.name}' не объявлен в default")
            if lim.name in self.limits:
                raise _val_error(lim, f"limits: ограничение на '{lim.name}' задано дважды")
            lo = comp.expr(lim.lower, const_scope)(Frame({}, {}, const_sim))
            hi = comp.expr(lim.upper, const_scope)(Frame({}, {}, const_sim))
            if isinstance(lo, list) or isinstance(hi, list):
                raise _val_error(lim, "границы limits должны быть числами")
            if lo > hi:
                raise _val_error(lim, f"limits: нижняя граница {lo} больше верхней {hi}")
            self.limits[lim.name] = _Limit(lo, lim.lower_closed, hi, lim.upper_closed)
        for name, value in self.defaults.items():
            if name in self.limits:
                self._check_limit(name, value, MagnumValidationError, "значение по умолчанию")

        # 2. функции (сначала собираем сигнатуры — функции могут вызывать друг друга)
        for fdef in d.functions:
            if fdef.name in self._function_defs:
                raise _val_error(fdef, f"функция '{fdef.name}' объявлена дважды")
            if fdef.name in BUILTINS or fdef.name == "if":
                raise _val_error(fdef, f"'{fdef.name}' — имя встроенной функции")
            self._check_params(fdef, fdef.params)
            self._function_defs[fdef.name] = fdef
        for fdef in d.functions:
            self._functions[fdef.name] = self._compile_function(comp, fdef)

        # 3. события
        for ev in d.events:
            if ev.name in self.events:
                raise _val_error(ev, f"событие '{ev.name}' объявлено дважды")
            self.events[ev.name] = None  # резервируем имя (для проверки emit)
        for ev in d.events:
            self.events[ev.name] = self._compile_event(comp, ev)

    @staticmethod
    def _check_params(node, params):
        if len(set(params)) != len(params):
            raise _val_error(node, "повторяющиеся имена параметров")
        for p in params:
            if p in RESERVED_NAMES:
                raise _val_error(node, f"'{p}' — зарезервированное имя, нельзя использовать как параметр")

    def _compile_function(self, comp, fdef: N.Function):
        assigned = set()
        for st in fdef.body:
            if isinstance(st, N.Assign):
                base = st.target
                while isinstance(base, N.Index):
                    base = base.target
                if isinstance(base, N.Var):
                    assigned.add(base.name)
        if not any(isinstance(st, N.Return) for st in fdef.body):
            raise _val_error(fdef, f"в функции '{fdef.name}' нет return")
        scope = _Scope("function", fdef.name, locals=frozenset(fdef.params) | frozenset(assigned))
        body = []
        for st in fdef.body:
            if isinstance(st, N.Return):
                fn = comp.expr(st.value, scope)
                body.append(lambda f, fn=fn: _copy(fn(f)))
            else:
                a = comp.statement(st, scope)
                body.append(lambda f, a=a: (a(f), _NO_RETURN)[1])
        params, name = fdef.params, fdef.name

        def run(sim, args, call_node):
            frame = Frame(dict(zip(params, args)), None, sim)
            for st in body:
                r = st(frame)
                if r is not _NO_RETURN:
                    return r
            raise _rt_error(call_node, f"функция '{name}' завершилась без return")  # pragma: no cover
        return run

    def _compile_emit(self, comp, em: N.Emit, scope):
        target = self.events.get(em.event, False)
        if target is False:
            raise _val_error(em, f"emit: событие '{em.event}' не объявлено")
        tdef = next(e for e in self.definition.events if e.name == em.event)
        if len(em.args) != len(tdef.params):
            raise _val_error(em, f"emit: событие '{em.event}' ожидает {len(tdef.params)} аргумент(ов), передано {len(em.args)}")
        args = [comp.expr(a, scope) for a in em.args]
        delay = comp.expr(em.delay, scope) if em.delay is not None else None
        name = em.event

        def do_emit(f):
            values = tuple(_copy(a(f)) for a in args)
            dt = 0.0
            if delay is not None:
                dt = delay(f)
                if isinstance(dt, list) or not (0 <= dt < math.inf):
                    raise _rt_error(em, f"задержка after должна быть конечным числом >= 0, получено {dt!r}")
            f.sim.schedule(f.sim.time + dt, name, values)
        return do_emit

    def _compile_event(self, comp, ev: N.Event) -> _CEvent:
        self._check_params(ev, ev.params)
        if ev.triggers and ev.params:
            raise _val_error(ev, f"событие '{ev.name}' с параметрами не может иметь trigger (его можно только emit-ить)")
        scope = _Scope("event", ev.name, locals=frozenset(ev.params), readonly_locals=True, allow_instant=True)
        trig_scope = _Scope("trigger", ev.name)

        triggers = []
        for t in ev.triggers:
            if isinstance(t, N.OnStart):
                triggers.append(("start", None))
            elif isinstance(t, N.Every):
                triggers.append(("every", comp.expr(t.period, trig_scope), t))
            else:
                dist = ARRIVALS.get(t.distribution)
                if dist is None:
                    raise _val_error(t, f"arrival: неизвестное распределение '{t.distribution}' (доступны: {', '.join(ARRIVALS)})")
                err = dist.check_call(len(t.args), [k for k, _ in t.kwargs])
                if err:
                    raise _val_error(t, f"arrival {t.distribution}: неверные аргументы ({err})")
                args = [comp.expr(a, trig_scope) for a in t.args]
                kwargs = [(k, comp.expr(v, trig_scope)) for k, v in t.kwargs]
                triggers.append(("arrival", (dist, args, kwargs), t))

        do = [comp.statement(a, scope) for a in ev.do]
        choose = [[(comp.expr(b.probability, scope), [self._compile_emit(comp, e, scope) for e in b.emits], b)
                   for b in block]
                  for block in ev.choose]
        emits = [self._compile_emit(comp, e, scope) for e in ev.emits]
        return _CEvent(ev.name, ev.params, do, triggers, choose, emits, ev)

    # ------------------------------------------------------------ контролы
    def _check_limit(self, name, value, exc, what):
        lim = self.limits.get(name)
        if lim is None:
            return
        for x in _flatten(value):
            if not lim.contains(x):
                raise exc(f"{what} '{name}' = {x:g} вне допустимого диапазона {lim}")

    def resolve_controls(self, controls: Mapping[str, Any] | None) -> dict[str, Value]:
        """Значения по умолчанию, перекрытые переданными контролами, с проверкой limits."""
        result = {k: copy.deepcopy(v) for k, v in self.defaults.items()}
        for name, raw in (controls or {}).items():
            if name not in self.defaults:
                raise MagnumControlError(f"неизвестный контрол '{name}' (доступны: {', '.join(self.defaults) or 'нет'})")
            value = _normalize_value(raw, name)
            if _shape(value) != _shape(self.defaults[name]):
                raise MagnumControlError(
                    f"контрол '{name}': форма {_shape(value)} не совпадает с формой по умолчанию {_shape(self.defaults[name])}")
            self._check_limit(name, value, MagnumControlError, "контрол")
            result[name] = value
        return result

    # --------------------------------------------------------------- запуск
    def run(self, controls: Mapping[str, Any] | None = None, seed: int | None = 0, horizon: float = 365,
            sampling: str | float = "1d", max_events: int = 10_000_000, debug: bool = True) -> dict:
        """Запустить модель.

        :param controls: значения контролов (перекрывают default; проверяются по limits)
        :param seed: сид генератора случайных чисел
        :param horizon: горизонт моделирования в днях
        :param sampling: политика снятия состояния: длительность ('1d', '6h', '30m', ...) —
            снимок всех переменных с этим шагом; 'on_change' — запись переменной при её изменении
        :param max_events: предохранитель от бесконечной генерации событий
        :param debug: выполнять ли print(...) из модели (False — для массовых прогонов)
        :return: словарь, см. :class:`Simulation.result`
        """
        sim = Simulation(self, controls, seed, horizon, sampling, max_events)
        sim.debug = debug
        return sim.run()

    def __repr__(self):
        return f"Model({self.name!r}, controls={list(self.defaults)}, events={list(self.events)}, functions={list(self._functions)})"


# ============================================================== симуляция
@dataclass
class _Occurrence:
    event: str
    args: tuple
    trigger: int | None = None      # индекс триггера, породившего событие (None — emit / on start)
    anchor: float = 0.0             # для every: время = anchor + k * period (без накопления ошибки)
    k: int = 0
    period: float = 0.0


_BOOTSTRAP = "__bootstrap__"


class Simulation:
    def __init__(self, model: Model, controls, seed, horizon, sampling, max_events):
        if horizon < 0:
            raise ValueError("horizon должен быть >= 0")
        self.model = model
        self.seed = seed
        self.rng = random.Random(seed)
        self.horizon = float(horizon)
        self.max_events = max_events
        self.debug = True
        self.sampling = sampling
        self.recorder = make_recorder(sampling, self.horizon)
        self.changed: set[str] = set()  # глобальные переменные, изменённые текущим событием
        self.controls = model.resolve_controls(controls)
        self.globals: dict[str, Value] = copy.deepcopy(self.controls)
        self.time = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self.event_counts = {name: 0 for name in model.events}

    def schedule(self, t: float, event: str, args: tuple = (), **occ_kw):
        heapq.heappush(self._heap, (t, next(self._seq), _Occurrence(event, args, **occ_kw)))

    # --------------------------------------------------------------- триггеры
    def _schedule_trigger(self, ev: _CEvent, ti: int, prev: _Occurrence | None):
        trig = ev.triggers[ti]
        frame = Frame({}, None, self)
        if trig[0] == "every":
            period = trig[1](frame)
            if isinstance(period, list) or not (0 < period < math.inf):
                raise _rt_error(trig[2], f"every: период должен быть конечным числом > 0, получено {period!r}")
            if prev is not None and prev.period == period:
                anchor, k = prev.anchor, prev.k + 1
            else:
                anchor, k = self.time, 1
            self.schedule(anchor + k * period, ev.name, (), trigger=ti, anchor=anchor, k=k, period=period)
        elif trig[0] == "arrival":
            dist, args, kwargs = trig[1]
            try:
                dt = dist.call(self.rng, [a(frame) for a in args], {k: v(frame) for k, v in kwargs})
            except (ValueError, TypeError) as e:
                raise _rt_error(trig[2], str(e)) from None
            if dt < math.inf:
                self.schedule(self.time + dt, ev.name, (), trigger=ti)

    # ------------------------------------------------------------- событие
    def _process(self, occ: _Occurrence):
        if occ.event == _BOOTSTRAP:
            for ev in self.model.events.values():
                for ti, trig in enumerate(ev.triggers):
                    if trig[0] != "start":
                        self._schedule_trigger(ev, ti, None)
            return
        ev = self.model.events[occ.event]
        # 1. trigger — планируем следующее срабатывание
        if occ.trigger is not None:
            self._schedule_trigger(ev, occ.trigger, occ)
        frame = Frame(dict(zip(ev.params, occ.args)), {}, self)
        # 2. do
        for st in ev.do:
            st(frame)
        # 3. choose
        for block in ev.choose:
            u = self.rng.random()
            acc = 0.0
            chosen = None
            for prob_fn, emit_fns, node in block:
                p = prob_fn(frame)
                if isinstance(p, list) or not (0.0 <= p <= 1.0):
                    raise _rt_error(node, f"choose: вероятность должна быть числом в [0, 1], получено {p!r}")
                acc += p
                if chosen is None and u < acc:
                    chosen = emit_fns
            if acc > 1.0 + 1e-9:
                raise _rt_error(block[0][2], f"choose: сумма вероятностей {acc:g} больше 1")
            if chosen is not None:
                for emit_fn in chosen:
                    emit_fn(frame)
        # 4. emit
        for em in ev.emits:
            em(frame)
        self.event_counts[ev.name] += 1

    # ---------------------------------------------------------------- цикл
    def _snapshot(self):
        return {k: _copy(v) for k, v in self.globals.items()}

    def run(self) -> dict:
        rec = self.recorder
        for ev in self.model.events.values():
            for trig in ev.triggers:
                if trig[0] == "start":
                    self.schedule(0.0, ev.name)
        self.schedule(0.0, _BOOTSTRAP)
        rec.start(self.globals)

        processed = 0
        while self._heap:
            t = self._heap[0][0]
            if t > self.horizon + EPS:
                break
            _, _, occ = heapq.heappop(self._heap)
            rec.before_event(t, self.globals)
            self.time = t
            self.changed.clear()
            self._process(occ)
            rec.after_event(t, self.globals, self.changed)
            processed += 1
            if processed > self.max_events:
                raise MagnumRuntimeError(
                    f"превышено max_events={self.max_events} (t={t:g}); возможна бесконечная цепочка emit без задержки")
        rec.finish(self.globals)
        self.time = self.horizon

        return {
            "model": self.model.name,
            "seed": self.seed,
            "horizon": self.horizon,
            "sampling": self.sampling,
            "controls": copy.deepcopy(self.controls),
            "variables": rec.series,
            "final": self._snapshot(),
            "event_counts": dict(self.event_counts),
        }


# ============================================================ публичное API
def load_model(source: str) -> Model:
    """Загрузить модель из строки."""
    return Model(parse(source), source)


def load_model_file(path: str | Path, encoding: str = "utf-8") -> Model:
    """Загрузить модель из файла."""
    return load_model(Path(path).read_text(encoding=encoding))


def run_model(model: Model | str, controls: Mapping[str, Any] | None = None, seed: int | None = 0,
              horizon: float = 365, sampling: str | float = "1d", max_events: int = 10_000_000,
              debug: bool = True) -> dict:
    """Запустить модель (объект Model или текст модели). См. :meth:`Model.run`."""
    if isinstance(model, str):
        model = load_model(model)
    return model.run(controls=controls, seed=seed, horizon=horizon, sampling=sampling, max_events=max_events,
                     debug=debug)
