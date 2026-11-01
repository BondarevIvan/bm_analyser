"""Описание модели на Python: класс-наследник Model, параметры param(...) и события-методы с триггерами.

    import magnum as mg

    class Kitchen(mg.Model):
        cooks = mg.param(2, low=1, high=10)              # параметр: значение по умолчанию и границы
        order_rate = mg.param(10.0, low=0)               # заказов в час

        @mg.on_start
        def init(self):
            self.orders = mg.Queue()
            self.idle = self.cooks
            self.ready = 0

        @mg.arrival(rate=lambda m: m.order_rate, per=mg.HOUR)
        def new_order(self):
            self.orders.push(self.time)

        @mg.when(lambda m: m.idle > 0 and len(m.orders) > 0)
        def start_cooking(self):
            self.orders.pop()
            self.idle -= 1
            self.emit(self.dish_ready, after=self.rng.exponential(7 * mg.MINUTE))

        def dish_ready(self):
            self.idle += 1
            self.ready += 1

    result = Kitchen.run(controls={"cooks": 3}, seed=1, horizon=30)
"""
from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Callable

import torch

from .errors import MagnumControlError, MagnumValidationError
from .queues import Queue, Resource, plain
from .rand import _grad
from .units import DAY

_TRIGGERS_ATTR = "__magnum_triggers__"


# ============================================================ параметры
def _to_tensor(v, name, exc=MagnumControlError) -> torch.Tensor:
    """Значение параметра -> torch.Tensor (число, вложенный список, numpy-массив или тензор)."""
    if isinstance(v, torch.Tensor):
        t = v
    else:
        if isinstance(v, bool):
            raise exc(f"параметр '{name}': bool не поддерживается, ожидалось число")
        try:
            t = torch.as_tensor(v)
        except (TypeError, ValueError, RuntimeError) as e:
            raise exc(f"параметр '{name}': неподдерживаемый тип или форма значения — ожидалось число "
                      f"или прямоугольный массив чисел ({e})") from None
    if t.dtype == torch.bool or t.is_complex():
        raise exc(f"параметр '{name}': неподдерживаемый тип {t.dtype}")
    return t


def _shape_str(t) -> str:
    return "число" if t.dim() == 0 else "×".join(map(str, t.shape))


class param:  # noqa: N801 — читается как объявление
    """Параметр модели — тензор torch (число или массив). Его подбирают, перекрывают при запуске и обучают.

    :param default: значение по умолчанию — число, (вложенный) список, numpy-массив или torch.Tensor
    :param low, high: границы (None — без границы); для массива — для каждого элемента
    :param low_open, high_open: граница не включается (low < x вместо low <= x)
    :param integer: только целые значения (тензор int64, не обучается); по умолчанию — если default целочисленный
    :param doc: описание

    В модели параметр — torch.Tensor: вещественный float64 (или тензор, переданный в controls, как есть —
    с requires_grad), целый int64. Подробнее об обучении — magnum.train.
    """

    def __init__(self, default, low=None, high=None, *, low_open=False, high_open=False, integer=None, doc=""):
        self.name = None
        self.low, self.high, self.low_open, self.high_open = low, high, low_open, high_open
        t = _to_tensor(default, "?", MagnumValidationError)
        self.integer = (not t.is_floating_point()) if integer is None else bool(integer)
        self.doc = doc
        self.default = t.detach().to(torch.int64 if self.integer else torch.float64).clone()
        if low is not None and high is not None and low > high:
            raise MagnumValidationError(f"param: нижняя граница {low} больше верхней {high}")

    def __set_name__(self, owner, name):
        self.name = name   # проверка значения по умолчанию — в Model.__init_subclass__ (в Python < 3.12
                           # исключение из __set_name__ оборачивается в RuntimeError)

    @property
    def shape(self) -> tuple:
        return tuple(self.default.shape)

    def bounds_str(self) -> str:
        lo = "-inf" if self.low is None else f"{self.low:g}"
        hi = "inf" if self.high is None else f"{self.high:g}"
        return f"{'(' if self.low_open or self.low is None else '['}{lo}, {hi}{')' if self.high_open or self.high is None else ']'}"

    def validate(self, value, exc=MagnumControlError, what="параметр") -> torch.Tensor:
        """Проверить значение и вернуть тензор параметра. Тензор с requires_grad возвращается как есть
        (без копии), чтобы градиенты дошли до него."""
        t = _to_tensor(value, self.name, exc)
        if tuple(t.shape) != self.shape:
            raise exc(f"{what} '{self.name}': форма {_shape_str(t)} не совпадает с формой по умолчанию "
                      f"{_shape_str(self.default)}")
        d = t.detach()
        if self.integer:
            if t.requires_grad:
                raise exc(f"{what} '{self.name}': целочисленный параметр не может быть обучаемым (requires_grad)")
            if d.is_floating_point():
                bad = d != d.round()
                if bad.any():
                    raise exc(f"{what} '{self.name}' = {d[bad].flatten()[0].item():g}: ожидается целое число")
            t = d.to(torch.int64).clone()
        elif t.requires_grad:
            if not t.is_floating_point():
                raise exc(f"{what} '{self.name}': обучаемый параметр должен быть вещественным тензором")
        else:
            t = d.to(torch.float64).clone()
        df = d.to(torch.float64)
        ok = ~torch.isnan(df)
        if self.low is not None:
            ok &= (df > self.low) if self.low_open else (df >= self.low)
        if self.high is not None:
            ok &= (df < self.high) if self.high_open else (df <= self.high)
        if not ok.all():
            x = df[~ok].flatten()[0].item()
            raise exc(f"{what} '{self.name}' = {x:g} вне допустимого диапазона {self.bounds_str()}")
        return t

    def clamp_(self, t: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Вернуть тензор в границы на месте (после шага оптимизатора); открытые границы — с запасом eps."""
        if not t.is_floating_point():
            eps = 1
        lo = None if self.low is None else self.low + (eps if self.low_open else 0)
        hi = None if self.high is None else self.high - (eps if self.high_open else 0)
        if not t.is_floating_point():
            lo = None if lo is None else math.ceil(lo)
            hi = None if hi is None else math.floor(hi)
        if lo is None and hi is None:
            return t
        with torch.no_grad():
            t.clamp_(min=lo, max=hi)
        return t

    def __repr__(self):
        v = self.default.item() if self.default.dim() == 0 else f"tensor{list(self.default.shape)}"
        return f"param({v!r}, {self.bounds_str()})"

    def _renamed(self, name, default=None):
        """Копия параметра под другим именем (параметр подмодели во внешней модели), возможно с другим default."""
        p = copy.copy(self)
        p.name = name
        if default is not None:
            p.default = p.validate(default, MagnumValidationError, "значение по умолчанию").detach().clone()
        return p


def _final(v):
    if isinstance(v, torch.Tensor) and v.requires_grad:
        return v
    if isinstance(v, list):
        return [_final(x) for x in v]
    return plain(v)


def array(*shape, value=0.0) -> list:
    """Вложенный список заданной формы: array(7, 24) — 7 строк по 24 нуля."""
    if not shape:
        raise ValueError("array: нужна хотя бы одна размерность")

    def build(level):
        if level == len(shape):
            return value
        return [build(level + 1) for _ in range(int(shape[level]))]
    return build(0)


# ============================================================ подмодели
class sub:  # noqa: N801 — читается как объявление
    """Подмодель: экземпляр другой модели внутри этой, в общем времени и с общим rng.

        class Network(mg.Model):
            center = mg.sub(CoffeeQueue, count_masters=4)   # значения параметров подмодели по умолчанию
            mall = mg.sub(CoffeeQueue)

    В событиях: ``self.mall.clients``, ``self.emit(self.mall.register_client)``, ``when`` на переменные
    подмодели. Параметры подмодели становятся параметрами внешней модели: ``controls={"mall.count_masters": 3}``.
    Переменные подмодели в результате — под именами ``mall.<имя>``.
    """

    def __init__(self, model, **defaults):
        if not (isinstance(model, type) and issubclass(model, Model)) or model is Model:
            raise MagnumValidationError(f"sub: ожидался класс модели (наследник mg.Model), получено {model!r}")
        for k in defaults:
            if k not in model.params:
                raise MagnumValidationError(f"sub({model.__name__}): у модели нет параметра '{k}' "
                                            f"(есть: {', '.join(model.params) or 'нет'})")
        self.model = model
        self.defaults = defaults
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(f"подмодель '{self.name}' доступна только во время запуска модели") from None

    def __repr__(self):
        return f"sub({self.model.__name__})"


# ============================================================ триггеры
def _add_trigger(fn, trig):
    lst = getattr(fn, _TRIGGERS_ATTR, None)
    if lst is None:
        lst = []
        setattr(fn, _TRIGGERS_ATTR, lst)
    if trig is not None:
        lst.insert(0, trig)  # декораторы применяются снизу вверх — сохраняем порядок записи
    return fn


def event(fn):
    """Явно пометить метод как событие (без триггера: его вызывают через self.emit)."""
    return _add_trigger(fn, None)


def on_start(fn):
    """Событие в момент 0, до любых периодических."""
    return _add_trigger(fn, ("start",))


def every(period):
    """Событие каждые period дней: в моменты P, 2P, ... (period — число или функция модели m -> число)."""
    def deco(fn):
        return _add_trigger(fn, ("every", period))
    return deco


def arrival(rate, per=DAY):
    """Пуассоновский поток: в среднем rate событий за время per (rate — число, имя параметра или m -> число).
    Интенсивность пересчитывается после каждого прихода."""
    def deco(fn):
        return _add_trigger(fn, ("arrival", rate, per))
    return deco


def when(condition: Callable[[Any], Any]):
    """Событие срабатывает, пока условие истинно. Условие проверяется только после событий,
    изменивших прочитанные в нём атрибуты модели (и один раз после on_start)."""
    if not callable(condition):
        raise MagnumValidationError("when: условие должно быть функцией модели, например lambda m: m.queue_len > 0")

    def deco(fn):
        return _add_trigger(fn, ("when", condition))
    return deco


class _EventSpec:
    __slots__ = ("name", "fn", "triggers", "min_args", "max_args")

    def __init__(self, name, fn, triggers):
        self.name, self.fn, self.triggers = name, fn, triggers
        sig = inspect.signature(fn)
        ps = list(sig.parameters.values())[1:]  # без self
        if any(p.kind == p.VAR_POSITIONAL for p in ps):
            self.min_args, self.max_args = sum(p.default is p.empty and p.kind == p.POSITIONAL_OR_KEYWORD
                                               for p in ps), math.inf
        else:
            self.min_args = sum(p.default is p.empty for p in ps)
            self.max_args = len(ps)
        if triggers and self.min_args > 0:
            raise MagnumValidationError(f"событие '{name}' с параметрами не может иметь триггер "
                                        f"(его можно только вызвать через self.emit)")


# ============================================================ отслеживаемый список
class TrackedList(list):
    """Список в состоянии модели: изменение элемента видно движку (when, on_change)."""

    __slots__ = ("_notify",)

    def __init__(self, items=(), notify=None):
        super().__init__(_adopt_list_item(x, notify) for x in items)
        self._notify = notify

    def _ch(self):
        if self._notify is not None:
            self._notify()

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            v = [_adopt_list_item(x, self._notify) for x in v]
        else:
            v = _adopt_list_item(v, self._notify)
        super().__setitem__(i, v)
        self._ch()

    def __delitem__(self, i):
        super().__delitem__(i)
        self._ch()

    def append(self, v):
        super().append(_adopt_list_item(v, self._notify))
        self._ch()

    def extend(self, vs):
        super().extend(_adopt_list_item(x, self._notify) for x in vs)
        self._ch()

    def insert(self, i, v):
        super().insert(i, _adopt_list_item(v, self._notify))
        self._ch()

    def pop(self, *a):
        v = super().pop(*a)
        self._ch()
        return v

    def remove(self, v):
        super().remove(v)
        self._ch()

    def clear(self):
        super().clear()
        self._ch()

    def sort(self, *a, **kw):
        super().sort(*a, **kw)
        self._ch()

    def reverse(self):
        super().reverse()
        self._ch()

    def __iadd__(self, vs):
        self.extend(vs)
        return self

    def __imul__(self, n):
        super().__imul__(n)
        self._ch()
        return self

    def __deepcopy__(self, memo):
        return [copy.deepcopy(x, memo) for x in self]

    def __reduce_ex__(self, protocol):
        return (list, (list(self),))


def _adopt_list_item(v, notify):
    if isinstance(v, list):
        return TrackedList(v, notify)
    if isinstance(v, (Queue, Resource)):
        v._notify = notify
    return v


# ============================================================ модель
_RESERVED = {"time", "rng", "emit", "choose", "print", "run", "name", "events", "params", "state",
             "resolve_controls", "subs", "make_params", "clamp_"}


class Model:
    """Базовый класс модели. Наследник объявляет параметры (param) и события (методы с триггерами).

    В событиях доступны: ``self.time`` (дни), ``self.rng`` (случайные величины), ``self.emit(...)``,
    ``self.choose(...)``, ``self.print(...)``. Всё, что событие присваивает в ``self.<имя>``, — состояние
    модели: оно попадает в результат и видно триггеру when.
    """

    name: str | None = None  # имя модели для результата; по умолчанию — имя класса

    params: dict = {}   # все параметры, включая параметры подмоделей ("mall.count_masters")
    events: dict = {}   # события этой модели (без событий подмоделей)
    subs: dict = {}     # подмодели: имя -> sub

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        own, events, subs = {}, {}, {}
        for klass in reversed(cls.__mro__):
            if klass is Model or not issubclass(klass, Model):
                continue
            for name, v in vars(klass).items():
                if isinstance(v, param):
                    own[name] = v
                    subs.pop(name, None)
                elif isinstance(v, sub):
                    subs[name] = v
                    own.pop(name, None)
                    events.pop(name, None)
                elif inspect.isfunction(v) and name.startswith("_") and not name.startswith("__") \
                        and hasattr(v, _TRIGGERS_ATTR):
                    raise MagnumValidationError(
                        f"{cls.__name__}.{name}: событие не может начинаться с '_' — методы с '_' служебные "
                        f"и событиями не считаются (триггер не сработает); назовите его, например, '{name.lstrip('_')}'")
                elif (inspect.isfunction(v) and not name.startswith("_")):
                    if name in _RESERVED:
                        raise MagnumValidationError(f"'{name}' — зарезервированное имя Model, выберите другое")
                    own.pop(name, None)
                    subs.pop(name, None)
                    events[name] = _EventSpec(name, v, tuple(getattr(v, _TRIGGERS_ATTR, ()) or ()))
        for name, p in vars(cls).items():
            if isinstance(p, param):
                p.validate(p.default, MagnumValidationError, "значение по умолчанию")
        for name in list(own) + list(subs):
            if name in _RESERVED:
                raise MagnumValidationError(f"'{name}' — зарезервированное имя Model, выберите другое")
        params = dict(own)
        for sname, s in subs.items():
            for pname, p in s.model.params.items():
                params[f"{sname}.{pname}"] = p._renamed(f"{sname}.{pname}", s.defaults.get(pname))
        cls.params, cls.events, cls.subs = params, events, subs
        if "__init__" in vars(cls):
            raise MagnumValidationError(f"{cls.__name__}: не определяйте __init__ — "
                                        f"начальное состояние задаётся событием @on_start")

    # ---------------------------------------------------------- состояние
    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in _RESERVED:
            raise AttributeError(f"'{name}' нельзя присваивать: это служебный атрибут модели")
        if name in type(self).subs:
            raise AttributeError(f"'{name}' — подмодель, её нельзя переприсвоить")
        if isinstance(value, torch.Tensor):
            if not value.requires_grad and value.dim() == 0:
                value = value.item()                  # self.idle = self.masters -> обычное число
            elif value is not self.__dict__.get(name):
                value = value.clone()                 # своя копия: x -= 1 не испортит параметр (градиент сохраняется)
        sim = self.__dict__.get("_sim")
        name_full = self.__dict__.get("_prefix", "") + name
        notify = sim._notifier(name_full) if sim is not None else None
        if isinstance(value, list):
            value = TrackedList(value, notify)        # копия: присваивание не связывает два атрибута
        elif isinstance(value, (Queue, Resource)):
            value._notify = notify
        self.__dict__[name] = value
        if sim is not None:
            sim.changed.add(name_full)

    def __delattr__(self, name):
        raise AttributeError("удалять атрибуты состояния модели нельзя")

    def __getattr__(self, name):  # вызывается только если атрибута нет
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"у модели нет атрибута '{name}' (состояние ещё не создано? "
                             f"задайте его в событии @on_start)")

    @property
    def time(self) -> float:
        """Текущее моделируемое время, дни."""
        return self._sim.time

    @property
    def rng(self):
        """Случайные величины запуска (см. magnum.rand.RandomSource)."""
        return self._sim.rng

    def state(self) -> dict:
        """Текущее состояние модели (копия, в «плоском» виде); переменные подмоделей — как "mall.<имя>".
        Тензоры без градиента становятся числами/списками; тензоры с градиентом остаются тензорами
        (их можно дифференцировать по параметрам)."""
        return {k: _final(v) for k, v in self._iter_state()}

    def _iter_state(self, prefix=""):
        subs = type(self).subs
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k in subs:
                yield from v._iter_state(f"{prefix}{k}.")
            else:
                yield prefix + k, v

    # ---------------------------------------------------------- действия
    def emit(self, event, *args, after: float = 0.0) -> None:
        """Запланировать событие: self.emit(self.done, order_id, after=5 * mg.MINUTE).
        Без after — в тот же момент, после уже запланированных на этот момент событий."""
        self._sim.emit(event, args, after, self)

    def choose(self, options) -> int | None:
        """Случайно выбрать одну ветку: options — список пар (вероятность, действие или None).
        Сумма вероятностей <= 1; с остатком ничего не происходит. Возвращает индекс ветки или None.
        Вероятности могут быть тензорами с градиентом — выбор учитывается в log_prob запуска."""
        rng = self._sim.rng
        u = rng._r.random()
        acc = 0.0
        chosen = None
        for i, (p, action) in enumerate(options):
            pf = p.detach().item() if isinstance(p, torch.Tensor) else float(p)
            if not 0.0 <= pf <= 1.0:
                raise ValueError(f"choose: вероятность должна быть в [0, 1], получено {pf!r}")
            acc += pf
            if chosen is None and u < acc:
                chosen = i
        if acc > 1.0 + 1e-9:
            raise ValueError(f"choose: сумма вероятностей {acc:g} больше 1")
        if rng._tape is not None and any(_grad(p) for p, _ in options):
            if chosen is not None:
                rng._tape.append(torch.log(torch.as_tensor(options[chosen][0])))
            else:
                total = sum(torch.as_tensor(p) for p, _ in options)
                rng._tape.append(torch.log1p(-total))
        if chosen is not None and options[chosen][1] is not None:
            options[chosen][1]()
        return chosen

    def print(self, *args, **kw) -> None:
        """Отладочный вывод; отключается run(debug=False)."""
        if self._sim.debug:
            print(*args, **kw)

    # ---------------------------------------------------------- запуск
    @classmethod
    def run(cls, controls=None, seed=0, horizon: float = 365, sampling="1d", record=None,
            max_events: int = 10_000_000, debug: bool = True) -> dict:
        """Запустить модель.

        :param controls: значения параметров (перекрывают значения по умолчанию, проверяются по границам)
        :param seed: сид случайных величин
        :param horizon: горизонт, дни
        :param sampling: '1d', '6h', ... — снимок всех переменных с этим шагом; 'on_change' — запись при изменении
        :param record: какие переменные записывать в variables (None — все)
        :param max_events: предохранитель от бесконечной цепочки событий
        :param debug: выполнять ли self.print(...)
        :return: словарь: model, seed, horizon, sampling, controls, variables, final, event_counts
        """
        from .engine import Simulation
        return Simulation(cls, controls, seed, horizon, sampling, record, max_events, debug).run()

    @classmethod
    def resolve_controls(cls, controls=None) -> dict:
        """Значения параметров для запуска: по умолчанию + переданные, с проверкой границ."""
        result = {name: p.default.clone() for name, p in cls.params.items()}
        for name, value in (controls or {}).items():
            if name not in cls.params:
                raise MagnumControlError(f"неизвестный параметр '{name}' (есть: {', '.join(cls.params) or 'нет'})")
            result[name] = cls.params[name].validate(value)
        return result

    @classmethod
    def make_params(cls, controls=None, requires_grad=True, names=None) -> dict:
        """Параметры для обучения: словарь имя -> тензор-лист (вещественные — с requires_grad).
        Передаётся в run(controls=...) и в оптимизатор: torch.optim.Adam(mg.trainable(params)).
        names — какие параметры обучать (по умолчанию все вещественные)."""
        out = {}
        for name, t in cls.resolve_controls(controls).items():
            t = t.detach().clone()
            if requires_grad and t.is_floating_point() and (names is None or name in names):
                t.requires_grad_(True)
            out[name] = t
        return out

    @classmethod
    def clamp_(cls, params: dict) -> dict:
        """Вернуть параметры в границы на месте — вызывать после optimizer.step()."""
        for name, t in params.items():
            cls.params[name].clamp_(t)
        return params


def hour_of_day(t: float) -> int:
    """Час суток 0..23 момента t (дни)."""
    return min(23, int((t - math.floor(t)) * 24 + 1e-9))


def weekday(t: float, start: int = 0) -> int:
    """День недели 0..6 момента t; start — день недели момента 0 (0 — понедельник)."""
    return (int(math.floor(t)) + start) % 7


def day_index(t: float) -> int:
    """Номер дня: 0, 1, 2, ..."""
    return int(math.floor(t))
