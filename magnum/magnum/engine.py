"""Движок дискретно-событийного моделирования для моделей на Python (magnum.Model)."""
from __future__ import annotations

import heapq
import itertools
import math
from typing import Any

import torch

from .errors import MagnumError, MagnumRuntimeError, MagnumValidationError
from .queues import plain
from .rand import RandomSource, _grad
from .sampling import make_recorder

EPS = 1e-9  # допуск сравнения моментов времени, дни
_BOOTSTRAP = "__bootstrap__"


class _Occurrence:
    __slots__ = ("event", "args", "trigger", "anchor", "k", "period", "lam", "t0")

    def __init__(self, event, args=(), trigger=None, anchor=0.0, k=0, period=0.0, lam=None, t0=0.0):
        self.event, self.args, self.trigger = event, args, trigger
        self.anchor, self.k, self.period = anchor, k, period
        self.lam, self.t0 = lam, t0   # для arrival с обучаемой интенсивностью: λ (в 1/день) и момент розыгрыша


class _ReadTracker:
    """Подставляется в условие when вместо модели и запоминает, какие атрибуты оно читает
    (полные имена: "mall.clients" для подмодели)."""

    __slots__ = ("_m", "_root", "_deps", "_opaque", "_event")

    def __init__(self, model, event, root=None):
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "_root", root if root is not None else self)
        object.__setattr__(self, "_deps", set())
        object.__setattr__(self, "_opaque", False)
        object.__setattr__(self, "_event", event)

    def __getattr__(self, name):
        if name in ("time", "rng"):
            raise MagnumValidationError(
                f"when в событии '{self._event}': условие не может зависеть от {name} — "
                f"для времени используйте every / emit(after=...), случайность — внутри события")
        m, root = self._m, self._root
        v = getattr(m, name)
        if name in type(m).subs:
            return _ReadTracker(v, self._event, root)
        if callable(v) and not hasattr(v, "__len__"):
            # вызов метода модели: что он читает, не видно — условие будем проверять после каждого события
            object.__setattr__(root, "_opaque", True)
        else:
            root._deps.add(m._prefix + name)
        return v

    def __setattr__(self, name, value):
        raise MagnumValidationError(f"when в событии '{self._event}': условие не должно менять состояние модели")


class _When:
    __slots__ = ("event", "full", "inst", "ti", "cond", "deps", "opaque")

    def __init__(self, event, full, inst, ti, cond):
        self.event, self.full, self.inst, self.ti, self.cond = event, full, inst, ti, cond
        self.deps: set = set()
        self.opaque = True  # до первой проверки зависимости неизвестны


class Simulation:
    def __init__(self, cls, controls, seed, horizon, sampling, record, max_events, debug):
        if horizon < 0:
            raise ValueError("horizon должен быть >= 0")
        self.cls = cls
        self.seed = seed
        self.log_probs: list = []      # log p случайных исходов, зависящих от обучаемых параметров
        self.rng = RandomSource(seed, self.log_probs)
        self._zero_rate: list = []     # (λ, t0) потоков arrival, остановленных при λ = 0 (для log_prob)
        # arrival с одним и тем же тензором интенсивности: копим число приходов и суммарное время ожидания,
        # log p = n·log λ − λ·Σ Δt — один узел графа вместо узла на каждый приход
        self._arrival_acc: dict = {}   # id(тензор rate) -> [rate, per, n, Σ Δt]
        self.horizon = float(horizon)
        self.sampling = sampling
        self.max_events = max_events
        self.debug = debug
        self.recorder = make_recorder(sampling, self.horizon, record)
        self.changed: set = set()
        self.time = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self._notifiers: dict = {}
        self.controls = cls.resolve_controls(controls)

        # дерево экземпляров: внешняя модель и подмодели; события — по полным именам ("mall.order_done")
        self.instances: list = []           # (префикс, экземпляр); подмодели раньше внешней модели
        self.specs: dict = {}               # полное имя события -> (экземпляр, _EventSpec)
        self.model = self._build(cls, "")
        self.has_subs = len(self.instances) > 1
        self.event_counts = {full: 0 for full, (_, ev) in self.specs.items()
                             if ev.triggers or hasattr(ev.fn, "__magnum_triggers__")}

        self._when = [_When(ev, full, inst, ti, trig[1]) for full, (inst, ev) in self.specs.items()
                      for ti, trig in enumerate(ev.triggers) if trig[0] == "when"]
        self._when_pending: set = set()
        self._started = False

    def _build(self, cls, prefix):
        inst = cls.__new__(cls)
        object.__setattr__(inst, "_sim", self)
        object.__setattr__(inst, "_prefix", prefix)
        for sname in cls.subs:
            child = self._build(cls.subs[sname].model, f"{prefix}{sname}.")
            inst.__dict__[sname] = child
        for name in cls.params:
            if "." not in name:
                v = self.controls[prefix + name]
                inst.__dict__[name] = v if v.requires_grad else v.clone()   # параметр — всегда тензор
                self.changed.add(prefix + name)
        self.instances.append((prefix, inst))
        for name, ev in cls.events.items():
            self.specs[prefix + name] = (inst, ev)
        return inst

    # ------------------------------------------------------------ служебное
    def _notifier(self, name):
        f = self._notifiers.get(name)
        if f is None:
            changed = self.changed
            f = self._notifiers[name] = lambda: changed.add(name)
        return f

    def _public_state(self) -> dict:
        return dict(self.model._iter_state())

    def schedule(self, t, event, args=(), **kw):
        heapq.heappush(self._heap, (t, next(self._seq), _Occurrence(event, args, **kw)))

    def _value(self, spec, what, event, inst, raw=False):
        """Число из спецификации триггера: число, имя атрибута модели или функция m -> число.
        raw=True — вернуть и само значение (тензор), чтобы учесть его в log_prob."""
        if callable(spec):
            v = spec(inst)
        elif isinstance(spec, str):
            v = inst
            for part in spec.split("."):
                v = getattr(v, part)
        else:
            v = spec
        if isinstance(v, torch.Tensor) and v.dim() == 0 and not v.dtype == torch.bool:
            f = v.item()
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MagnumRuntimeError(f"событие '{event}': {what} должно быть числом, получено {v!r}")
        else:
            f = v
        return (float(f), v) if raw else float(f)

    # ------------------------------------------------------------ emit
    def emit(self, event, args, after, caller=None):
        caller = self.model if caller is None else caller
        if callable(event):
            name = getattr(event, "__name__", None)
            owner = getattr(event, "__self__", None)
            if getattr(owner, "_sim", None) is not self:
                raise MagnumRuntimeError(
                    f"emit: {event!r} — не событие этой модели; передайте метод модели и аргументы: "
                    f"self.emit(self.client_leave, client_id, after=...) (lambda не подходит)")
            if str(name).startswith("_"):
                raise MagnumRuntimeError(
                    f"emit: '{name}' начинается с '_' — это служебный метод, а не событие; уберите '_' из имени")
            full = owner._prefix + str(name)
            entry = self.specs.get(full)
            if entry is None or entry[1].fn is not getattr(event, "__func__", None):
                raise MagnumRuntimeError(f"emit: {event!r} — не событие этой модели")
        else:
            name = event
            full = caller._prefix + str(name)
            entry = self.specs.get(full)
        if entry is None:
            raise MagnumRuntimeError(f"emit: у модели нет события '{name}'")
        spec = entry[1]
        if not spec.min_args <= len(args) <= spec.max_args:
            raise MagnumRuntimeError(f"emit: событие '{name}' ожидает {spec.min_args}"
                                     f"{'' if spec.max_args == spec.min_args else '+'} аргумент(ов), передано {len(args)}")
        if isinstance(after, torch.Tensor) and after.dim() == 0:
            after = after.item()
        if isinstance(after, bool) or not isinstance(after, (int, float)) or not (0 <= after < math.inf):
            raise MagnumRuntimeError(f"emit {name}: after должно быть конечным числом >= 0, получено {after!r}")
        self.schedule(self.time + after, full, tuple(args))

    # ------------------------------------------------------------ триггеры
    def _schedule_trigger(self, full, ti, prev):
        inst, ev = self.specs[full]
        trig = ev.triggers[ti]
        kind = trig[0]
        if kind == "when":
            self._when_pending.discard(full)
        elif kind == "every":
            period = self._value(trig[1], "период every", full, inst)
            if not 0 < period < math.inf:
                raise MagnumRuntimeError(f"событие '{full}': период every должен быть > 0, получено {period!r}")
            if prev is not None and prev.period == period:
                anchor, k = prev.anchor, prev.k + 1
            else:
                anchor, k = self.time, 1
            self.schedule(anchor + k * period, full, (), trigger=ti, anchor=anchor, k=k, period=period)
        elif kind == "arrival":
            rate, rate_v = self._value(trig[1], "интенсивность arrival", full, inst, raw=True)
            per, per_v = self._value(trig[2], "per в arrival", full, inst, raw=True)
            if rate < 0 or per <= 0:
                raise MagnumRuntimeError(f"событие '{full}': нужно rate >= 0 и per > 0")
            if _grad(per_v):
                lam = rate_v / per_v
            elif _grad(rate_v):
                acc = self._arrival_acc.get(id(rate_v))
                if acc is None:
                    acc = self._arrival_acc[id(rate_v)] = [rate_v, per, 0, 0.0]
                lam = acc if acc[1] == per else rate_v / per_v
            else:
                lam = None
            if rate > 0:
                self.schedule(self.time + self.rng._r.expovariate(rate / per), full, (), trigger=ti,
                              lam=lam, t0=self.time)
            elif lam is not None:
                self._zero_rate.append((lam, self.time))

    def _check_when(self, changed):
        if not self._started:      # до конца on_start (всех моделей) условия when не проверяем
            return
        for w in self._when:
            name = w.full
            if name in self._when_pending:
                continue
            if changed is not None and not w.opaque and w.deps.isdisjoint(changed):
                continue
            tracker = _ReadTracker(w.inst, name)
            try:
                ok = w.cond(tracker)
            except MagnumError:
                raise
            except Exception as e:
                raise MagnumRuntimeError(f"условие when события '{name}' (t={self.time:g}): "
                                         f"{type(e).__name__}: {e}") from e
            w.deps, w.opaque = tracker._deps, tracker._opaque
            if ok:
                self._when_pending.add(name)
                self.schedule(self.time, name, (), trigger=w.ti)

    def _arrival_lp(self, lam, dt, arrived):
        if isinstance(lam, list):        # накопитель для общего тензора интенсивности
            lam[2] += arrived
            lam[3] += dt
        else:
            self.log_probs.append(torch.log(lam) - lam * dt if arrived else -lam * dt)

    # ------------------------------------------------------------ событие
    def _process(self, occ):
        if occ.event == _BOOTSTRAP:
            self._started = True
            for full, (inst, ev) in self.specs.items():
                for ti, trig in enumerate(ev.triggers):
                    if trig[0] in ("every", "arrival"):
                        self._schedule_trigger(full, ti, None)
            self._check_when(None)
            return
        full = occ.event
        inst, ev = self.specs[full]
        if occ.lam is not None:   # интервал до прихода ~ Exp(λ): log p = log λ − λ·Δt
            self._arrival_lp(occ.lam, self.time - occ.t0, True)
        if occ.trigger is not None:
            self._schedule_trigger(full, occ.trigger, occ)
        try:
            ev.fn(inst, *occ.args)
        except MagnumError:
            raise
        except Exception as e:
            raise MagnumRuntimeError(f"событие '{full}' (t={self.time:g}): {type(e).__name__}: {e}") from e
        self.event_counts[full] = self.event_counts.get(full, 0) + 1

    # ------------------------------------------------------------ цикл
    def run(self) -> dict:
        rec = self.recorder
        for full, (inst, ev) in self.specs.items():   # подмодели — раньше внешней модели
            for trig in ev.triggers:
                if trig[0] == "start":
                    self.schedule(0.0, full, (), trigger=None)
        self.schedule(0.0, _BOOTSTRAP)
        rec.start(self._public_state())
        self.changed.clear()

        heap, heappop, process, changed = self._heap, heapq.heappop, self._process, self.changed
        state = _FlatState(self) if self.has_subs else self.model.__dict__
        limit = self.horizon + EPS
        has_when = bool(self._when)
        tracks = rec.tracks_changes or has_when
        next_due = rec.next_due
        processed = 0
        while heap:
            t = heap[0][0]
            if t > limit:
                break
            occ = heappop(heap)[2]
            if t > next_due + EPS:
                rec.before_event(t, self._public_state())
                next_due = rec.next_due
            self.time = t
            if tracks:
                changed.clear()
                process(occ)
                if changed:
                    rec.after_event(t, state, changed)
                    if has_when:
                        self._check_when(changed)
                elif has_when and any(w.opaque for w in self._when):
                    self._check_when(changed)
            else:
                process(occ)
            processed += 1
            if processed > self.max_events:
                raise MagnumRuntimeError(
                    f"превышено max_events={self.max_events} (t={t:g}); возможна бесконечная цепочка событий")
        rec.finish(self._public_state())
        self.time = self.horizon
        # приходы, не случившиеся до горизонта: P(Δt > horizon − t0) = exp(−λ·(horizon − t0))
        for _, _, occ in self._heap:
            if occ.lam is not None:
                self._arrival_lp(occ.lam, self.horizon - occ.t0, False)
        for lam, t0 in self._zero_rate:
            self._arrival_lp(lam, self.horizon - t0, False)
        self.rng.flush()
        for rate_v, per, n, wait in self._arrival_acc.values():
            lam = rate_v / per
            self.log_probs.append(n * torch.log(lam) - lam * wait if n else -lam * wait)
        log_prob = torch.stack([torch.as_tensor(x, dtype=torch.float64).reshape(()) for x in self.log_probs]).sum() \
            if self.log_probs else torch.zeros((), dtype=torch.float64)
        return {
            "model": self.cls.name or self.cls.__name__,
            "seed": self.seed,
            "horizon": self.horizon,
            "sampling": self.sampling,
            "controls": {k: v.detach().clone() for k, v in self.controls.items()},
            "variables": rec.series,
            "final": self.model.state(),
            "event_counts": dict(self.event_counts),
            "log_prob": log_prob,
        }


# ============================================================ серии запусков
def _run_one(args):
    cls, kw = args
    return cls.run(**kw)


def run_many(cls, seeds, processes: int = 1, **kw) -> list[Any]:
    """Запустить модель на нескольких сидах; processes > 1 — параллельно в отдельных процессах.
    Класс модели должен быть объявлен в импортируемом модуле (не в __main__ интерактивной сессии)."""
    jobs = [(cls, dict(kw, seed=s)) for s in seeds]
    if processes <= 1:
        return [_run_one(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(processes) as pool:
        return list(pool.map(_run_one, jobs, chunksize=max(1, len(jobs) // (processes * 4))))


class _FlatState:
    """Доступ к переменным по полным именам ("mall.clients") — для записи on_change при наличии подмоделей."""

    __slots__ = ("_by_prefix",)

    def __init__(self, sim):
        self._by_prefix = {prefix: inst.__dict__ for prefix, inst in sim.instances}

    def __getitem__(self, full):
        i = full.rfind(".")
        return self._by_prefix[full[:i + 1]][full[i + 1:]]
