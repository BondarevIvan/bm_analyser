"""Политики снятия состояния переменных (параметр ``sampling`` у ``Model.run``).

* ``"1d"``, ``"6h"``, ``"30m"``, ``"1w"``, ``"0.5d"`` (или число дней) — снимок всех переменных
  через равные промежутки времени: t = 0, P, 2P, ... <= horizon;
* ``"on_change"`` — значение переменной сохраняется в момент её изменения.

Результат в обоих случаях одного вида::

    {"имя": {"time": [t0, t1, ...], "value": [v0, v1, ...]}, ...}
"""
from __future__ import annotations

import copy
import math
import re

from .parser import DURATION_UNITS
from .queues import plain as _copy  # копия массива / развёрнутая очередь и ресурс

EPS = 1e-9
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*([wdhms])\s*$")


def make_recorder(spec, horizon: float, record=None):
    """Создать объект записи по строке политики. record — имена переменных для записи (None — все)."""
    rec = _make_recorder(spec, horizon)
    if record is not None:
        if isinstance(record, str):
            record = [record]
        rec.record = frozenset(record)
    return rec


def _make_recorder(spec, horizon: float):
    if isinstance(spec, bool):
        raise ValueError(f"sampling: неверная политика {spec!r}")
    if isinstance(spec, (int, float)):
        period = float(spec)
    elif isinstance(spec, str):
        s = spec.strip().lower()
        if s == "on_change":
            return OnChangeRecorder()
        m = _DURATION_RE.match(s)
        if not m:
            raise ValueError(
                f"sampling: неверная политика {spec!r}; ожидалась длительность ('1d', '6h', '30m') или 'on_change'")
        period = float(m.group(1)) * DURATION_UNITS[m.group(2)]
    else:
        raise ValueError(f"sampling: неверная политика {spec!r}")
    if not (0 < period < math.inf):
        raise ValueError(f"sampling: период должен быть > 0, получено {spec!r}")
    return PeriodicRecorder(period, horizon)


class _Recorder:
    next_due = math.inf      # ближайший момент снимка (для периодической записи)
    tracks_changes = False   # нужен ли список изменённых переменных после каждого события

    def __init__(self):
        self.series: dict[str, dict[str, list]] = {}
        self.record = None   # frozenset имён или None — все переменные

    def _append(self, name, t, value):
        s = self.series.get(name)
        if s is None:
            s = self.series[name] = {"time": [], "value": []}
        s["time"].append(t)
        s["value"].append(_copy(value))

    # хуки симуляции
    def start(self, globals_):
        pass

    def before_event(self, t, globals_):
        pass

    def after_event(self, t, globals_, changed):
        pass

    def finish(self, globals_):
        pass


class PeriodicRecorder(_Recorder):
    """Снимок всех переменных в моменты 0, P, 2P, ... <= horizon.
    Снимок в момент t — состояние после всех событий с временем <= t."""

    def __init__(self, period: float, horizon: float):
        super().__init__()
        self.period = period
        n = int(math.floor(horizon / period + EPS)) + 1
        self.times = [round(i * period, 12) for i in range(n)]
        self._i = 0
        self.next_due = self.times[0]

    def _take(self, globals_):
        t = self.times[self._i]
        record = self.record
        for name, value in globals_.items():
            if record is None or name in record:
                self._append(name, t, value)
        self._i += 1
        self.next_due = self.times[self._i] if self._i < len(self.times) else math.inf

    def before_event(self, t, globals_):
        while self._i < len(self.times) and self.times[self._i] < t - EPS:
            self._take(globals_)

    def finish(self, globals_):
        while self._i < len(self.times):
            self._take(globals_)


class OnChangeRecorder(_Recorder):
    """Значение переменной сохраняется, когда оно меняется.
    Несколько изменений в один и тот же момент времени схлопываются в одно (последнее) значение."""

    tracks_changes = True

    def start(self, globals_):
        for name, value in globals_.items():  # начальные значения контролов
            if self.record is None or name in self.record:
                self._append(name, 0.0, value)

    def after_event(self, t, globals_, changed):
        record = self.record
        for name in changed:
            if record is None or name in record:
                self._record(name, t, _copy(globals_[name]))

    def _record(self, name, t, value):
        s = self.series.get(name)
        if s is None:
            self._append(name, t, value)
            return
        times, values = s["time"], s["value"]
        if values[-1] == value:
            return
        if abs(times[-1] - t) <= EPS:  # уже меняли в этот момент — перезаписываем
            values[-1] = _copy(value)
            if len(values) >= 2 and values[-2] == value:  # вернулось к прежнему — изменения не было
                times.pop()
                values.pop()
        else:
            times.append(t)
            values.append(_copy(value))
