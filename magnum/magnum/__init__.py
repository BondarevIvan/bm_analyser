"""Magnum — моделирование бизнеса как случайного потока событий.

Модель описывается на Python::

    import magnum as mg

    class Shop(mg.Model):
        rate = mg.param(10.0, low=0)                 # покупателей в час

        @mg.on_start
        def init(self):
            self.visitors = 0

        @mg.arrival(rate="rate", per=mg.HOUR)
        def visit(self):
            self.visitors += 1

    result = Shop.run(seed=1, horizon=30)
    result["final"]["visitors"]

Подробно — docs/python_api.md.
"""
from .core import (
    Model,
    TrackedList,
    arrival,
    array,
    day_index,
    event,
    every,
    hour_of_day,
    on_start,
    param,
    sub,
    weekday,
    when,
)
from .engine import run_many
from .errors import (
    MagnumControlError,
    MagnumError,
    MagnumRuntimeError,
    MagnumValidationError,
)
from .queues import Queue, Resource
from .rand import RandomSource
from .train import expected, trainable
from .units import DAY, HOUR, MINUTE, SECOND, WEEK, days, hours, minutes, seconds, weeks

__all__ = [
    "Model", "param", "sub", "array", "expected", "trainable", "event", "on_start", "every", "arrival", "when", "run_many",
    "Queue", "Resource", "TrackedList", "RandomSource",
    "hour_of_day", "weekday", "day_index",
    "SECOND", "MINUTE", "HOUR", "DAY", "WEEK", "seconds", "minutes", "hours", "days", "weeks",
    "MagnumError", "MagnumValidationError", "MagnumControlError", "MagnumRuntimeError",
]
__version__ = "0.2.0"
