"""Единицы времени. Время модели всегда в днях."""

SECOND = 1.0 / 86400
MINUTE = 1.0 / 1440
HOUR = 1.0 / 24
DAY = 1.0
WEEK = 7.0

# суффиксы длительностей: 1d, 6h, 30m, 15s, 1w
DURATION_UNITS = {"w": WEEK, "d": DAY, "h": HOUR, "m": MINUTE, "s": SECOND}


def seconds(x: float) -> float:
    return x * SECOND


def minutes(x: float) -> float:
    return x * MINUTE


def hours(x: float) -> float:
    return x * HOUR


def days(x: float) -> float:
    return x * DAY


def weeks(x: float) -> float:
    return x * WEEK
