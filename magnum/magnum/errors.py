"""Исключения библиотеки."""
from __future__ import annotations


class MagnumError(Exception):
    """Базовое исключение библиотеки."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        self.message = message
        self.line = line
        self.column = column
        where = ""
        if line is not None:
            where = f"строка {line}" + (f", столбец {column}" if column is not None else "") + ": "
        super().__init__(where + message)


class MagnumSyntaxError(MagnumError):
    """Текст модели не соответствует грамматике."""


class MagnumValidationError(MagnumError):
    """Модель синтаксически корректна, но содержит смысловую ошибку
    (неизвестное событие, неверное число аргументов и т.п.)."""


class MagnumControlError(MagnumError):
    """Переданные в запуск контролы неверны (неизвестное имя, выход за limits, форма массива)."""


class MagnumRuntimeError(MagnumError):
    """Ошибка во время моделирования (деление на ноль, неопределённая переменная и т.п.)."""
