"""Исключения библиотеки."""


class MagnumError(Exception):
    """Базовое исключение библиотеки."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MagnumValidationError(MagnumError):
    """Ошибка в описании модели (параметр вне своих границ, зарезервированное имя, неверный триггер и т.п.)."""


class MagnumControlError(MagnumError):
    """Переданные в запуск controls неверны (неизвестное имя, выход за границы, форма массива)."""


class MagnumRuntimeError(MagnumError):
    """Ошибка во время моделирования (исключение в событии, неверный emit, превышен max_events и т.п.)."""
