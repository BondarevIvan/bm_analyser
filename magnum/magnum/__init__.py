"""Magnum — язык описания бизнес-модели как случайного процесса и движок его расчёта.

Пример::

    from magnum import load_model

    model = load_model(text)
    result = model.run(controls={"value": 2}, seed=42, horizon=365)
    result["variables"]["summa"]   # значения по дням 0..365
"""
from .errors import (
    MagnumControlError,
    MagnumError,
    MagnumRuntimeError,
    MagnumSyntaxError,
    MagnumValidationError,
)
from .model import Model, load_model, load_model_file, run_model
from .parser import parse

__all__ = [
    "Model",
    "load_model",
    "load_model_file",
    "run_model",
    "parse",
    "MagnumError",
    "MagnumSyntaxError",
    "MagnumValidationError",
    "MagnumControlError",
    "MagnumRuntimeError",
]
__version__ = "0.1.0"
