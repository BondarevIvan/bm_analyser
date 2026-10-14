"""Абстрактное синтаксическое дерево модели."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class Pos:
    line: int | None = None
    column: int | None = None


_NOPOS = Pos()


def _pos():
    return field(default=_NOPOS, compare=False, repr=False)


# ----------------------------------------------------------------- выражения
@dataclass(frozen=True)
class Number:
    value: float
    pos: Pos = _pos()


@dataclass(frozen=True)
class Var:
    name: str
    pos: Pos = _pos()


@dataclass(frozen=True)
class Instant:
    name: str
    pos: Pos = _pos()


@dataclass(frozen=True)
class Index:
    target: "Expr"
    index: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class BinOp:
    op: str  # + - * / ^
    left: "Expr"
    right: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class Neg:
    operand: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class Compare:
    op: str  # < <= > >= == !=
    left: "Expr"
    right: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class Logic:
    op: str  # and | or
    left: "Expr"
    right: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class Not:
    operand: "Expr"
    pos: Pos = _pos()


@dataclass(frozen=True)
class ArrayLit:
    items: tuple["Expr", ...]
    pos: Pos = _pos()


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Expr", ...]
    kwargs: tuple[tuple[str, "Expr"], ...]
    user: bool  # True — `call f(...)` (функция модели), False — встроенная функция
    pos: Pos = _pos()


Expr = Union[Number, Var, Instant, Index, BinOp, Neg, Compare, Logic, Not, ArrayLit, Call]


# ----------------------------------------------------------------- операторы
@dataclass(frozen=True)
class Assign:
    target: Expr  # Var | Instant | Index(...)
    value: Expr
    pos: Pos = _pos()


@dataclass(frozen=True)
class Print:
    args: tuple  # Expr | str (строковый литерал)
    pos: Pos = _pos()


@dataclass(frozen=True)
class Return:
    value: Expr
    pos: Pos = _pos()


# ------------------------------------------------------------------ триггеры
@dataclass(frozen=True)
class Every:
    period: Expr
    pos: Pos = _pos()


@dataclass(frozen=True)
class Arrival:
    distribution: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]
    pos: Pos = _pos()


@dataclass(frozen=True)
class OnStart:
    pos: Pos = _pos()


Trigger = Union[Every, Arrival, OnStart]


# -------------------------------------------------------------------- события
@dataclass(frozen=True)
class Emit:
    event: str
    args: tuple[Expr, ...]
    delay: Expr | None
    pos: Pos = _pos()


@dataclass(frozen=True)
class Branch:
    probability: Expr
    emits: tuple[Emit, ...]  # все emit ветки выполняются по порядку
    pos: Pos = _pos()


@dataclass(frozen=True)
class Event:
    name: str
    params: tuple[str, ...]
    do: tuple[Union[Assign, Print], ...]
    triggers: tuple[Trigger, ...]
    choose: tuple[tuple[Branch, ...], ...]  # каждый блок choose — независимый розыгрыш
    emits: tuple[Emit, ...]
    pos: Pos = _pos()


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[str, ...]
    body: tuple[Union[Assign, Return, Print], ...]
    pos: Pos = _pos()


# ------------------------------------------------------------------ controls
@dataclass(frozen=True)
class Limit:
    name: str
    lower: Expr
    lower_closed: bool
    upper: Expr
    upper_closed: bool
    pos: Pos = _pos()


@dataclass(frozen=True)
class Default:
    name: str
    value: Expr
    pos: Pos = _pos()


@dataclass(frozen=True)
class Controls:
    defaults: tuple[Default, ...]
    limits: tuple[Limit, ...]
    pos: Pos = _pos()


@dataclass(frozen=True)
class ModelDef:
    name: str
    controls: tuple[Controls, ...]
    functions: tuple[Function, ...]
    events: tuple[Event, ...]
    pos: Pos = _pos()
