"""Разбор текста модели в AST (lark, LALR)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken, VisitError

from . import nodes as N
from .errors import MagnumError, MagnumSyntaxError

GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")

# длительности переводятся в дни
DURATION_UNITS = {"w": 7.0, "d": 1.0, "h": 1.0 / 24, "m": 1.0 / (24 * 60), "s": 1.0 / (24 * 3600)}


@lru_cache(maxsize=1)
def get_parser() -> Lark:
    return Lark(
        GRAMMAR_PATH.read_text(encoding="utf-8"),
        parser="lalr",
        maybe_placeholders=True,
        propagate_positions=True,
    )


def _unquote(tok) -> str:
    return re.sub(r"\\(.)", r"\1", str(tok)[1:-1])


def _pos(meta) -> N.Pos:
    if getattr(meta, "empty", True):
        return N.Pos()
    return N.Pos(meta.line, meta.column)


def _tok_pos(tok: Token) -> N.Pos:
    return N.Pos(tok.line, tok.column)


@v_args(meta=True)
class _Builder(Transformer):
    # ------------------------------------------------------------- модель
    def start(self, meta, ch):
        return ch[0]

    def model(self, meta, ch):
        name_tok, *items = ch
        name = _unquote(name_tok)
        return N.ModelDef(
            name=name,
            controls=tuple(i for i in items if isinstance(i, N.Controls)),
            functions=tuple(i for i in items if isinstance(i, N.Function)),
            events=tuple(i for i in items if isinstance(i, N.Event)),
            pos=_pos(meta),
        )

    # -------------------------------------------------- очереди и ресурсы
    def _queue_op(op):  # noqa: N805
        def method(self, meta, ch):
            return N.QueueOp(op, str(ch[0]), ch[1] if len(ch) > 1 else None, pos=_tok_pos(ch[0]))
        return method

    push_stmt, remove_stmt, clear_stmt = _queue_op("push"), _queue_op("remove"), _queue_op("clear")
    seize_stmt, release_stmt = _queue_op("seize"), _queue_op("release")
    del _queue_op

    def pop_stmt(self, meta, ch):
        return N.PopAssign(ch[0], str(ch[1]), pos=_pos(meta))

    # ----------------------------------------------------------- controls
    def controls(self, meta, ch):
        defaults, limits = [], []
        for kind, items in ch:
            (defaults if kind == "default" else limits).extend(items)
        return N.Controls(tuple(defaults), tuple(limits), pos=_pos(meta))

    def defaults(self, meta, ch):
        return ("default", ch)

    def default_item(self, meta, ch):
        return N.Default(str(ch[0]), ch[1], pos=_tok_pos(ch[0]))

    def limits(self, meta, ch):
        return ("limits", ch)

    def limit_item(self, meta, ch):
        name, (lo, lo_closed), (hi, hi_closed) = ch
        return N.Limit(str(name), lo, lo_closed, hi, hi_closed, pos=_tok_pos(name))

    def closed_lower(self, meta, ch):
        return (ch[0], True)

    def open_lower(self, meta, ch):
        return (ch[0], False)

    closed_upper = closed_lower
    open_upper = open_lower

    # ------------------------------------------------------------ функции
    def param_list(self, meta, ch):
        return tuple(str(t) for t in ch)

    def function_def(self, meta, ch):
        name, params, *body = ch
        return N.Function(str(name), params or (), tuple(body), pos=_tok_pos(name))

    def return_stmt(self, meta, ch):
        return N.Return(ch[0], pos=_pos(meta))

    # ------------------------------------------------------------- события
    def event_def(self, meta, ch):
        name, params, *sections = ch
        do, triggers, chooses, emits = None, None, [], []
        for kind, payload, spos in sections:
            if kind == "do":
                if do is not None:
                    raise MagnumSyntaxError(f"в событии '{name}' секция do встречается больше одного раза", spos.line, spos.column)
                do = payload
            elif kind == "trigger":
                if triggers is not None:
                    raise MagnumSyntaxError(f"в событии '{name}' секция trigger встречается больше одного раза", spos.line, spos.column)
                triggers = payload
            elif kind == "choose":
                chooses.append(payload)
            else:
                emits.append(payload)
        return N.Event(
            name=str(name),
            params=params or (),
            do=tuple(do or ()),
            triggers=tuple(triggers or ()),
            choose=tuple(chooses),
            emits=tuple(emits),
            pos=_tok_pos(name),
        )

    def do_block(self, meta, ch):
        return ("do", tuple(ch), _pos(meta))

    def trigger_block(self, meta, ch):
        return ("trigger", tuple(ch), _pos(meta))

    def choose_block(self, meta, ch):
        return ("choose", tuple(ch), _pos(meta))

    def every_trigger(self, meta, ch):
        return N.Every(ch[0], pos=_pos(meta))

    def arrival_trigger(self, meta, ch):
        name, arguments = ch
        args, kwargs = arguments or ((), ())
        return N.Arrival(str(name), args, kwargs, pos=_tok_pos(name))

    def when_trigger(self, meta, ch):
        return N.When(ch[0], pos=_pos(meta))

    def start_trigger(self, meta, ch):
        return N.OnStart(pos=_pos(meta))

    def branch(self, meta, ch):
        return N.Branch(ch[0], tuple(e[1] for e in ch[1:]), pos=_pos(meta))

    def emit_stmt(self, meta, ch):
        name, arguments, delay = ch
        args, kwargs = arguments or ((), ())
        if kwargs:
            raise MagnumSyntaxError("в emit допускаются только позиционные аргументы", name.line, name.column)
        return ("emit", N.Emit(str(name), args, delay, pos=_tok_pos(name)), _pos(meta))

    # -------------------------------------------------------- присваивание
    def print_stmt(self, meta, ch):
        return N.Print(tuple(c for c in ch if c is not None), pos=_pos(meta))

    def string_lit(self, meta, ch):
        return _unquote(ch[0])

    def assignment(self, meta, ch):
        return N.Assign(ch[0], ch[1], pos=_pos(meta))

    def target(self, meta, ch):
        base, *indices = ch
        node = N.Var(str(base), pos=_tok_pos(base)) if isinstance(base, Token) else base
        for idx in indices:
            node = N.Index(node, idx, pos=node.pos)
        return node

    def instant_ref(self, meta, ch):
        return N.Instant(str(ch[0]), pos=_pos(meta))

    # ----------------------------------------------------------- выражения
    def _bin(op):  # noqa: N805
        def method(self, meta, ch):
            return N.BinOp(op, ch[0], ch[1], pos=_pos(meta))
        return method

    add, sub, mul, div, pow = _bin("+"), _bin("-"), _bin("*"), _bin("/"), _bin("^")
    del _bin

    def compare(self, meta, ch):
        return N.Compare(str(ch[1]), ch[0], ch[2], pos=_pos(meta))

    def and_op(self, meta, ch):
        return N.Logic("and", ch[0], ch[1], pos=_pos(meta))

    def or_op(self, meta, ch):
        return N.Logic("or", ch[0], ch[1], pos=_pos(meta))

    def not_op(self, meta, ch):
        return N.Not(ch[0], pos=_pos(meta))

    def neg(self, meta, ch):
        return N.Neg(ch[0], pos=_pos(meta))

    def index(self, meta, ch):
        return N.Index(ch[0], ch[1], pos=_pos(meta))

    def number(self, meta, ch):
        return N.Number(float(ch[0]), pos=_tok_pos(ch[0]))

    def duration(self, meta, ch):
        tok = ch[0]
        return N.Number(float(tok[:-1]) * DURATION_UNITS[tok[-1]], pos=_tok_pos(tok))

    def var(self, meta, ch):
        return N.Var(str(ch[0]), pos=_tok_pos(ch[0]))

    def array(self, meta, ch):
        return N.ArrayLit(tuple(c for c in ch if c is not None), pos=_pos(meta))

    def _call(self, meta, ch, user):
        name, arguments = ch
        args, kwargs = arguments or ((), ())
        return N.Call(str(name), args, kwargs, user, pos=_tok_pos(name))

    def user_call(self, meta, ch):
        return self._call(meta, ch, True)

    def builtin_call(self, meta, ch):
        return self._call(meta, ch, False)

    def arguments(self, meta, ch):
        args, kwargs = [], []
        for a in ch:
            if isinstance(a, tuple) and len(a) == 3 and a[0] == "__kwarg__":
                kwargs.append((a[1], a[2]))
            else:
                if kwargs:
                    raise MagnumSyntaxError("позиционный аргумент после именованного", meta.line, meta.column)
                args.append(a)
        names = [k for k, _ in kwargs]
        if len(names) != len(set(names)):
            raise MagnumSyntaxError("именованный аргумент указан дважды", meta.line, meta.column)
        return (tuple(args), tuple(kwargs))

    def kwarg(self, meta, ch):
        return ("__kwarg__", str(ch[0]), ch[1])


def parse(source: str) -> N.ModelDef:
    """Разобрать текст модели и вернуть AST."""
    try:
        tree = get_parser().parse(source)
    except UnexpectedEOF as e:
        raise MagnumSyntaxError("неожиданный конец текста модели (не закрыта скобка?)") from None
    except UnexpectedCharacters as e:
        raise MagnumSyntaxError(f"недопустимый символ {source[e.pos_in_stream]!r}", e.line, e.column) from None
    except UnexpectedToken as e:
        if e.token.type == "$END":
            raise MagnumSyntaxError("неожиданный конец текста модели (не закрыта скобка?)") from None
        expected = ", ".join(sorted(e.expected)[:8])
        raise MagnumSyntaxError(f"неожиданный токен {str(e.token)!r}; ожидалось: {expected}", e.line, e.column) from None
    except UnexpectedInput as e:  # pragma: no cover
        raise MagnumSyntaxError(str(e), getattr(e, "line", None), getattr(e, "column", None)) from None
    try:
        return _Builder().transform(tree)
    except VisitError as e:
        if isinstance(e.orig_exc, MagnumError):
            raise e.orig_exc from None
        raise
