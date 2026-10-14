"""Сравнения, логика (and/or/not) и условное выражение if(условие, a, b)."""
import pytest

from magnum import MagnumRuntimeError, MagnumSyntaxError, MagnumValidationError, load_model, parse, run_model
from magnum import nodes as N
from conftest import wrap


def calc(expr, controls_block="controls { default { a = 3  b = 5  arr = [1, 2] } }"):
    r = run_model(wrap(controls_block + f" event e {{ do {{ x = {expr} }} trigger {{ on start }} }}"), horizon=0)
    return r["final"]["x"]


@pytest.mark.parametrize("expr,expected", [
    ("a < b", 1), ("a > b", 0), ("a <= 3", 1), ("a >= 4", 0), ("a == 3", 1), ("a != 3", 0),
    ("a + 2 == b", 1),                    # арифметика сильнее сравнения
    ("-a < 0", 1),
    ("a < b and b < 10", 1), ("a > b and b < 10", 0),
    ("a > b or b == 5", 1), ("a > b or b == 4", 0),
    ("not a > b", 1), ("not a", 0), ("not 0", 1),
    ("a > 1 or b > 1 and a > 100", 1),    # and сильнее or
    ("(a > 1 or b > 1) and a > 100", 0),
    ("arr == [1, 2]", 1), ("arr != [1, 2]", 0),
    ("if(a < b, 10, 20)", 10), ("if(a > b, 10, 20)", 20),
    ("if(a, 1, 2)", 1), ("if(0, 1, 2)", 2),          # истина — любое ненулевое число
    ("if(a > b, 1, if(a == 3, 2, 3))", 2),          # вложенный if
    ("1 + if(a < b, 1, 0) * 10", 11),
    ("if(a < b, arr, [0])", [1, 2]),
])
def test_expressions(expr, expected):
    assert calc(expr) == expected


def test_if_evaluates_only_chosen_branch():
    # деление на ноль в невыбранной ветке не выполняется
    assert calc("if(b == 0, 0, a / b)") == pytest.approx(0.6)
    assert calc("if(a > 0, 1, missing_variable)") == 1


def test_and_or_short_circuit():
    assert calc("a > 100 and missing_variable") == 0
    assert calc("a < 100 or missing_variable") == 1


def test_if_as_conditional_update_and_conditional_emit():
    r = run_model(wrap("""
        controls { default { capacity = 3 } }
        event _init { do { stock = 5  restocks = 0 } trigger { on start } }
        event restock { do { stock = stock + 10  restocks = restocks + 1 } }
        event sell {
            do { stock = if(stock > 0, stock - 1, stock) }
            choose { if(stock < capacity, 1, 0): emit restock }   # условный emit
            trigger { every 1d }
        }
    """), horizon=20)
    stock = r["variables"]["stock"]["value"]
    assert min(stock) >= 2
    assert r["final"]["restocks"] == r["event_counts"]["restock"] > 0


def test_if_in_controls_default():
    m = load_model(wrap("controls { default { a = 2  b = if(a > 1, 10, 0) } }"))
    assert m.defaults["b"] == 10


def test_parse_tree():
    m = parse(wrap("event e { do { x = not a < b + 1 and c } }"))
    v = m.events[0].do[0].value
    assert isinstance(v, N.Logic) and v.op == "and"
    assert isinstance(v.left, N.Not) and isinstance(v.left.operand, N.Compare)
    assert v.left.operand.right.op == "+"


@pytest.mark.parametrize("src", ["a < b < c", "a == ", "a = = b", "a <> b"])
def test_syntax_errors(src):
    with pytest.raises(MagnumSyntaxError):
        parse(wrap(f"event e {{ do {{ x = {src} }} }}"))


@pytest.mark.parametrize("expr", ["if(a, b)", "if(a, b, c, d)", "if(a, b, else=c)"])
def test_if_arity(expr):
    with pytest.raises(MagnumValidationError, match="if ожидает"):
        load_model(wrap(f"event e {{ do {{ x = {expr} }} }}"))


def test_function_named_if_is_forbidden():
    with pytest.raises(MagnumValidationError):
        load_model(wrap("function if(a) { return a }"))


@pytest.mark.parametrize("expr", ["if(arr, 1, 0)", "arr < 1", "not arr", "arr and 1"])
def test_array_as_condition_is_error(expr):
    with pytest.raises(MagnumRuntimeError):
        calc(expr)
