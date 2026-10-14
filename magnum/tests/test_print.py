"""Отладочный вывод print(...) в do и в функциях."""
import pytest

from magnum import MagnumRuntimeError, MagnumSyntaxError, MagnumValidationError, load_model, parse, run_model
from magnum import nodes as N
from conftest import wrap

MODEL = wrap("""
    controls { default { price = 2.5 } }
    function twice(x) { print("twice:", x) return x * 2 }
    event _init { do { n = 0  arr = [1, 2.5] } trigger { on start } }
    event tick {
        do {
            n = n + 1
            instant[t] = call twice(n)
            print("день", TIME, "n =", n, instant[t], price, arr)
        }
        trigger { every 1d }
    }
""")


def test_print_output(capsys):
    run_model(MODEL, horizon=2)
    assert capsys.readouterr().out.splitlines() == [
        "twice: 1",
        "день 1 n = 1 2 2.5 [1, 2.5]",
        "twice: 2",
        "день 2 n = 2 4 2.5 [1, 2.5]",
    ]


def test_print_disabled_with_debug_false(capsys):
    r = load_model(MODEL).run(horizon=2, debug=False)
    assert capsys.readouterr().out == ""
    assert r["final"]["n"] == 2  # на расчёт не влияет


def test_print_empty_and_order(capsys):
    run_model(wrap('event e { do { print()  x = 1  print("x", x)  x = 2  print("x", x) } trigger { on start } }'),
              horizon=0)
    assert capsys.readouterr().out == "\nx 1\nx 2\n"


def test_print_string_escapes(capsys):
    run_model(wrap(r'event e { do { print("кавычка \" тут") } trigger { on start } }'), horizon=0)
    assert capsys.readouterr().out == 'кавычка " тут\n'


def test_print_parse_tree():
    m = parse(wrap('event e { do { print("a", x + 1) } }'))
    p = m.events[0].do[0]
    assert isinstance(p, N.Print) and p.args[0] == "a" and p.args[1].op == "+"


def test_print_not_a_value():
    # print — команда, а не функция: значения не возвращает
    with pytest.raises((MagnumSyntaxError, MagnumValidationError)):
        load_model(wrap('event e { do { x = print(1) } }'))


def test_print_reports_errors_in_arguments():
    with pytest.raises(MagnumRuntimeError, match="не определена"):
        run_model(wrap("event e { do { print(missing) } trigger { on start } }"), horizon=0)
