"""Смысловые проверки при загрузке модели."""
import pytest

from magnum import MagnumValidationError, load_model
from conftest import wrap


@pytest.mark.parametrize("body,match", [
    ("event e { emit nope }", "не объявлено"),
    ("event a(x) {}  event e { emit a }", "ожидает 1"),
    ("event a {}  event e { choose { 0.5: emit a(1, 2) } }", "ожидает 0"),
    ("event a(x) { trigger { every 1d } }", "с параметрами"),
    ("event e { do { x = foo(1) } }", "неизвестная встроенная функция"),
    ("function f(x) { return x }  event e { do { y = f(1) } }", "call"),
    ("event e { do { y = call floor(1) } }", "не объявлена"),
    ("function f(x) { return x }  event e { do { y = call f(1, 2) } }", "ожидает 1"),
    ("function f(x) { y = x }", "нет return"),
    ("function f(x) { return instant[a] }", "instant"),
    ("event e(p) { do { p = 1 } }", "параметр"),
    ("event e { do { TIME = 1 } }", "зарезервированное"),
    ("function f(x) { TIME = x  return x }", "зарезервированное"),
    ("function f(TIME) { return 1 }", "зарезервированное"),
    ("event e(TIME) {}", "зарезервированное"),
    ("event e { do { inf = 1 } }", "зарезервированное"),
    ("event e {}  event e {}", "дважды"),
    ("function f() { return 1 }  function f() { return 2 }", "дважды"),
    ("function floor(x) { return x }", "встроенной"),
    ("event e(a, a) {}", "повторяющиеся"),
    ("event e { do { x = floor(1, 2) } }", "неверные аргументы"),
    ("event e { do { x = create_array(2, size=3) } }", "неверные аргументы"),
    ("event e { trigger { arrival gamma(1) } }", "неизвестное распределение"),
    ("event e { trigger { arrival poisson(rate=1) } }", "неверные аргументы"),
    ("event e { trigger { every instant[x] } }", "instant"),
    ("controls { default { a = 1 } limits { b in [0, 1] } }", "не объявлен"),
    ("controls { default { a = 5 } limits { a in [0, 1] } }", "вне допустимого"),
    ("controls { default { a = 1 } limits { a in (1, 2] } }", "вне допустимого"),
    ("controls { default { a = 1 } limits { a in [2, 1] } }", "больше верхней"),
    ("controls { default { a = 1 a = 2 } }", "дважды"),
    ("controls { default { a = uniform(0, 1) } }", "случайная"),
    ("controls { default { a = b } }", "не определена"),
    ("controls { default { TIME = 1 } }", "зарезервированное"),
    ("controls { default { m = create_array(2, 2, value=20) } limits { m in [-10, 10] } }", "вне допустимого"),
    ("controls { default { a = 1 } }  controls { default { b = 1 } }", "одна"),
])
def test_validation_errors(body, match):
    with pytest.raises(MagnumValidationError, match=match):
        load_model(wrap(body))


def test_validation_error_has_line():
    src = wrap("event e {\n  emit missing\n}")
    with pytest.raises(MagnumValidationError) as ei:
        load_model(src)
    assert ei.value.line == 3  # +1 строка от wrap


def test_defaults_can_reference_previous_defaults():
    m = load_model(wrap("controls { default { a = 2  b = a * 3  m = create_array(2, value=b) } }"))
    assert m.defaults == {"a": 2.0, "b": 6.0, "m": [6.0, 6.0]}


def test_functions_may_reference_each_other_in_any_order():
    load_model(wrap("""
        function g(x) { return call f(x) + 1 }
        function f(x) { return x * 2 }
    """))


def test_time_may_be_function_parameter():
    # как в README: function compute_hour(time)
    load_model(wrap("function compute_hour(time) { day = floor(time) return floor((time - day) * 24) }"))


def test_lowercase_time_is_ordinary_variable():
    from magnum import run_model
    r = run_model(wrap("event e { do { time = 5  t = TIME } trigger { every 1d } }"), horizon=2)
    assert r["final"] == {"time": 5, "t": 2}
