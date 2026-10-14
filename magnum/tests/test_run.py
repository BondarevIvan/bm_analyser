"""Запуск моделей: семантика событий, триггеров, функций, массивов и результат."""
import math

import pytest

from magnum import MagnumControlError, MagnumRuntimeError, load_model, run_model
from conftest import ROOT, wrap

COUNTER = """
    event _init { do { n = 0 } trigger { on start } }
"""


def run(body, **kw):
    return run_model(wrap(body), **kw)


def V(r, name):
    """Значения переменной из результата."""
    return r["variables"][name]["value"]


def T(r, name):
    """Моменты времени снимков переменной."""
    return r["variables"][name]["time"]


# --------------------------------------------------------------- результат
def test_result_structure(summator_text):
    r = run_model(summator_text, controls={"value": 2}, seed=7, horizon=10)
    assert set(r) == {"model", "seed", "horizon", "sampling", "controls", "variables", "final", "event_counts"}
    assert r["sampling"] == "1d"
    assert set(r["variables"]) == {"value", "summa"}
    for series in r["variables"].values():
        assert series["time"] == [float(i) for i in range(11)] and len(series["value"]) == 11
    assert r["controls"] == {"value": 2.0}
    assert r["final"]["summa"] == V(r, "summa")[-1]
    assert r["event_counts"]["_init"] == 1 and r["event_counts"]["_accumulator"] == 10


def test_summator_accumulates_until_reset(summator_text):
    r = run_model(summator_text, controls={"value": 3}, seed=0, horizon=200)
    summa = V(r, "summa")
    assert summa[0] == 0
    steps = [b - a for a, b in zip(summa, summa[1:])]
    assert all(s in (0.0, 3.0) for s in steps)
    # после сброса value=0 сумма больше не растёт
    if 0.0 in steps:
        assert all(s == 0.0 for s in steps[steps.index(0.0):])


def test_summator_reset_happens_with_probability_one(summator_text):
    text = summator_text.replace("0.001", "1")
    r = run_model(text, seed=0, horizon=5)
    # день 1: summa += 1, затем choose -> emit _set_value(0) в тот же момент
    assert V(r, "summa") == [0, 1, 1, 1, 1, 1]
    assert V(r, "value") == [1, 0, 0, 0, 0, 0]
    assert r["event_counts"]["_set_value"] == 5


def test_load_from_file():
    from magnum import load_model_file
    m = load_model_file(ROOT / "examples" / "summator.mgn")
    assert m.run(horizon=3)["final"]["summa"] == 3


def test_variable_appears_after_first_assignment():
    r = run("event later { do { x = 1 } trigger { every 2d } }", horizon=4)
    assert T(r, "x") == [2, 3, 4] and V(r, "x") == [1, 1, 1]


# ---------------------------------------------------------------- триггеры
def test_every_first_fire_after_one_period_and_no_float_drift():
    r = run(COUNTER + "event tick { do { n = n + 1 } trigger { every 1m } }", horizon=30)
    assert V(r, "n") == [i * 1440 for i in range(31)]


def test_on_start_runs_before_periodic():
    r = run("""
        event a { do { log = 1 } trigger { on start } }
        event b { do { log = log * 10 + 2 } trigger { every 1d } }
    """, horizon=1)
    assert V(r, "log") == [1, 12]


def test_every_period_from_control():
    r = run("controls { default { p = 2 } }" + COUNTER + "event t { do { n = n + 1 } trigger { every p * 1d } }",
            controls={"p": 0.5}, horizon=3)
    assert r["final"]["n"] == 6


def test_events_beyond_horizon_are_not_processed():
    r = run(COUNTER + "event t { do { n = n + 1 } trigger { every 1d } }", horizon=2.5)
    assert T(r, "n") == [0, 1, 2] and r["final"]["n"] == 2


def test_poisson_arrival_mean():
    r = run(COUNTER + "event visit { do { n = n + 1 } trigger { arrival poisson(lambda=10, per=1d) } }",
            seed=123, horizon=2000)
    rate = r["final"]["n"] / 2000
    assert abs(rate - 10) < 0.3  # стд. ошибка ~ sqrt(10/2000) ≈ 0.07


def test_poisson_arrival_per_hour_and_zero_rate():
    r = run("controls { default { lam = 2 } }" + COUNTER +
            "event v { do { n = n + 1 } trigger { arrival poisson(lambda=lam, per=1h) } }",
            seed=5, horizon=100)
    assert abs(r["final"]["n"] / 100 - 48) < 3
    r = run("controls { default { lam = 2 } }" + COUNTER +
            "event v { do { n = n + 1 } trigger { arrival poisson(lambda=lam, per=1h) } }",
            controls={"lam": 0}, horizon=100)
    assert r["final"]["n"] == 0


# -------------------------------------------------------------- emit / choose
def test_emit_with_parameter():
    r = run("""
        event _init { do { sum = 0 } trigger { on start } }
        event child(param) { do { sum = sum + param } }
        event parent { trigger { every 1h } emit child(3) }
    """, horizon=2)
    assert V(r, "sum") == [0, 72, 144]
    assert r["event_counts"] == {"_init": 1, "child": 48, "parent": 48}


def test_emit_after_delay():
    r = run("""
        event _init { do { a = 0  b = 0 } trigger { on start } }
        event late { do { b = b + 1 } }
        event p { do { a = a + 1 } trigger { every 1d } emit late after 12h }
    """, horizon=3)
    assert V(r, "a") == [0, 1, 2, 3]
    assert V(r, "b") == [0, 0, 1, 2]  # late из дня 3 выполнится в 3.5 — за горизонтом


def test_emit_without_delay_runs_after_current_event_at_same_time():
    r = run("""
        event _init { do { log = 0 } trigger { on start } }
        event child { do { log = log * 10 + 2 } }
        event parent { do { log = log * 10 + 1 } emit child  trigger { every 1d } }
    """, horizon=1)
    assert r["final"]["log"] == 12


def test_section_order_trigger_do_choose_emit():
    r = run("""
        event _init { do { log = 0 } trigger { on start } }
        event from_choose { do { log = log * 10 + 2 } }
        event from_emit { do { log = log * 10 + 3 } }
        event e {
            emit from_emit
            choose { 1: emit from_choose }
            do { log = log * 10 + 1 }
            trigger { every 1d }
        }
    """, horizon=1)
    assert r["final"]["log"] == 123


def test_choose_frequencies():
    body = """
        controls { default { probability_one = 0.3  probability_two = 0.5 } limits { probability_one in [0, 1] } }
        event _init { do { one = 0  two = 0 } trigger { on start } }
        event OneHourDelay { do { one = one + 1 } }
        event TwoHourDelay { do { two = two + 1 } }
        event RandomBranch {
            choose {
                probability_one: emit OneHourDelay after 1h
                probability_two: emit TwoHourDelay after 2h
            }
            trigger { every 1m }
        }
    """
    r = run(body, seed=1, horizon=20)
    n = r["event_counts"]["RandomBranch"]
    assert n == 20 * 1440
    # события, выпущенные после horizon - 2h, не успевают выполниться — допуск это покрывает
    assert abs(r["final"]["one"] / n - 0.3) < 0.015
    assert abs(r["final"]["two"] / n - 0.5) < 0.015


def test_choose_with_instant_probability_and_nothing_branch():
    r = run("""
        event _init { do { k = 0 } trigger { on start } }
        event hit { do { k = k + 1 } }
        event e { do { instant[p] = 0 } choose { instant[p]: emit hit } trigger { every 1h } }
    """, horizon=10)
    assert r["final"]["k"] == 0


def test_multiple_choose_blocks_are_independent():
    r = run("""
        event _init { do { k = 0 } trigger { on start } }
        event hit { do { k = k + 1 } }
        event e { choose { 1: emit hit } choose { 1: emit hit } trigger { every 1d } }
    """, horizon=3)
    assert r["final"]["k"] == 6


def test_choose_probability_sum_above_one_is_error():
    with pytest.raises(MagnumRuntimeError, match="больше 1"):
        run("event a {} event e { choose { 0.7: emit a  0.6: emit a } trigger { every 1d } }", horizon=2)


def test_choose_probability_out_of_range_is_error():
    with pytest.raises(MagnumRuntimeError, match=r"\[0, 1\]"):
        run("event a {} event e { choose { 1.5: emit a } trigger { every 1d } }", horizon=2)


# ------------------------------------------------------- do / функции / instant
def test_functions_and_time():
    r = run("""
        function compute_hour(time) {
            day = floor(time)
            hour = floor((time - day) * 24)
            return hour
        }
        event tick { do { hour = call compute_hour(TIME - 12h)  t = TIME } trigger { every 1h } }
    """, horizon=2)
    assert T(r, "hour") == [1, 2] and V(r, "hour") == [12, 12]
    assert V(r, "t") == [1, 2]


def test_snapshots_are_daily():
    r = run(COUNTER + "event tick { do { n = n + 1 } trigger { every 1h } }", horizon=2.5)
    assert T(r, "n") == [0, 1, 2]
    assert V(r, "n") == [0, 24, 48]
    assert r["final"]["n"] == 60  # final — состояние на самом горизонте


def test_function_locals_do_not_leak_and_can_read_globals():
    r = run("""
        controls { default { k = 10 } }
        function f(x) { tmp = x * k  return tmp + 1 }
        event e { do { y = call f(2) } trigger { on start } }
    """, horizon=0)
    assert r["final"] == {"k": 10, "y": 21}


def test_recursive_function():
    r = run("""
        function fact(n) { return n * call fact_or_one(n - 1) }
        function fact_or_one(n) { return max(1, n) }
        event e { do { y = call fact(5) } trigger { on start } }
    """, horizon=0)
    assert r["final"]["y"] == 20


def test_instant_variables_are_local_to_event_and_not_in_result():
    r = run(COUNTER + """
        event e { do { instant[temp] = 1.5  n = n + instant[temp] } trigger { every 1d } }
    """, horizon=2)
    assert r["variables"] == {"n": {"time": [0, 1, 2], "value": [0, 1.5, 3.0]}}


def test_instant_is_reset_between_occurrences():
    with pytest.raises(MagnumRuntimeError, match="до присваивания"):
        run("""
            event e { do { x = instant[a] } trigger { every 1d } }
            event first { do { instant[a] = 1 } trigger { on start } }
        """, horizon=2)


def test_do_executes_in_order():
    r = run("event e { do { a = 1  b = a + 1  a = b * 10 } trigger { on start } }", horizon=0)
    assert r["final"] == {"a": 20, "b": 2}


def test_controls_can_be_modified_in_do():
    r = run("controls { default { price = 100 } } event e { do { price = price * 1.1 } trigger { every 1d } }",
            horizon=2)
    assert V(r, "price") == pytest.approx([100, 110, 121])
    assert r["controls"] == {"price": 100}


def test_arithmetic():
    r = run("event e { do { a = 2 + 3 * 4  b = (2 + 3) * 4  c = -2^2  d = 2^3^2  e = 7 / 2  f = 2^-1 } "
            "trigger { on start } }", horizon=0)
    assert r["final"] == {"a": 14, "b": 20, "c": -4, "d": 512, "e": 3.5, "f": 0.5}


def test_builtins():
    r = run("event e { do { a = floor(2.7)  b = ceil(2.1)  c = abs(-3)  d = sqrt(16)  e = min(3, 1, 2) "
            " f = max([1, 5], 3)  g = round(2.5)  h = log(exp(2)) } trigger { on start } }", horizon=0)
    assert r["final"] == pytest.approx({"a": 2, "b": 3, "c": 3, "d": 4, "e": 1, "f": 5, "g": 2, "h": 2})


# ------------------------------------------------------------------- массивы
def test_arrays():
    r = run("""
        controls {
            default {
                mat3x3_autogen = create_array(3, 3, value=1)
                mat3x3_manual = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            }
            limits { mat3x3_autogen in [-10, 10] }
        }
        event e {
            do {
                mat3x3_autogen[1][2] = mat3x3_manual[2][0] - 2
                s = sum(mat3x3_manual)
                rows = len(mat3x3_manual)
                row = mat3x3_manual[1]
                row[0] = 100
                instant[v] = [0, 0]
                instant[v][1] = 5
                w = instant[v][1]
            }
            trigger { on start }
        }
    """, horizon=0)
    f = r["final"]
    assert f["mat3x3_autogen"] == [[1, 1, 1], [1, 1, 5], [1, 1, 1]]
    assert f["s"] == 45 and f["rows"] == 3 and f["w"] == 5
    assert f["row"] == [100, 5, 6]
    assert f["mat3x3_manual"][1] == [4, 5, 6]  # присваивание копирует массив


def test_array_snapshots_are_independent():
    r = run("""
        event _init { do { a = create_array(2) } trigger { on start } }
        event e { do { a[0] = a[0] + 1 } trigger { every 1d } }
    """, horizon=2)
    assert V(r, "a") == [[0, 0], [1, 0], [2, 0]]


@pytest.mark.parametrize("expr,match", [
    ("a[3]", "вне диапазона"),
    ("a[0.5]", "целым"),
    ("a + 1", "массив"),
    ("x[0]", "только к массивам"),
])
def test_array_errors(expr, match):
    with pytest.raises(MagnumRuntimeError, match=match):
        run(f"event e {{ do {{ a = [1, 2]  x = 1  y = {expr} }} trigger {{ on start }} }}", horizon=0)


# ------------------------------------------------------------------- контролы
CONTROLS = """
    controls {
        default {
            probability_leave = 0.01
            example = 0.5
            unbounded = 1
            m = create_array(2, 2, value=1)
        }
        limits {
            probability_leave in [0, 1]
            example in (0, 1]
            m in [-10, 10]
        }
    }
"""


def test_controls_override_and_defaults():
    m = load_model(wrap(CONTROLS))
    r = m.run(controls={"example": 1, "unbounded": -1e9, "m": [[0, 1], [2, 3]]}, horizon=0)
    assert r["final"] == {"probability_leave": 0.01, "example": 1.0, "unbounded": -1e9, "m": [[0, 1], [2, 3]]}


@pytest.mark.parametrize("controls,match", [
    ({"example": 0}, "вне допустимого"),           # открытая граница
    ({"probability_leave": 1.01}, "вне допустимого"),
    ({"m": [[0, 1], [2, 30]]}, "вне допустимого"),  # граница на все элементы массива
    ({"m": [1, 2]}, "форма"),
    ({"example": [0.5]}, "форма"),
    ({"nope": 1}, "неизвестный"),
    ({"example": "0.5"}, "тип"),
    ({"example": True}, "bool"),
])
def test_control_errors(controls, match):
    with pytest.raises(MagnumControlError, match=match):
        load_model(wrap(CONTROLS)).run(controls=controls, horizon=0)


def test_input_controls_are_not_mutated():
    m = load_model(wrap(CONTROLS + "event e { do { m[0][0] = 9 } trigger { on start } }"))
    ctrl = {"m": [[1, 1], [1, 1]]}
    r = m.run(controls=ctrl, horizon=0)
    assert ctrl == {"m": [[1, 1], [1, 1]]}
    assert r["final"]["m"][0][0] == 9 and r["controls"]["m"][0][0] == 1
    assert m.defaults["m"] == [[1, 1], [1, 1]]


# --------------------------------------------------------------------- сид
RANDOM_MODEL = """
    controls { default { lam = 5 } }
    event _init { do { n = 0  u = 0 } trigger { on start } }
    event v { do { n = n + 1  u = u + uniform(0, 1) + normal(0, 1) } trigger { arrival poisson(lambda=lam, per=1d) } }
"""


def test_same_seed_same_result():
    m = load_model(wrap(RANDOM_MODEL))
    assert m.run(seed=42, horizon=50) == m.run(seed=42, horizon=50)


def test_different_seeds_differ():
    m = load_model(wrap(RANDOM_MODEL))
    assert m.run(seed=1, horizon=50)["variables"] != m.run(seed=2, horizon=50)["variables"]


def test_model_can_be_run_many_times_independently(summator_text):
    m = load_model(summator_text)
    a = m.run(controls={"value": 5}, horizon=3)
    b = m.run(horizon=3)
    assert a["final"]["summa"] == 15 and b["final"]["summa"] == 3


# -------------------------------------------------------------------- ошибки
def test_runtime_error_undefined_variable_has_line():
    with pytest.raises(MagnumRuntimeError, match="не определена") as ei:
        run("event StartDay {\n do { number_day = number_day + 1 }\n trigger { every 1d } }", horizon=2)
    assert ei.value.line == 3


def test_division_by_zero():
    with pytest.raises(MagnumRuntimeError, match="деление на ноль"):
        run("event e { do { x = 1 / 0 } trigger { on start } }", horizon=0)


def test_infinite_zero_delay_loop_is_detected():
    with pytest.raises(MagnumRuntimeError, match="max_events"):
        run("event a { emit b  trigger { on start } } event b { emit a }", horizon=1, max_events=1000)


def test_negative_delay_is_error():
    with pytest.raises(MagnumRuntimeError, match="after"):
        run("event b {} event a { emit b after -1h  trigger { on start } }", horizon=1)


def test_bad_period_is_error():
    with pytest.raises(MagnumRuntimeError, match="every"):
        run("controls { default { p = 0 } } event a { trigger { every p } }", horizon=1)


def test_bad_run_arguments():
    m = load_model(wrap(""))
    with pytest.raises(ValueError):
        m.run(horizon=-1)


def test_choose_branch_emits_all_events_in_order():
    r = run("""
        event _init { do { log = 0  late = 0 } trigger { on start } }
        event add(d) { do { log = log * 10 + d } }
        event delayed { do { late = late + 1 } }
        event e {
            choose {
                1: emit add(1)
                   emit delayed after 12h
                   emit add(2)
            }
            trigger { every 1d }
        }
    """, horizon=1)
    assert r["final"]["log"] == 12
    assert r["final"]["late"] == 0  # delayed выполнится в 1.5 — за горизонтом
    assert r["event_counts"]["add"] == 2


def test_choose_multi_emit_only_chosen_branch_runs():
    r = run("""
        event _init { do { a = 0  b = 0 } trigger { on start } }
        event inc_a { do { a = a + 1 } }
        event inc_b { do { b = b + 1 } }
        event e {
            choose {
                0.3: emit inc_a
                     emit inc_a
                0.7: emit inc_b
            }
            trigger { every 1m }
        }
    """, seed=3, horizon=10)
    n = r["event_counts"]["e"]
    assert r["final"]["a"] % 2 == 0                      # ветка либо целиком, либо никак
    assert r["final"]["a"] / 2 + r["final"]["b"] == n     # ровно одна ветка на срабатывание
    assert r["final"]["b"] / n == pytest.approx(0.7, abs=0.02)


def test_choose_multi_emit_validation():
    from magnum import MagnumValidationError
    with pytest.raises(MagnumValidationError, match="не объявлено"):
        load_model(wrap("event a {} event e { choose { 1: emit a  emit missing } }"))
