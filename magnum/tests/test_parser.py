"""Разбор текста: грамматика и построение AST."""
import math

import pytest

from magnum import MagnumSyntaxError, parse
from magnum import nodes as N
from conftest import wrap


def first_assign_value(src_expr: str):
    m = parse(wrap(f"event e {{ do {{ x = {src_expr} }} }}"))
    return m.events[0].do[0].value


# ------------------------------------------------------------ примеры из README
def test_readme_full_model(summator_text):
    m = parse(summator_text)
    assert m.name == "Простой пример сумматора"
    assert [e.name for e in m.events] == ["_init", "_set_value", "_accumulator"]
    ctrl = m.controls[0]
    assert [d.name for d in ctrl.defaults] == ["value"]
    lim = ctrl.limits[0]
    assert (lim.name, lim.lower_closed, lim.upper_closed) == ("value", True, True)
    acc = m.events[2]
    assert isinstance(acc.triggers[0], N.Every)
    assert len(acc.choose) == 1 and acc.choose[0][0].emits[0].event == "_set_value"
    assert m.events[1].params == ("new_value",)


README_SNIPPETS = [
    """
    event StartDay {
        do {
            number_day = number_day + 1
        }
        trigger {
            every 1d
        }
    }""",
    """
    event VisitClient {
        do {
            count_client = count_client + 1
        }
        trigger {
            arrival poisson(lambda=1, per=1d)
        }
    }""",
    """
    event child(param) {
        do {
            sum = sum + param
        }
    }

    event parent {
        trigger {
            every 1h
        }
        emit child(3)
    }""",
    """
    event OneHourDelay {}
    event TwoHourDelay {}

    event RandomBranch {
        choose {
            probability_one: emit OneHourDelay after 1h
            probability_two: emit TwoHourDelay after 2h
        }
        trigger {
            every 1m
        }
    }""",
    """
    # time передается в днях
    function compute_hour(time) {
        day = floor(time)
        hour = floor((time - day) * 24)
        return hour
    }""",
    """
    event e {
        do {
            # мгновенные переменные описываются как instant[temp]
            instant[temp] = 1.5
            sum = sum + instant[temp]
            hour = call compute_hour(instant[temp])
        }
    }""",
    """
    controls {
        default {
            average_client_per_hour = 6
            probability_leave = 0.01
            example = 0.5
            mat3x3_autogen = create_array(3, 3, value=1)
            mat3x3_manual = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
        limits {
            probability_leave in [0, 1]
            example in (0, 1]
            mat3x3_autogen in [-10, 10]
        }
    }""",
]


@pytest.mark.parametrize("snippet", README_SNIPPETS)
def test_readme_snippets_parse(snippet):
    parse(wrap(snippet))


# ----------------------------------------------------------------- выражения
@pytest.mark.parametrize("text,days", [("1d", 1), ("2d", 2), ("1h", 1 / 24), ("30m", 30 / 1440),
                                       ("1w", 7), ("90s", 90 / 86400), ("1.5h", 1.5 / 24)])
def test_durations_in_days(text, days):
    v = first_assign_value(text)
    assert isinstance(v, N.Number) and math.isclose(v.value, days)


def test_precedence_and_associativity():
    v = first_assign_value("1 + 2 * 3")
    assert v.op == "+" and v.right.op == "*"
    v = first_assign_value("-2^2")          # -(2^2)
    assert isinstance(v, N.Neg) and v.operand.op == "^"
    v = first_assign_value("2^3^2")         # 2^(3^2)
    assert v.op == "^" and v.right.op == "^"
    v = first_assign_value("10 - 3 - 2")    # (10-3)-2
    assert v.op == "-" and v.left.op == "-"
    v = first_assign_value("2^-1")
    assert v.op == "^" and isinstance(v.right, N.Neg)


def test_calls_and_indexing():
    v = first_assign_value("call f(a[0][1], 2) + floor(x)")
    assert isinstance(v.left, N.Call) and v.left.user
    assert isinstance(v.left.args[0], N.Index) and isinstance(v.left.args[0].target, N.Index)
    assert isinstance(v.right, N.Call) and not v.right.user
    v = first_assign_value("create_array(2, 3, value=1.5)")
    assert len(v.args) == 2 and v.kwargs[0][0] == "value"


def test_assignment_targets():
    m = parse(wrap("event e { do { a[1][2] = 3  instant[t][0] = 1 } }"))
    t1, t2 = (a.target for a in m.events[0].do)
    assert isinstance(t1, N.Index) and isinstance(t1.target, N.Index) and t1.target.target == N.Var("a")
    assert isinstance(t2, N.Index) and t2.target == N.Instant("t")


def test_limits_open_closed_and_inf():
    m = parse(wrap("controls { default { a = 1 b = 1 } limits { a in (0, 1] b in [0, inf) } }"))
    a, b = m.controls[0].limits
    assert (a.lower_closed, a.upper_closed) == (False, True)
    assert (b.lower_closed, b.upper_closed) == (True, False) and b.upper == N.Var("inf")


def test_emit_forms():
    m = parse(wrap("event a(x) {}  event b {}  event e { emit b  emit a(1) after 2h  emit b after x * 1d }"))
    e1, e2, e3 = m.events[2].emits
    assert e1.args == () and e1.delay is None
    assert len(e2.args) == 1 and math.isclose(e2.delay.value, 1 / 12)
    assert e3.delay.op == "*"


def test_comments_and_unicode_names():
    m = parse(wrap("""
        # комментарий
        event Покупка {   # ещё комментарий
            do { выручка = выручка + 1 }
        }
    """))
    assert m.events[0].name == "Покупка"
    assert m.events[0].do[0].target == N.Var("выручка")


def test_builtin_function_names_usable_as_variables():
    # как в README: переменная sum и встроенная функция sum() живут в разных пространствах имён
    m = parse(wrap("event e { do { sum = sum + floor(1)  len = sum(a) } }"))
    assert m.events[0].do[0].target == N.Var("sum")


def test_empty_model_and_event():
    m = parse('model "пустая" {}')
    assert m.events == () and m.controls == ()
    parse(wrap("event e {}"))


# --------------------------------------------------------------- ошибки
@pytest.mark.parametrize("src", [
    'model "x" { event e { do { x = } } }',
    'model "x" { event e { do { x = 1 + } } }',
    'model "x" { event e { trigger { sometimes 1d } } }',
    'model "x" { event e { do { x = 1 }',
    'model "x" { event e { do { x = 1 $ 2 } } }',
    'event e {}',
])
def test_syntax_errors(src):
    with pytest.raises(MagnumSyntaxError):
        parse(src)


def test_syntax_error_has_position():
    src = 'model "x" {\n  event e {\n    do { x = * 2 }\n  }\n}'
    with pytest.raises(MagnumSyntaxError) as ei:
        parse(src)
    assert ei.value.line == 3


def test_duplicate_do_section():
    with pytest.raises(MagnumSyntaxError, match="do"):
        parse(wrap("event e { do { x = 1 } do { y = 2 } }"))


def test_positional_after_keyword():
    with pytest.raises(MagnumSyntaxError):
        parse(wrap("event e { do { x = create_array(value=1, 3) } }"))


def test_choose_branch_with_several_emits():
    m = parse(wrap("""
        event a {} event b(x) {} event c {}
        event e {
            choose {
                p: emit a after 1h
                   emit b(2)
                   emit c
                q: emit c
            }
        }
    """))
    p_branch, q_branch = m.events[3].choose[0]
    assert [x.event for x in p_branch.emits] == ["a", "b", "c"]
    assert p_branch.emits[0].delay is not None and p_branch.emits[1].args[0] == N.Number(2.0)
    assert [x.event for x in q_branch.emits] == ["c"]
