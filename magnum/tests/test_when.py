"""Триггер when: событие срабатывает, пока условие истинно, без опроса."""
import pytest

from magnum import MagnumRuntimeError, MagnumValidationError, load_model, run_model
from conftest import wrap

# Склад: поставка раз в 3 дня, отгрузка срабатывает по условию «есть товар и есть заказ»
WAREHOUSE = wrap("""
    controls { default { batch = 5 } }
    event _init { do { stock = 0  orders = 0  shipped = 0 } trigger { on start } }
    event Supply { do { stock = stock + batch } trigger { every 3d } }
    event Order { do { orders = orders + 1 } trigger { every 1d } }
    event Ship {
        trigger { when stock > 0 and orders > 0 }
        do { stock = stock - 1  orders = orders - 1  shipped = shipped + 1 }
    }
""")


def test_when_fires_while_condition_true():
    r = run_model(WAREHOUSE, horizon=9, sampling="on_change", record=["shipped", "stock", "orders"])
    v = r["variables"]
    # день 3: поставка 5 шт., накоплено 3 заказа -> 3 отгрузки в тот же момент
    assert list(zip(v["shipped"]["time"], v["shipped"]["value"])) == [
        (0, 0), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    assert r["final"] == {"batch": 5, "stock": 6, "orders": 0, "shipped": 9}
    assert r["event_counts"]["Ship"] == 9


def test_when_without_polling_number_of_events():
    r = run_model(WAREHOUSE, horizon=300, debug=False)
    # событий ровно столько, сколько полезной работы: никаких проверок «впустую»
    assert r["event_counts"]["Ship"] == r["final"]["shipped"]


def test_when_checks_after_on_start():
    r = run_model(wrap("""
        event _init { do { x = 1  fired = 0 } trigger { on start } }
        event e { trigger { when x > 0 } do { x = 0  fired = fired + 1 } }
    """), horizon=5)
    assert r["final"]["fired"] == 1


def test_when_reacts_to_array_element_and_function_dependencies():
    r = run_model(wrap("""
        function total(a) { return sum(a) + offset }
        controls { default { offset = 0 } }
        event _init { do { slots = [0, 0]  fired = 0 } trigger { on start } }
        event fill { do { slots[1] = 1 } trigger { every 2d } }
        event bump { do { offset = offset + 10 } trigger { every 5d } }
        event e {
            trigger { when call total(slots) > 0 }
            do { slots[1] = 0  offset = 0  fired = fired + 1 }
        }
    """), horizon=10, sampling="on_change", record=["fired"])
    times = r["variables"]["fired"]["time"]
    # срабатывает в моменты заполнения ячейки (2, 4, 6, 8, 10) и роста offset (5)
    assert times == [0, 2, 4, 5, 6, 8, 10]


def test_when_does_not_fire_if_its_event_does_not_change_dependencies():
    # условие всегда истинно, но событие не меняет переменные условия -> одно срабатывание, без зацикливания
    r = run_model(wrap("""
        event _init { do { x = 1  n = 0 } trigger { on start } }
        event e { trigger { when x > 0 } do { n = n + 1 } }
    """), horizon=10)
    assert r["final"]["n"] == 1


def test_when_combined_with_other_triggers():
    r = run_model(wrap("""
        event _init { do { x = 0  n = 0 } trigger { on start } }
        event inc { do { x = x + 1 } trigger { every 1d } }
        event e { trigger { when x >= 3  every 10d } do { x = 0  n = n + 1 } }
    """), horizon=10)
    # по условию — в дни 3, 6, 9; по every — день 10
    assert r["final"]["n"] == 4


def test_when_infinite_loop_detected():
    with pytest.raises(MagnumRuntimeError, match="max_events"):
        run_model(wrap("""
            event _init { do { x = 1 } trigger { on start } }
            event e { trigger { when x > 0 } do { x = x + 1 } }
        """), horizon=1, max_events=1000)


def test_when_same_seed_same_result():
    text = wrap("""
        event _init { do { q = 0  served = 0  free = 2 } trigger { on start } }
        event arrive { do { q = q + 1 } trigger { arrival poisson(lambda=20, per=1d) } }
        event start { trigger { when free > 0 and q > 0 } do { q = q - 1  free = free - 1 } emit done after exponential(1h) }
        event done { do { free = free + 1  served = served + 1 } }
    """)
    m = load_model(text)
    a, b = m.run(seed=4, horizon=30), m.run(seed=4, horizon=30)
    assert a == b
    assert a["final"]["served"] > 500 and 0 <= a["final"]["free"] <= 2


@pytest.mark.parametrize("body,match", [
    ("event e { trigger { when TIME > 5 } }", "TIME"),
    ("event e { trigger { when 1 > 0 } }", "не зависит"),
    ("event e { trigger { when uniform() > 0.5 } }", "случайная"),
    ("event e { trigger { when instant[a] > 0 } }", "instant"),
    ("event e(p) { trigger { when x > 0 } }", "с параметрами"),
])
def test_when_validation(body, match):
    with pytest.raises(MagnumValidationError, match=match):
        load_model(wrap(body))
