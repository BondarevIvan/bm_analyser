"""Политики снятия состояния: периодическая ('1d', '6h', ...) и 'on_change'."""
import pytest

from magnum import load_model, run_model
from conftest import wrap

COUNTER = "event _init { do { n = 0 } trigger { on start } }\n"
HOURLY = COUNTER + "event tick { do { n = n + 1 } trigger { every 1h } }"


def series(r, name):
    s = r["variables"][name]
    return list(zip(s["time"], s["value"]))


@pytest.mark.parametrize("spec,times", [
    ("1d", [0, 1, 2]),
    ("12h", [0, 0.5, 1, 1.5, 2]),
    ("0.5d", [0, 0.5, 1, 1.5, 2]),
    ("720m", [0, 0.5, 1, 1.5, 2]),
    (" 1D ", [0, 1, 2]),
    (1, [0, 1, 2]),          # число — период в днях
    ("1w", [0]),
])
def test_periodic_times(spec, times):
    r = run_model(wrap(HOURLY), horizon=2, sampling=spec)
    assert r["variables"]["n"]["time"] == pytest.approx(times)
    assert r["variables"]["n"]["value"] == pytest.approx([t * 24 for t in times])


def test_periodic_snapshot_includes_events_at_that_moment():
    r = run_model(wrap(HOURLY), horizon=1, sampling="6h")
    assert series(r, "n") == [(0, 0), (0.25, 6), (0.5, 12), (0.75, 18), (1.0, 24)]


def test_on_change_records_only_changes_with_time():
    r = run_model(wrap("""
        controls { default { price = 100 } }
        event _init { do { n = 0 } trigger { on start } }
        event tick { do { n = n + 1 } trigger { every 1d } }
        event raise_price { do { price = price + 10 } trigger { every 3d } }
        event same { do { n = n } trigger { every 1h } }
    """), horizon=7, sampling="on_change")
    assert series(r, "price") == [(0, 100), (3, 110), (6, 120)]
    assert series(r, "n") == [(0, 0)] + [(float(d), d) for d in range(1, 8)]


def test_on_change_exact_event_times():
    r = run_model(wrap("""
        event _init { do { x = 0 } trigger { on start } }
        event set(v) { do { x = v } }
        event e { emit set(1) after 3h  emit set(2) after 30m  trigger { on start } }
    """), horizon=1, sampling="on_change")
    assert r["variables"]["x"]["time"] == pytest.approx([0, 0.5 / 24, 3 / 24])
    assert r["variables"]["x"]["value"] == [0, 2, 1]


def test_on_change_collapses_changes_at_same_time():
    r = run_model(wrap("""
        event _init { do { x = 0 } trigger { on start } }
        event e { do { x = x + 5  x = x + 2 } emit back  trigger { every 1d } }
        event back { do { x = x + 1 } }
    """), horizon=2, sampling="on_change")
    # в момент t: e делает +5, +2, сразу за ним back +1; сохраняется только итог
    assert series(r, "x") == [(0, 0), (1, 8), (2, 16)]


def test_on_change_change_and_revert_at_same_time_is_not_recorded():
    r = run_model(wrap("""
        controls { default { x = 1 } }
        event e { do { x = 2 } emit back  trigger { every 1d } }
        event back { do { x = 1 } }
    """), horizon=3, sampling="on_change")
    assert series(r, "x") == [(0, 1)]


def test_on_change_array_element_assignment():
    r = run_model(wrap("""
        controls { default { a = create_array(2) } }
        event e { do { a[1] = a[1] + 1 } trigger { every 2d } }
    """), horizon=4, sampling="on_change")
    assert series(r, "a") == [(0, [0, 0]), (2, [0, 1]), (4, [0, 2])]


def test_on_change_new_variable_and_final():
    r = run_model(wrap("event later { do { x = 1 } trigger { every 2d } }"), horizon=5, sampling="on_change")
    assert series(r, "x") == [(2, 1)]
    assert r["final"] == {"x": 1} and r["sampling"] == "on_change"


def test_on_change_same_as_periodic_final_state():
    text = wrap("""
        controls { default { lam = 3 } }
        event _init { do { n = 0 } trigger { on start } }
        event v { do { n = n + 1 } trigger { arrival poisson(lambda=lam, per=1d) } }
    """)
    m = load_model(text)
    a, b = m.run(seed=3, horizon=30), m.run(seed=3, horizon=30, sampling="on_change")
    assert a["final"] == b["final"]
    assert b["variables"]["n"]["value"] == list(range(len(b["variables"]["n"]["value"])))  # +1 на каждом шаге


@pytest.mark.parametrize("spec", ["", "day", "1x", "-1d", "0d", "on change", None, True, 0, [1]])
def test_bad_sampling(spec):
    with pytest.raises(ValueError, match="sampling"):
        run_model(wrap(HOURLY), horizon=1, sampling=spec)


@pytest.mark.parametrize("spec", ["1d", "on_change"])
def test_record_filters_variables(spec):
    text = wrap(COUNTER + "event tick { do { n = n + 1  m = n * 2  big = create_array(100) } trigger { every 1h } }")
    full = run_model(text, horizon=2, sampling=spec)
    part = run_model(text, horizon=2, sampling=spec, record=["n"])
    assert set(part["variables"]) == {"n"}
    assert part["variables"]["n"] == full["variables"]["n"]
    assert part["final"] == full["final"]           # final — всегда все переменные
    assert run_model(text, horizon=2, sampling=spec, record="m")["variables"].keys() == {"m"}
    assert run_model(text, horizon=2, sampling=spec, record=[])["variables"] == {}
