"""Тесты кофейни (models/coffee/coffee_queue.py). Запуск из корня репозитория: pytest models/coffee/tests"""
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.coffee.coffee_queue import CoffeeQueue  # noqa: E402

CONTROLS = {
    "average_wait": 60,
    "std_wait": 5,
    "average_count_clients_per_1h": 10,
    "open_hour": 8,
    "close_hour": 21,
}


def visit_times():
    r = CoffeeQueue.run(controls=CONTROLS, horizon=300, sampling="on_change",
                        record=["last_client_id"], debug=False)
    return r["variables"]["last_client_id"]["time"][1:]


def test_client_visits_exponential():
    """Интервалы между приходами внутри дня — экспоненциальные с интенсивностью average_count_clients_per_1h."""
    grouped = {}
    for t in visit_times():
        grouped.setdefault(int(t), []).append(t)
    diffs = [r - l for lt in grouped.values() for l, r in zip(lt, lt[1:])]
    rate_per_day = 24 * CONTROLS["average_count_clients_per_1h"]
    _, p_value = stats.kstest(diffs, "expon", args=(0, 1 / rate_per_day))
    assert p_value > 0.001


def test_client_visits_only_when_open():
    frac = [t - int(t) for t in visit_times()]
    assert min(frac) > CONTROLS["open_hour"] / 24
    assert max(frac) < CONTROLS["close_hour"] / 24


def test_conservation():
    """Каждый пришедший либо обслужен, либо ушёл, либо ещё в очереди; мастера не теряются."""
    r = CoffeeQueue.run(seed=3, horizon=10, debug=False)
    f = r["final"]
    assert f["count_served"] + f["count_lost"] + len(f["clients"]) == f["last_client_id"]
    assert f["count_served"] - f["count_processed"] == CoffeeQueue.count_masters.default - f["idle_masters"]


def test_more_masters_fewer_lost():
    def lost(n):
        return sum(CoffeeQueue.run(controls={"count_masters": n}, seed=s, horizon=7, debug=False)["final"]["count_lost"]
                   for s in range(5))
    assert lost(1) > lost(2) > lost(4)
