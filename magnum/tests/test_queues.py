"""Встроенные очереди (queue) и ресурсы (resource)."""
import random

import pytest

from magnum import MagnumControlError, MagnumRuntimeError, MagnumValidationError, load_model, parse, run_model
from magnum.queues import Queue
from conftest import wrap


# ------------------------------------------------------------- структура Queue
def test_queue_fifo_and_remove():
    q = Queue()
    for x in [1.0, 2.0, 3.0, 2.0, 4.0]:
        q.push(x)
    assert len(q) == 5 and q.snapshot() == [1, 2, 3, 2, 4]
    assert q.remove(2.0) and q.snapshot() == [1, 3, 2, 4]   # убирается самое раннее вхождение
    assert not q.remove(7.0)
    assert q.contains(2.0) and not q.contains(7.0)
    assert q.pop() == 1 and q.pop() == 3 and q.first() == 2
    q.remove(2.0)
    assert q.pop() == 4 and len(q) == 0
    with pytest.raises(IndexError):
        q.pop()


def test_queue_matches_reference_list():
    rng = random.Random(0)
    q, ref = Queue(), []
    for _ in range(5000):
        op = rng.random()
        x = float(rng.randint(0, 20))
        if op < 0.45:
            q.push(x)
            ref.append(x)
        elif op < 0.7 and ref:
            assert q.pop() == ref.pop(0)
        else:
            assert q.remove(x) == (x in ref)
            if x in ref:
                ref.remove(x)
        assert len(q) == len(ref) and q.snapshot() == ref
        assert all(q.contains(v) for v in ref)


# ------------------------------------------------------------- синтаксис
def test_parse_statements():
    m = parse(wrap("""
        event e { do { clients = queue()  masters = resource(capacity = 2 * k)
                       push clients(1)  x = pop clients  instant[y] = pop clients
                       remove clients(x)  clear clients  seize masters  release masters } }
    """))
    ops = [type(s).__name__ + ":" + getattr(s, "op", "") for s in m.events[0].do]
    assert ops == ["Assign:", "Assign:", "QueueOp:push", "PopAssign:", "PopAssign:", "QueueOp:remove",
                   "QueueOp:clear", "QueueOp:seize", "QueueOp:release"]


# ------------------------------------------------------------- выполнение
SHOP = wrap("""
    controls {
        default {
            clients_per_hour = 10  service = 6  patience = 15  count_masters = 2
            masters = resource(capacity = count_masters)
        }
    }

    event _init { do { clients = queue()  next_id = 0  served = 0  lost = 0  done = 0 } trigger { on start } }

    event Arrive {
        do {
            next_id = next_id + 1
            push clients(next_id)
        }
        emit GiveUp(next_id) after patience * 1m
        trigger { arrival poisson(lambda=clients_per_hour, per=1h) }
    }

    event GiveUp(id) {
        do {
            instant[waiting] = contains(clients, id)
            remove clients(id)
            lost = lost + instant[waiting]
        }
    }

    event StartService {
        trigger { when free(masters) > 0 and size(clients) > 0 }
        do {
            instant[id] = pop clients
            seize masters
            served = served + 1
        }
        emit EndService after exponential(service) * 1m
    }

    event EndService {
        do { release masters  done = done + 1 }
    }
""")


def test_shop_balance_and_limits():
    m = load_model(SHOP)
    for seed in range(5):
        r = m.run(seed=seed, horizon=10, sampling="on_change", record=["clients", "masters"])
        f = r["final"]
        arrived = r["event_counts"]["Arrive"]
        assert arrived == f["served"] + f["lost"] + len(f["clients"])
        assert f["served"] - f["done"] == f["masters"]               # в работе = занятые мастера
        assert all(0 <= b <= 2 for b in r["variables"]["masters"]["value"])
        assert all(len(q) >= 0 for q in r["variables"]["clients"]["value"])
        assert r["event_counts"]["StartService"] == f["served"]       # ни одного холостого события


def test_shop_overload_loses_clients():
    m = load_model(SHOP)
    light = m.run(seed=1, horizon=20, controls={"clients_per_hour": 5})["final"]
    heavy = m.run(seed=1, horizon=20, controls={"clients_per_hour": 40})["final"]
    assert light["lost"] / (light["served"] + light["lost"]) < 0.05
    assert heavy["lost"] / (heavy["served"] + heavy["lost"]) > 0.3


def test_result_contains_plain_values():
    r = run_model(wrap("""
        controls { default { q = queue()  r = resource(capacity = 3) } }
        event e { do { push q(TIME)  seize r } trigger { every 1d } }
        event pop_one { do { instant[x] = pop q  release r } trigger { every 2d } }
    """), horizon=4)
    assert r["variables"]["q"]["value"] == [[], [1], [2], [2, 3], [3, 4]]
    assert r["variables"]["r"]["value"] == [0, 1, 1, 2, 2]
    assert r["final"]["q"] == [3, 4] and r["final"]["r"] == 2


def test_on_change_records_queue_changes_with_time():
    r = run_model(wrap("""
        controls { default { q = queue() } }
        event a { do { push q(1)  push q(2) } trigger { every 1d } }
        event b { do { x = pop q } trigger { every 1.5d } }
    """), horizon=3, sampling="on_change", record=["q"])
    s = r["variables"]["q"]
    assert list(zip(s["time"], s["value"])) == [(0, []), (1, [1, 2]), (1.5, [2]), (2, [2, 1, 2]), (3, [1, 2, 1, 2])]  # в t=3: сначала pop, потом push


def test_functions_and_print(capsys):
    run_model(wrap("""
        controls { default { q = queue()  r = resource(2) } }
        event e {
            do {
                push q(5)  push q(7)  seize r
                print(size(q), first(q), contains(q, 7), contains(q, 1), free(r), busy(r), capacity(r))
                print(q, r)
            }
            trigger { on start }
        }
    """), horizon=0)
    assert capsys.readouterr().out.splitlines() == ["2 5 1 0 1 1 2", "queue[5.0, 7.0] resource(busy=1, capacity=2)"]


def test_clear():
    r = run_model(wrap("""
        controls { default { q = queue() } }
        event a { do { push q(1)  push q(2) } trigger { every 1d } }
        event night { do { clear q } trigger { every 2d } }
    """), horizon=4)
    # в дни 2 и 4 сначала clear (запланирован раньше), потом push
    assert r["variables"]["q"]["value"] == [[], [1, 2], [1, 2], [1, 2, 1, 2], [1, 2]]


Q = "controls { default { q = queue() } } "
R = "controls { default { r = resource(capacity = 1) } } "


@pytest.mark.parametrize("body,match", [
    (Q + "event e { do { x = pop q } trigger { on start } }", "пуста"),
    (Q + "event e { do { x = first(q) } trigger { on start } }", "пуста"),
    (R + "event e { do { seize r  seize r } trigger { on start } }", "нет свободных"),
    (R + "event e { do { release r } trigger { on start } }", "не занята"),
    (Q + "event e { do { push q([1, 2]) } trigger { on start } }", "массив"),
    ("event a { do { push q(1) } trigger { on start } }  event b { do { q = queue() } trigger { every 1d } }",
     "ещё не создан"),
    ("controls { default { k = 1.5 } } event e { do { r = resource(k) } trigger { on start } }", "целым"),
])
def test_runtime_errors(body, match):
    with pytest.raises(MagnumRuntimeError, match=match):
        run_model(wrap(body), horizon=1)


@pytest.mark.parametrize("body,match", [
    (Q + "event e { do { x = q + 1 } }", "очередь"),
    (R + "event e { do { x = r } }", "ресурс"),
    (Q + "event e { do { q = 1 } }", "присвоить можно только"),
    ("event e { do { push missing(1) } }", "не является очередью"),
    (Q + "event e { do { seize q } }", "ресурсом"),
    (R + "event e { do { x = pop r } }", "очередью"),
    (Q + "event e { do { x = free(q) } }", "ресурс"),
    ("event e { do { x = size(nope) } }", "очередь"),
    (Q + "event e { do { x = contains(q) } }", "аргумента"),
    ("controls { default { q = queue() } } event e { do { q = resource(1) } }", "уже объявлен"),
    ("controls { default { q = 1 } } event e { do { q = queue() } }", "числовой контрол"),
    (Q + "event e(q) {}", "параметр"),
    ("controls { default { r = resource(capacity = uniform()) } }", "случайная"),
    ("controls { default { r = resource(size = 2) } }", "capacity"),
    ("controls { default { q = queue(5) } }", "без аргументов"),
    ("controls { default { x = 1 + queue() } }", "только присвоить"),
    ("event e { do { instant[q] = queue() } }", "только переменной"),
    ("event e { do { a = [0]  a[0] = queue() } }", "только переменной"),
    ("function f(x) { q = queue()  return 1 }", "только глобальной"),
    ("event e { do { x = 0 } emit e2(queue()) }  event e2(a) {}", "только присвоить"),
    ("controls { default { q = queue() } limits { q in [0, 1] } }", "ограничения"),
    ("controls { default { r = resource(5) } limits { r in [0, 3] } }", "вне допустимого"),
])
def test_validation_errors(body, match):
    with pytest.raises(MagnumValidationError, match=match):
        load_model(wrap(body))


def test_resource_in_controls_follows_other_controls_and_can_be_overridden():
    m = load_model(wrap("""
        controls { default { n = 2  r = resource(capacity = n * 2) } limits { r in [0, 8] } }
        event e { do { c = capacity(r) } trigger { on start } }
    """))
    assert m.run(horizon=0)["final"]["c"] == 4
    assert m.run(horizon=0, controls={"n": 3})["final"]["c"] == 6     # capacity пересчитана от n
    r = m.run(horizon=0, controls={"r": 7})
    assert r["final"]["c"] == 7 and r["controls"]["r"] == 7           # capacity задана напрямую
    with pytest.raises(MagnumControlError, match="вне допустимого"):
        m.run(horizon=0, controls={"r": 9})
    with pytest.raises(MagnumControlError, match="целое"):
        m.run(horizon=0, controls={"r": 1.5})


def test_queue_in_controls_initial_content():
    m = load_model(wrap("""
        controls { default { q = queue() } }
        event e { do { x = pop q } trigger { on start } }
    """))
    r = m.run(horizon=0, controls={"q": [3, 4]})
    assert r["final"]["x"] == 3 and r["final"]["q"] == [4] and r["controls"]["q"] == [3, 4]
    with pytest.raises(MagnumRuntimeError, match="пуста"):
        m.run(horizon=0)
    with pytest.raises(MagnumControlError, match="список чисел"):
        m.run(horizon=0, controls={"q": 5})


def test_queue_recreated_in_do_is_new_and_empty():
    r = run_model(wrap("""
        event _init { do { q = queue() } trigger { on start } }
        event a { do { push q(TIME) } trigger { every 1d } }
        event night { do { q = queue() } trigger { every 3d } }
    """), horizon=4)
    assert r["variables"]["q"]["value"] == [[], [1], [1, 2], [3], [3, 4]]


def test_each_run_starts_with_fresh_objects():
    m = load_model(wrap("""
        controls { default { q = queue()  r = resource(2) } }
        event e { do { push q(1)  seize r } trigger { on start } }
    """))
    assert m.run(horizon=0)["final"] == m.run(horizon=0)["final"] == {"q": [1], "r": 1}
