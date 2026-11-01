"""Python API: when, отслеживание изменений, очереди и ресурсы, запись результата."""
import pytest

import magnum as mg
from magnum import MagnumRuntimeError, MagnumValidationError


class Warehouse(mg.Model):
    batch = mg.param(5)

    @mg.on_start
    def init(self):
        self.stock = 0
        self.orders = 0
        self.shipped = 0

    @mg.every(3 * mg.DAY)
    def supply(self):
        self.stock += self.batch

    @mg.every(mg.DAY)
    def order(self):
        self.orders += 1

    @mg.when(lambda m: m.stock > 0 and m.orders > 0)
    def ship(self):
        self.stock -= 1
        self.orders -= 1
        self.shipped += 1


def test_when_fires_while_true_without_polling():
    r = Warehouse.run(horizon=9, sampling="on_change", record=["shipped"])
    s = r["variables"]["shipped"]
    assert list(zip(s["time"], s["value"])) == [(0, 0), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    assert r["event_counts"]["ship"] == 9
    assert r["final"] == {"batch": 5, "stock": 6, "orders": 0, "shipped": 9}


def test_when_tracks_list_element_and_method_conditions():
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.slots = [0, 0]
            self.offset = 0
            self.fired = 0

        def total(self):                       # условие через метод — проверяется после каждого события
            return sum(self.slots) + self.offset

        @mg.every(2 * mg.DAY)
        def fill(self):
            self.slots[1] = 1                  # изменение элемента списка видно движку

        @mg.every(5 * mg.DAY)
        def bump(self):
            self.offset += 10

        @mg.when(lambda m: m.total() > 0)
        def fire(self):
            self.slots[1] = 0
            self.offset = 0
            self.fired += 1

    r = M.run(horizon=10, sampling="on_change", record=["fired"])
    assert r["variables"]["fired"]["time"] == [0, 2, 4, 5, 6, 8, 10]


def test_when_without_change_does_not_loop_and_infinite_loop_detected():
    class Once(mg.Model):
        @mg.on_start
        def init(self):
            self.x, self.n = 1, 0

        @mg.when(lambda m: m.x > 0)
        def e(self):
            self.n += 1

    assert Once.run(horizon=10)["final"]["n"] == 1

    class Loop(mg.Model):
        @mg.on_start
        def init(self):
            self.x = 1

        @mg.when(lambda m: m.x > 0)
        def e(self):
            self.x += 1

    with pytest.raises(MagnumRuntimeError, match="max_events"):
        Loop.run(horizon=1, max_events=500)


@pytest.mark.parametrize("cond,match", [
    (lambda m: m.time > 5, "time"),
    (lambda m: m.rng.random() > 0.5, "rng"),
])
def test_when_forbidden_reads(cond, match):
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.x = 1

        @mg.when(cond)
        def e(self):
            pass

    with pytest.raises(MagnumValidationError, match=match):
        M.run(horizon=1)


class Coffee(mg.Model):
    rate = mg.param(20.0, low=0)          # клиентов в час
    masters = mg.param(2, low=0)
    service = mg.param(6.0, low=0, low_open=True)    # мин
    patience = mg.param(5.0, low=0, low_open=True)   # мин

    @mg.on_start
    def init(self):
        self.clients = mg.Queue()
        self.staff = mg.Resource(self.masters)
        self.next_id = self.served = self.lost = self.done = 0

    @mg.arrival(rate="rate", per=mg.HOUR)
    def arrive(self):
        self.next_id += 1
        self.clients.push(self.next_id)
        self.emit(self.give_up, self.next_id, after=self.patience * mg.MINUTE)

    def give_up(self, cid):
        if self.clients.remove(cid):
            self.lost += 1

    @mg.when(lambda m: m.staff.free > 0 and len(m.clients) > 0)
    def start(self):
        self.clients.pop()
        self.staff.seize()
        self.served += 1
        self.emit(self.finish, after=self.rng.exponential(self.service * mg.MINUTE))

    def finish(self):
        self.staff.release()
        self.done += 1


def test_queue_and_resource_with_when():
    for seed in range(3):
        r = Coffee.run(seed=seed, horizon=3, sampling="on_change", record=["staff", "clients"])
        f = r["final"]
        assert r["event_counts"]["arrive"] == f["served"] + f["lost"] + len(f["clients"])
        assert f["served"] - f["done"] == f["staff"]
        assert all(0 <= b <= 2 for b in r["variables"]["staff"]["value"])
        assert r["event_counts"]["start"] == f["served"]           # ни одного холостого срабатывания
        assert isinstance(f["clients"], list)


def test_more_staff_fewer_lost():
    lost = [sum(Coffee.run(seed=s, horizon=5, controls={"masters": m})["final"]["lost"] for s in range(5))
            for m in (1, 2, 3)]
    assert lost[0] > lost[1] > lost[2]


def test_resource_errors():
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.r = mg.Resource(1)
            self.r.seize()
            self.r.seize()

    with pytest.raises(MagnumRuntimeError, match="нет свободных"):
        M.run(horizon=0)


def test_lists_are_copied_on_assignment_and_snapshots_independent():
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.a = [0, 0]
            self.b = self.a          # копия, а не та же самая ссылка

        @mg.every(mg.DAY)
        def inc(self):
            self.a[0] += 1

    r = M.run(horizon=2)
    assert r["variables"]["a"]["value"] == [[0, 0], [1, 0], [2, 0]]
    assert r["final"]["b"] == [0, 0]


def test_on_change_collapses_same_time_and_records_param():
    class M(mg.Model):
        k = mg.param(1.0)

        @mg.on_start
        def init(self):
            self.x = 0

        @mg.every(mg.DAY)
        def e(self):
            self.x += 5
            self.x += 2
            self.emit(self.back)

        def back(self):
            self.x += 1

    r = M.run(horizon=2, sampling="on_change")
    assert list(zip(r["variables"]["x"]["time"], r["variables"]["x"]["value"])) == [(0, 0), (1, 8), (2, 16)]
    assert r["variables"]["k"] == {"time": [0.0], "value": [1.0]}


def test_periodic_sampling_and_record_filter():
    r = Warehouse.run(horizon=2, sampling="12h", record=["orders"])
    assert set(r["variables"]) == {"orders"}
    assert r["variables"]["orders"]["time"] == [0, 0.5, 1, 1.5, 2]


def test_rng_functions_and_validation():
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.cat = [0, 0, 0]
            self.u = self.n = self.e = 0.0
            self.b = 0

        @mg.every(mg.MINUTE)
        def draw(self):
            self.cat[self.rng.categorical([1, 0, 3])] += 1
            self.u += self.rng.uniform(2, 4)
            self.n += self.rng.normal(10, 2)
            self.e += self.rng.exponential(5)
            self.b += self.rng.bernoulli(0.3)

    f = M.run(horizon=5, seed=1, sampling="1w")["final"]
    n = 5 * 1440
    assert f["cat"][1] == 0 and f["cat"][2] / n == pytest.approx(0.75, abs=0.02)
    assert f["u"] / n == pytest.approx(3, abs=0.03) and f["n"] / n == pytest.approx(10, abs=0.1)
    assert f["e"] / n == pytest.approx(5, rel=0.05) and f["b"] / n == pytest.approx(0.3, abs=0.02)
    rng = mg.RandomSource(0)
    with pytest.raises(ValueError):
        rng.categorical([0, 0])
    with pytest.raises(ValueError):
        rng.exponential(0)


def test_run_many_parallel_equals_sequential():
    seq = mg.run_many(Shop, [1, 2, 3], horizon=5, debug=False)
    par = mg.run_many(Shop, [1, 2, 3], processes=2, horizon=5, debug=False)
    assert [r["final"] for r in seq] == [r["final"] for r in par]


class Shop(mg.Model):
    @mg.on_start
    def init(self):
        self.n = 0

    @mg.arrival(rate=24.0)
    def v(self):
        self.n += 1
