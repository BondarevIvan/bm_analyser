"""Подмодели: mg.sub — экземпляр другой модели в общем времени."""
import pytest

import magnum as mg
from magnum.errors import MagnumControlError, MagnumRuntimeError, MagnumValidationError


class Cafe(mg.Model):
    masters = mg.param(2, low=0)
    rate = mg.param(10.0, low=0)          # клиентов в час

    @mg.on_start
    def init(self):
        self.clients = mg.Queue()
        self.idle = self.masters
        self.served = 0
        self.arrived = 0

    @mg.arrival(rate="rate", per=mg.HOUR)
    def visit(self):
        self.register()

    def register(self):
        self.arrived += 1
        self.clients.push(self.arrived)

    @mg.when(lambda m: m.idle > 0 and len(m.clients) > 0)
    def start(self):
        self.clients.pop()
        self.idle -= 1
        self.emit(self.done, after=self.rng.exponential(5 * mg.MINUTE))

    def done(self):
        self.idle += 1
        self.served += 1


class Network(mg.Model):
    center = mg.sub(Cafe, masters=3)
    mall = mg.sub(Cafe, rate=4.0)
    walk_in = mg.param(2.0, low=0)        # дополнительные клиенты торгового центра, в час

    @mg.on_start
    def init(self):
        self.total = 0
        self.mall_queue_alarms = 0

    @mg.arrival(rate="walk_in", per=mg.HOUR)
    def extra(self):
        self.emit(self.mall.register)                  # событие подмодели

    @mg.every(mg.DAY)
    def report(self):
        self.total = self.center.served + self.mall.served

    @mg.when(lambda m: len(m.mall.clients) >= 3)     # when на переменную подмодели
    def alarm(self):
        self.mall_queue_alarms += 1
        self.mall.clients.pop()                        # уводим клиента — условие снова ложно


def test_params_are_flattened_with_sub_defaults():
    ps = Network.params
    assert set(ps) == {"center.masters", "center.rate", "mall.masters", "mall.rate", "walk_in"}
    assert ps["center.masters"].default == 3 and ps["mall.rate"].default == 4.0
    assert ps["mall.masters"].default == 2
    assert Cafe.params["masters"].default == 2     # исходная модель не изменилась


def test_run_state_names_and_counts():
    r = Network.run(seed=1, horizon=3, debug=False)
    f = r["final"]
    assert {"center.served", "mall.served", "mall.clients", "total", "walk_in"} <= set(f)
    assert "center" not in f and "mall" not in f
    assert f["center.masters"] == 3
    assert f["total"] == f["center.served"] + f["mall.served"] or f["total"] <= f["center.served"] + f["mall.served"]
    ec = r["event_counts"]
    assert ec["center.visit"] > 0 and ec["mall.visit"] > 0 and ec["extra"] > 0
    assert ec["mall.register"] == ec["extra"]
    # приходы: собственный поток + walk_in
    assert f["mall.arrived"] == ec["mall.visit"] + ec["mall.register"]


def test_controls_for_sub_params():
    a = Network.run(controls={"center.masters": 1}, seed=2, horizon=2, debug=False)["final"]
    b = Network.run(controls={"center.masters": 6}, seed=2, horizon=2, debug=False)["final"]
    assert a["center.masters"] == 1 and b["center.masters"] == 6
    assert len(a["center.clients"]) >= len(b["center.clients"])
    with pytest.raises(MagnumControlError):
        Network.run(controls={"center.nope": 1}, horizon=1)
    with pytest.raises(MagnumControlError):
        Network.run(controls={"mall.masters": -1}, horizon=1)


def test_when_on_sub_variable():
    r = Network.run(controls={"mall.masters": 0}, seed=3, horizon=1, debug=False,
                    sampling="on_change", record=["mall.clients"])
    f = r["final"]
    assert f["mall_queue_alarms"] > 0
    assert len(f["mall.clients"]) <= 2           # alarm не даёт очереди вырасти до 3
    assert max(len(v) for v in r["variables"]["mall.clients"]["value"]) <= 2


def test_sub_alone_equals_standalone_model():
    class Wrap(mg.Model):
        cafe = mg.sub(Cafe)
    a = Wrap.run(seed=5, horizon=2, debug=False)["final"]
    b = Cafe.run(seed=5, horizon=2, debug=False)["final"]
    assert {k.split(".", 1)[1]: v for k, v in a.items()} == b


def test_on_change_recording_of_sub_vars():
    r = Network.run(seed=1, horizon=1, sampling="on_change", record=["center.served", "total"], debug=False)
    s = r["variables"]["center.served"]
    assert s["value"][0] == 0 and s["value"][-1] == r["final"]["center.served"]
    assert s["time"] == sorted(s["time"])
    assert set(r["variables"]) == {"center.served", "total"}


def test_periodic_recording_includes_sub_vars():
    r = Network.run(seed=1, horizon=2, sampling="1d", debug=False)
    assert len(r["variables"]["mall.served"]["value"]) == 3


def test_nested_subs():
    class City(mg.Model):
        north = mg.sub(Network, walk_in=0.0)

        @mg.every(mg.DAY)
        def count(self):
            self.n = self.north.center.served

    assert "north.center.masters" in City.params and City.params["north.walk_in"].default == 0.0
    r = City.run(controls={"north.center.masters": 1}, seed=0, horizon=2, debug=False)
    f = r["final"]
    assert f["north.center.masters"] == 1 and f["n"] <= f["north.center.served"]
    assert r["event_counts"]["north.center.visit"] > 0 and r["event_counts"]["north.extra"] == 0


def test_sub_inits_run_before_outer_init():
    class Outer(mg.Model):
        cafe = mg.sub(Cafe)

        @mg.on_start
        def init(self):
            self.start_idle = self.cafe.idle       # состояние подмодели уже создано

    assert Outer.run(horizon=0, debug=False)["final"]["start_idle"] == 2


def test_string_emit_is_relative_to_caller():
    class Pinger(mg.Model):
        @mg.on_start
        def init(self):
            self.n = 0
            self.emit("ping")

        def ping(self):
            self.n += 1

    class Outer(mg.Model):
        p = mg.sub(Pinger)

        @mg.on_start
        def init(self):
            self.emit("p.ping")

    assert Outer.run(horizon=1, debug=False)["final"]["p.n"] == 2


def test_sub_errors():
    with pytest.raises(MagnumValidationError):
        mg.sub(Cafe, nope=1)
    with pytest.raises(MagnumValidationError):
        mg.sub(int)

    class Bad(mg.Model):
        cafe = mg.sub(Cafe)

        @mg.on_start
        def init(self):
            self.cafe = 1
    with pytest.raises(MagnumRuntimeError, match="подмодель"):
        Bad.run(horizon=1, debug=False)

    class Other(mg.Model):
        cafe = mg.sub(Cafe)

        @mg.on_start
        def init(self):
            self.emit("cafe.nope")
    with pytest.raises(MagnumRuntimeError, match="нет события"):
        Other.run(horizon=1, debug=False)


def test_sub_errors_name_full_event():
    class Boom(mg.Model):
        @mg.on_start
        def init(self):
            raise ValueError("x")

    class Outer(mg.Model):
        b = mg.sub(Boom)
    with pytest.raises(MagnumRuntimeError, match="'b.init'"):
        Outer.run(horizon=1, debug=False)


def test_run_many_with_subs():
    a = mg.run_many(Network, seeds=[1, 2], horizon=1, debug=False)
    b = [Network.run(seed=s, horizon=1, debug=False) for s in (1, 2)]
    assert [x["final"] for x in a] == [x["final"] for x in b]
