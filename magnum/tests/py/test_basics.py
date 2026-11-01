"""Python API: параметры, триггеры, emit, результат."""
import pytest

import magnum as mg
from magnum import MagnumControlError, MagnumRuntimeError, MagnumValidationError


class Summator(mg.Model):
    """Пример из README: каждый день прибавляем value; с малой вероятностью value обнуляется."""
    value = mg.param(1.0, low=-10, high=100)
    p_reset = mg.param(0.0, low=0, high=1)

    @mg.on_start
    def init(self):
        self.summa = 0.0

    @mg.every(mg.DAY)
    def accumulate(self):
        self.summa += self.value
        self.choose([(self.p_reset, lambda: self.emit(self.set_value, 0.0))])

    def set_value(self, new_value):
        self.value = new_value


def test_result_structure():
    r = Summator.run(controls={"value": 2}, seed=7, horizon=10)
    assert set(r) == {"model", "seed", "horizon", "sampling", "controls", "variables", "final", "event_counts",
                      "log_prob"}
    assert r["model"] == "Summator" and {k: v.item() for k, v in r["controls"].items()} == {"value": 2.0, "p_reset": 0.0}
    assert r["variables"]["summa"]["time"] == [float(i) for i in range(11)]
    assert r["variables"]["summa"]["value"] == [2.0 * i for i in range(11)]
    assert r["final"] == {"value": 2.0, "p_reset": 0.0, "summa": 20.0}
    assert r["event_counts"] == {"init": 1, "accumulate": 10}  # методы без триггера — только если вызывались


def test_choose_and_emit_with_args():
    r = Summator.run(controls={"p_reset": 1}, seed=0, horizon=5)
    assert r["variables"]["summa"]["value"] == [0, 1, 1, 1, 1, 1]
    assert r["variables"]["value"]["value"] == [1, 0, 0, 0, 0, 0]


class Shop(mg.Model):
    rate = mg.param(10.0, low=0)

    @mg.on_start
    def init(self):
        self.n = 0
        self.log = []

    @mg.arrival(rate="rate", per=mg.HOUR)
    def visit(self):
        self.n += 1


def test_poisson_arrival_mean_and_seed():
    a = Shop.run(seed=3, horizon=100)
    assert abs(a["final"]["n"] / 100 / 24 - 10) < 0.4
    assert a == Shop.run(seed=3, horizon=100)
    assert a["final"] != Shop.run(seed=4, horizon=100)["final"]
    assert Shop.run(controls={"rate": 0}, horizon=10)["final"]["n"] == 0


class Order(mg.Model):
    @mg.on_start
    def init(self):
        self.log = 0

    @mg.every(mg.DAY)
    def parent(self):
        self.log = self.log * 10 + 1
        self.emit(self.child)                 # тот же момент, после текущего
        self.emit("late", 7, after=12 * mg.HOUR)

    def child(self):
        self.log = self.log * 10 + 2

    def late(self, x):
        self.log = self.log * 10 + x


def test_emit_order_and_delay():
    r = Order.run(horizon=1.6, sampling="12h")
    assert r["variables"]["log"]["value"] == [0, 0, 12, 127]


def test_every_without_float_drift_and_on_start_first():
    class M(mg.Model):
        @mg.on_start
        def init(self):
            self.n = 0

        @mg.every(mg.MINUTE)
        def tick(self):
            self.n += 1

    assert M.run(horizon=30)["variables"]["n"]["value"] == [i * 1440 for i in range(31)]


def test_every_period_from_param():
    class M(mg.Model):
        p = mg.param(2.0, low=0, low_open=True)

        @mg.on_start
        def init(self):
            self.n = 0

        @mg.every(lambda m: m.p)
        def tick(self):
            self.n += 1

    assert M.run(controls={"p": 0.5}, horizon=3)["final"]["n"] == 6


# ---------------------------------------------------------------- параметры
class P(mg.Model):
    a = mg.param(0.5, low=0, high=1, low_open=True)
    k = mg.param(2, low=1)
    m = mg.param(mg.array(2, 2, value=1.0), low=-10, high=10)


def test_param_defaults_types_and_override():
    r = P.run(horizon=0)
    assert r["final"] == {"a": 0.5, "k": 2, "m": [[1.0, 1.0], [1.0, 1.0]]}
    assert isinstance(r["final"]["k"], int)
    r = P.run(controls={"a": 1, "m": [[0, 1], [2, 3]]}, horizon=0)
    assert r["final"]["m"] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("controls,match", [
    ({"a": 0}, "вне допустимого"),
    ({"k": 1.5}, "целое"),
    ({"m": [[0, 1], [2, 30]]}, "вне допустимого"),
    ({"m": [1, 2]}, "форма"),
    ({"nope": 1}, "неизвестный"),
    ({"a": True}, "bool"),
    ({"a": "x"}, "тип"),
])
def test_param_errors(controls, match):
    with pytest.raises(MagnumControlError, match=match):
        P.run(controls=controls, horizon=0)


def test_numpy_param_value():
    np = pytest.importorskip("numpy")
    assert P.run(controls={"m": np.zeros((2, 2))}, horizon=0)["final"]["m"] == [[0, 0], [0, 0]]


@pytest.mark.parametrize("make,match", [
    (lambda: type("X", (mg.Model,), {"a": mg.param(5, low=0, high=1)}), "вне допустимого"),
    (lambda: type("X", (mg.Model,), {"time": mg.param(1)}), "зарезервированное"),
    (lambda: type("X", (mg.Model,), {"e": mg.every(1)(lambda self, x: None)}), "с параметрами"),
    (lambda: type("X", (mg.Model,), {"__init__": lambda self: None}), "__init__"),
])
def test_validation_errors(make, match):
    with pytest.raises(MagnumValidationError, match=match):
        make()


# ---------------------------------------------------------------- ошибки
def test_runtime_errors_are_wrapped_with_event_and_time():
    class M(mg.Model):
        @mg.every(mg.DAY)
        def boom(self):
            self.x = 1 / 0

    with pytest.raises(MagnumRuntimeError, match=r"boom.*t=1.*ZeroDivisionError"):
        M.run(horizon=2)


def test_emit_errors():
    class M(mg.Model):
        @mg.on_start
        def a(self):
            self.emit("nope")

    with pytest.raises(MagnumRuntimeError, match="нет события"):
        M.run(horizon=0)

    class N(mg.Model):
        @mg.on_start
        def a(self):
            self.emit(self.b)

        def b(self, x):
            pass

    with pytest.raises(MagnumRuntimeError, match="ожидает 1"):
        N.run(horizon=0)

    class K(mg.Model):
        @mg.on_start
        def a(self):
            self.emit(self.a, after=-1)

    with pytest.raises(MagnumRuntimeError, match="after"):
        K.run(horizon=0)


def test_max_events():
    class M(mg.Model):
        @mg.on_start
        def a(self):
            self.emit(self.b)

        def b(self):
            self.emit(self.a)

    with pytest.raises(MagnumRuntimeError, match="max_events"):
        M.run(horizon=1, max_events=1000)


def test_reserved_attributes_and_missing_state():
    class M(mg.Model):
        @mg.on_start
        def a(self):
            self.time = 5

    with pytest.raises(MagnumRuntimeError, match="служебный"):
        M.run(horizon=0)

    class N(mg.Model):
        @mg.on_start
        def a(self):
            self.x = self.y + 1

    with pytest.raises(MagnumRuntimeError, match="нет атрибута 'y'"):
        N.run(horizon=0)


def test_print_and_debug(capsys):
    class M(mg.Model):
        @mg.every(mg.DAY)
        def a(self):
            self.print("день", self.time)

    M.run(horizon=2)
    assert capsys.readouterr().out == "день 1.0\nдень 2.0\n"
    M.run(horizon=2, debug=False)
    assert capsys.readouterr().out == ""


def test_time_helpers():
    assert mg.hour_of_day(1.5) == 12 and mg.hour_of_day(0.9999999999) == 23
    assert mg.weekday(5.5) == 5 and mg.weekday(7) == 0 and mg.weekday(0, start=6) == 6
    assert mg.day_index(3.9) == 3
    assert mg.minutes(30) == pytest.approx(1 / 48) and mg.hours(6) == 0.25


def test_private_methods_are_not_events_and_say_so():
    from magnum.errors import MagnumRuntimeError, MagnumValidationError
    with pytest.raises(MagnumValidationError, match="не может начинаться с '_'"):
        class Bad(mg.Model):
            @mg.on_start
            def _init(self):
                self.x = 1

    class Emitter(mg.Model):
        @mg.on_start
        def init(self):
            self.emit(self._helper, after=1)

        def _helper(self):
            pass

    with pytest.raises(MagnumRuntimeError, match="служебный метод"):
        Emitter.run(horizon=2, debug=False)

    class Lam(mg.Model):
        @mg.on_start
        def init(self):
            self.emit(lambda: None, after=1)

    with pytest.raises(MagnumRuntimeError, match="lambda не подходит"):
        Lam.run(horizon=2, debug=False)
