"""Параметры — тензоры torch; градиенты ожиданий через log_prob и expected()."""
import math

import numpy as np
import pytest
import torch

import magnum as mg
from magnum.errors import MagnumControlError

T = 2.0  # горизонт, дни


# ------------------------------------------------------------------ параметры
class P(mg.Model):
    rate = mg.param(3.0, low=0)
    n = mg.param(4, low=0)
    w = mg.param([[1.0, 2.0], [3.0, 4.0]], low=0)

    @mg.on_start
    def init(self):
        self.seen = [type(self.rate).__name__, str(self.rate.dtype), str(self.n.dtype), list(self.w.shape)]
        self.idle = self.n          # 0-мерный тензор без градиента -> обычное число
        self.idle -= 1


def test_params_are_tensors():
    assert isinstance(P.params["rate"].default, torch.Tensor)
    assert P.params["n"].integer and P.params["n"].default.dtype == torch.int64
    assert P.params["w"].shape == (2, 2)
    r = P.run(horizon=0, debug=False)
    assert r["final"]["seen"] == ["Tensor", "torch.float64", "torch.int64", [2, 2]]
    assert r["final"]["idle"] == 3 and r["final"]["n"] == 4      # параметр не испорчен x -= 1
    assert isinstance(r["controls"]["w"], torch.Tensor)
    assert r["final"]["w"] == [[1.0, 2.0], [3.0, 4.0]]           # в final — обычные числа/списки


def test_controls_accept_numbers_lists_numpy_tensors():
    for w in ([[0, 1], [2, 3]], np.array([[0.0, 1], [2, 3]]), torch.tensor([[0.0, 1], [2, 3]])):
        r = P.run(controls={"w": w, "n": torch.tensor(2)}, horizon=0, debug=False)
        assert r["final"]["w"] == [[0.0, 1.0], [2.0, 3.0]] and r["final"]["n"] == 2


@pytest.mark.parametrize("controls,match", [
    ({"w": [1.0, 2.0]}, "форма"),
    ({"w": [[1.0, -2.0], [3.0, 4.0]]}, "вне допустимого"),
    ({"n": 2.5}, "целое"),
    ({"n": torch.tensor(2.0, requires_grad=True)}, "обучаемым"),
    ({"rate": float("nan")}, "вне допустимого"),
    ({"w": [[1.0], [2.0, 3.0]]}, "форма значения"),
])
def test_param_errors(controls, match):
    with pytest.raises(MagnumControlError, match=match):
        P.run(controls=controls, horizon=0)


def test_make_params_and_clamp():
    ps = P.make_params()
    assert ps["rate"].requires_grad and ps["w"].requires_grad and not ps["n"].requires_grad
    assert mg.trainable(ps) == [ps["rate"], ps["w"]]
    only = P.make_params(names=["rate"])
    assert only["rate"].requires_grad and not only["w"].requires_grad
    with torch.no_grad():
        ps["w"] -= 10
    P.clamp_(ps)
    assert (ps["w"] >= 0).all()

    class Free(mg.Model):
        x = mg.param(1.0)            # без границ

    Free.clamp_(Free.make_params())  # не падает


def test_grad_param_passed_without_copy():
    rate = torch.tensor(2.0, requires_grad=True)

    class M(mg.Model):
        rate = mg.param(1.0, low=0)

        @mg.on_start
        def init(self):
            self.same = self.rate is RATE[0]
            self.double = self.rate * 2

    RATE = [rate]
    f = M.run(controls={"rate": rate}, horizon=0, debug=False)["final"]
    assert f["same"] is True
    assert isinstance(f["double"], torch.Tensor) and f["double"].requires_grad    # final сохраняет граф
    f["double"].backward()
    assert rate.grad.item() == 2.0


# ------------------------------------------------------------------ градиенты
def grad_of_expected(model, name, value, controls=None, n=600, horizon=T):
    ps = model.make_params(controls, names=[name])
    runs = [model.run(controls=ps, seed=s, horizon=horizon, debug=False, record=[]) for s in range(n)]
    est = mg.expected(runs, value)
    est.backward()
    return est.item(), ps[name].grad.item()


class Shop(mg.Model):
    rate = mg.param(5.0, low=0)          # в час
    price = mg.param(100.0, low=0)

    @mg.on_start
    def init(self):
        self.n = 0
        self.revenue = 0.0

    @mg.arrival(rate="rate", per=mg.HOUR)
    def visit(self):
        self.n += 1
        self.revenue = self.revenue + self.price


def test_arrival_rate_gradient_with_censoring():
    # E[n] = rate·24·T  =>  dE/drate = 24·T = 48
    value, g = grad_of_expected(Shop, "rate", "n")
    assert abs(value - 5 * 24 * T) < 3
    assert abs(g - 24 * T) < 0.15 * 24 * T


def test_pathwise_gradient():
    # E[revenue] = rate·24·T·price  =>  dE/dprice = rate·24·T = 240 (точно по траектории)
    value, g = grad_of_expected(Shop, "price", "revenue", n=200)
    assert abs(g - 240) < 0.1 * 240


class Draws(mg.Model):
    mean = mg.param(2.0, low=0, low_open=True)
    p = mg.param(0.3, low=0, high=1)
    w = mg.param([1.0, 3.0], low=0)
    lam = mg.param(4.0, low=0)
    mu = mg.param(1.0)

    @mg.on_start
    def init(self):
        self.x_exp = self.rng.exponential(self.mean)
        self.x_bern = int(self.rng.bernoulli(self.p))
        self.x_cat0 = int(self.rng.categorical(self.w) == 0)
        self.x_pois = self.rng.poisson(self.lam)
        self.x_norm = self.rng.normal(self.mu, 1.0)
        self.x_choose = int(self.choose([(self.p, None)]) == 0)


@pytest.mark.parametrize("name,var,true_grad,tol", [
    ("mean", "x_exp", 1.0, 0.2),          # E = mean
    ("p", "x_bern", 1.0, 0.2),            # E = p
    ("p", "x_choose", 1.0, 0.2),          # E = p
    ("w", None, None, None),              # отдельно ниже (вектор)
    ("lam", "x_pois", 1.0, 0.2),          # E = lam
    ("mu", "x_norm", 1.0, 0.2),           # E = mu
])
def test_score_function_gradients(name, var, true_grad, tol):
    if var is None:
        ps = Draws.make_params(names=["w"])
        runs = [Draws.run(controls=ps, seed=s, horizon=0, debug=False, record=[]) for s in range(4000)]
        mg.expected(runs, "x_cat0").backward()
        # P(0) = w0/(w0+w1): d/dw0 = w1/s² = 3/16, d/dw1 = −w0/s² = −1/16
        g = ps["w"].grad.tolist()
        assert abs(g[0] - 3 / 16) < 0.05 and abs(g[1] + 1 / 16) < 0.05
        return
    _, g = grad_of_expected(Draws, name, var, n=4000, horizon=0)
    assert abs(g - true_grad) < tol


def test_log_prob_is_zero_without_grad():
    r = Shop.run(seed=1, horizon=1, debug=False)
    assert r["log_prob"].item() == 0.0 and not r["log_prob"].requires_grad
    assert isinstance(r["final"]["revenue"], float)


def test_log_prob_value_matches_poisson_process():
    # для однородного потока log p = n·log λ − λ·T (λ в 1/день)
    ps = Shop.make_params(names=["rate"])
    r = Shop.run(controls=ps, seed=3, horizon=T, debug=False)
    lam = 5.0 * 24
    assert math.isclose(r["log_prob"].item(), r["final"]["n"] * math.log(lam) - lam * T, rel_tol=1e-9)


def test_queue_accepts_tensor_items():
    q = mg.Queue()
    q.push(torch.tensor(3))
    assert 3 in q and q.remove(torch.tensor(3)) and len(q) == 0
    assert mg.Resource(torch.tensor(2)).capacity == 2


def test_aggregated_draws_match_direct_log_prob():
    """Исходы по элементам таблицы копятся счётчиками; изменение таблицы на месте начинает новую группу."""
    log = []

    class Tab(mg.Model):
        q = mg.param([0.2, 0.7], low=0, high=1)

        @mg.on_start
        def init(self):
            self._p = self.q * 1.0              # таблица, посчитанная один раз (не лист графа)
            for i in range(40):
                if i == 20:
                    self._p.mul_(0.5)           # меняем на месте — дальше вероятности другие
                j = i % 2
                x = self.rng.bernoulli(self._p[j])
                log.append(math.log(self._p[j].item()) if x else math.log1p(-self._p[j].item()))
                k = self.rng.categorical(self.q)
                w = self.q.tolist()
                log.append(math.log(w[k] / sum(w)))

    ps = Tab.make_params()
    r = Tab.run(controls=ps, seed=4, horizon=0, debug=False)
    assert math.isclose(r["log_prob"].item(), sum(log), rel_tol=1e-12)
    r["log_prob"].backward()
    assert ps["q"].grad is not None and torch.isfinite(ps["q"].grad).all()
