"""Прототип magnum.process: эффекты отдельно от критериев -> автоматическое правдоподобие журнала."""
import math

import pytest
import torch

import magnum as mg
from magnum.process import Exponential, LogNormal, Process, daily, rate, timer, when


class Shop(Process):
    """Одна касса; приходящий видит очередь и с ней приходит реже (balk); обслуживание экспоненциальное."""
    per_hour = mg.param(12.0, low=0)
    balk = mg.param(0.5, low=0)
    service = mg.param(mg.minutes(4), low=0, low_open=True)
    tills = mg.param(1, low=1)

    serve = timer(lambda m: Exponential(m.service), fires="done")

    def init(self):
        self.queue = mg.Queue()
        self.free = int(self.tills)

    @rate(lambda m: m.per_hour / mg.HOUR * torch.exp(-m.balk * len(m.queue)))
    def arrive(self):
        self.queue.push(self.new_id())

    @when(lambda m: m.free > 0 and len(m.queue) > 0)
    def take(self):
        self.start("serve", self.queue.pop())
        self.free -= 1

    def done(self, cid):
        self.free += 1


def _fit(cls, stats, names, start, steps=400, lr=0.05):
    raw = {k: torch.tensor(math.log(start[k]), dtype=torch.float64, requires_grad=True) for k in names}
    opt = torch.optim.Adam(raw.values(), lr=lr)
    for _ in range(steps):
        loss = -cls.log_likelihood(stats, {k: torch.exp(v) for k, v in raw.items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: math.exp(v.item()) for k, v in raw.items()}


def test_state_dependent_rate_recovered():
    log, _ = Shop.simulate(seed=3, horizon=20)
    stats = Shop.replay(log, 20)
    assert len(stats["exposure"]) > 3                      # интенсивность прихода зависит от длины очереди
    names = ["per_hour", "balk", "service"]
    est = _fit(Shop, stats, names, {"per_hour": 6.0, "balk": 1.5, "service": mg.minutes(8)})
    assert abs(est["per_hour"] - 12.0) < 1.0
    assert abs(est["balk"] - 0.5) < 0.15
    assert abs(est["service"] / mg.MINUTE - 4.0) < 0.3


def test_when_records_are_not_needed():
    log, _ = Shop.simulate(seed=4, horizon=3)
    a = Shop.log_likelihood(Shop.replay(log, 3), {}).item()
    b = Shop.log_likelihood(Shop.replay([r for r in log if r[3] != "when"], 3), {}).item()
    assert a == b


def test_constant_rate_matches_poisson_formula():
    log, _ = Shop.simulate(seed=5, horizon=2)
    stats = Shop.replay(log, 2)
    ll = Shop.log_likelihood(stats, {"balk": 0.0}).item()
    n = sum(r[1] == "arrive" for r in log)
    lam = 12.0 / mg.HOUR
    serve = Exponential(mg.minutes(4))
    expect = n * math.log(lam) - lam * 2 + float(serve.log_prob(stats["done"]["serve"]).sum()) \
        + float(serve.log_survival(stats["cens"]["serve"]).sum())
    assert math.isclose(ll, expect, rel_tol=1e-9)


def test_inconsistent_log_rejected():
    with pytest.raises(ValueError, match="не запущен"):
        Shop.replay([(0.1, "done", 99, "timer")], 1)


def test_lognormal_mean_std():
    d = LogNormal(10.0, 3.0)
    x = torch.linspace(0.01, 80, 200001, dtype=torch.float64)
    p = torch.exp(d.log_prob(x))
    dx = x[1] - x[0]
    m = float((x * p).sum() * dx)
    sd = math.sqrt(float(((x - m) ** 2 * p).sum() * dx))
    assert abs(m - 10) < 0.01 and abs(sd - 3) < 0.01


# ------------------------------------------------------------ расписание
class Hours(Process):
    """Приходы только с open_hour до close_hour; ночью интенсивность 0."""
    open_hour = mg.param(8, low=0, high=23)
    close_hour = mg.param(20, low=0, high=23)
    per_hour = mg.param(10.0, low=0)

    def init(self):
        self.is_open = False

    @daily(lambda m: m.open_hour * mg.HOUR)
    def open(self):
        self.is_open = True

    @daily(lambda m: m.close_hour * mg.HOUR)
    def close(self):
        self.is_open = False

    @rate(lambda m: m.per_hour / mg.HOUR if m.is_open else 0.0)
    def arrive(self):
        pass


def test_daily_schedule_limits_arrivals_and_likelihood():
    log, _ = Hours.simulate(seed=1, horizon=5)
    hours = [(t % 1) * 24 for t, e, _, k in log if e == "arrive"]
    assert hours and min(hours) >= 8 and max(hours) <= 20
    arrivals = [r for r in log if r[3] == "rate"]                 # расписание движок восстанавливает сам
    ll = Hours.log_likelihood(Hours.replay(arrivals, 5), {}).item()
    lam = 10.0 / mg.HOUR
    assert math.isclose(ll, len(arrivals) * math.log(lam) - lam * 5 * 12 * mg.HOUR, rel_tol=1e-9)
    night = Hours.replay(arrivals + [(5 + 2 * mg.HOUR - 5, "arrive", None, "rate")], 5)   # приход в 2 часа ночи
    assert Hours.log_likelihood(night, {}).item() == -math.inf
