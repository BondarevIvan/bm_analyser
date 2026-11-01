"""magnum.hidden: фильтр частиц по наблюдаемым моментам одного таймера (скрытые приходы)."""
import math
import random

import numpy as np

import magnum as mg
from magnum.hidden import smc_day
from magnum.process import Exponential, Process, rate, timer, when


class MM1(Process):
    lam = mg.param(10.0, low=0)                       # приходов в час
    srv = mg.param(0.08, low=0, low_open=True)        # среднее обслуживание, часы
    order = timer(lambda m: Exponential(m.srv * mg.HOUR), fires="finish")

    def init(self):
        self.queue = mg.Queue()
        self.free = 1

    @rate(lambda m: m.lam / mg.HOUR)
    def arrive(self):
        self.queue.push(self.new_id())

    @when(lambda m: m.free > 0 and len(m.queue) > 0)
    def start_order(self):
        self.start("order", self.queue.pop())
        self.free -= 1

    def finish(self, cid):
        self.free += 1


def exact_mm1(times, end, lam, mu, N=60):
    """Точное log P(моментов уходов) для M/M/1 (время в часах): цепь по числу в системе после ухода."""
    lg = np.array([math.lgamma(i + 1) for i in range(N)])

    def pois(x):
        k = np.arange(N)
        return np.exp(-x + k * math.log(x) - lg) if x > 0 else np.eye(N)[0]

    a = np.zeros(N)
    a[0] = 1.0
    prev, ll = 0.0, 0.0
    for t in times:
        g, new = t - prev, np.zeros(N)
        pk = pois(lam * g)
        for n in range(1, N):
            new[n - 1:] += a[n] * mu * math.exp(-mu * g) * pk[:N - n + 1]
        ss = np.linspace(0, g, 801)
        w = lam * np.exp(-lam * ss) * mu * np.exp(-mu * (g - ss)) * (ss[1] - ss[0])
        w[0] /= 2
        w[-1] /= 2
        for s, wi in zip(ss, w):
            new += a[0] * wi * pois(lam * (g - s))
        ll += math.log(new.sum())
        a, prev = new / new.sum(), t
    g = end - prev
    surv = a[0] * (math.exp(-lam * g) + lam * (math.exp(-lam * g) - math.exp(-mu * g)) / (mu - lam)) \
        + a[1:].sum() * math.exp(-mu * g)
    return ll + math.log(surv)


def test_smc_matches_exact_mm1():
    end = 0.25
    log, _ = MM1.simulate(seed=2, horizon=end)
    d = [r[0] for r in log if r[1] == "finish"]
    for lam, srv in [(10, 0.08), (6, 0.08)]:
        ex = exact_mm1([t * 24 for t in d], end * 24, lam, 1 / srv) + len(d) * math.log(24)   # плотность в днях
        est = np.mean([smc_day(MM1, d, "finish", end, {"lam": lam, "srv": srv}, n=128, rng=random.Random(s))[0]
                       for s in range(3)])
        assert abs(est - ex) < 1.5


def test_sampled_log_is_consistent_with_observations():
    end = 0.25
    log, _ = MM1.simulate(seed=5, horizon=end)
    d = [r[0] for r in log if r[1] == "finish"]
    _, logs = smc_day(MM1, d, "finish", end, n=32, rng=random.Random(0), keep=2)
    for lg in logs:
        assert [r[0] for r in lg if r[1] == "finish"] == d
        MM1.replay(lg, end)                        # полный журнал согласован с моделью
