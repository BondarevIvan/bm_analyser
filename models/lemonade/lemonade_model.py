"""Лимонный киоск как модель Magnum: контролы — погода, расходы на листовки, себестоимость стакана,
цена стакана (и сколько стаканов сделать); спрос — маленькая нейросеть, которая по контролам
предсказывает параметр λ пуассоновского распределения числа покупателей за день.

    λ = exp(net(x)),   x = [погода one-hot (солнечно, пасмурно, жара), шанс дождя,
                            log(1 + листовки¢)/3, цена¢/20, log(цена¢)/2]
    net = tanh-слой из HIDDEN нейронов -> одно число

Себестоимость на спрос не влияет (покупатель её не видит) — она входит только в прибыль.
День: покупатели приходят пуассоновским потоком 9:00–17:00 (днём чаще), каждый берёт стакан, пока есть.

    from models.lemonade.lemonade_model import LemonadeStand
    r = LemonadeStand.run(controls={"price": 12, "ad_spend": 45, "glasses": 40}, seed=1, horizon=1, debug=False)
"""
import math

import torch

import magnum as mg

OPEN, CLOSE = 9, 17
HIDDEN = 8
N_IN = 7
WEATHER = ["солнечно", "пасмурно", "жара"]
WEIGHTS = ["w1", "b1", "w2", "b2"]


def _init(*shape, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float64) * scale).tolist()


def features(weather, rain, ad_spend, price):
    """Вход сети; аргументы — числа или тензоры одной формы (погода — целое 0/1/2)."""
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
    w = torch.nn.functional.one_hot(torch.as_tensor(weather).long(), 3).to(torch.float64)
    price = T(price).clamp_min(0.5)
    rest = torch.stack([T(rain), torch.log1p(T(ad_spend)) / 3, price / 20, torch.log(price) / 2], -1)
    return torch.cat([w, rest], -1)


def log_rate(p, x):
    """log λ — логарифм среднего числа покупателей за день."""
    return (torch.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]).squeeze(-1)


def day_cdf(t):
    """Доля дневного спроса, пришедшая к моменту t (часы): днём больше, утром и вечером меньше."""
    x = min(max((t - OPEN) / (CLOSE - OPEN), 0.0), 1.0)
    return 3 * x * x - 2 * x ** 3


def sample_time(u):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 3 * mid * mid - 2 * mid ** 3 < u else (lo, mid)
    return OPEN + lo * (CLOSE - OPEN)


class LemonadeStand(mg.Model):
    name = "Лимонный киоск"

    # ---- контролы
    weather = mg.param(0, low=0, high=2)              # 0 — солнечно, 1 — пасмурно, 2 — жара
    rain = mg.param(0.0, low=0, high=1)               # шанс дождя (для пасмурного дня)
    ad_spend = mg.param(30.0, low=0)                  # ¢ на листовки
    cost = mg.param(4.0, low=0)                       # ¢ себестоимость стакана
    price = mg.param(10.0, low=0, low_open=True)      # ¢ цена стакана
    glasses = mg.param(40, low=0)                     # сколько стаканов сделать
    # ---- модель спроса (веса подбираются по данным: fit_lemonade.py)
    w1 = mg.param(_init(N_IN, HIDDEN, scale=0.3, seed=1))
    b1 = mg.param(mg.array(HIDDEN, value=0.0))
    w2 = mg.param(_init(HIDDEN, 1, scale=0.3, seed=2))
    b2 = mg.param([math.log(20.0)])

    def demand(self):
        """λ — ожидаемое число покупателей за день при текущих контролах (тензор)."""
        x = features(self.weather, self.rain, self.ad_spend, self.price)
        return torch.exp(log_rate({k: getattr(self, k) for k in WEIGHTS}, x))

    @mg.on_start
    def init(self):
        self.lam = self.demand()
        self.left = int(self.glasses)
        self.sold = 0
        self.missed = 0
        self.sold_out_at = None
        self.revenue = 0.0
        self.expenses = self.cost * int(self.glasses) + self.ad_spend
        n = self.rng.poisson(self.lam)                # число покупателей за день ~ Пуассон(λ)
        for _ in range(int(n)):
            t = sample_time(self.rng.random())
            self.emit(self.customer, after=t * mg.HOUR)

    def customer(self):
        if self.left > 0:
            self.left -= 1
            self.sold += 1
            self.revenue = self.revenue + self.price
            if self.left == 0:
                self.sold_out_at = self.time * 24
        else:
            self.missed += 1

    @mg.every(mg.DAY)
    def close(self):
        self.profit = self.revenue - self.expenses


# ------------------------------------------------------------ точное ожидание прибыли (без симуляции)
def expected_sold(lam, q):
    """E[min(Q, D)] при D ~ Пуассон(λ): Σ_{k<Q} P(D > k). Дифференцируемо по λ."""
    if q <= 0:
        return lam * 0
    k = torch.arange(q, dtype=torch.float64)
    log_pmf = k * torch.log(lam) - lam - torch.lgamma(k + 1)
    cdf = torch.cumsum(torch.exp(log_pmf), 0)
    return (1 - cdf).clamp_min(0).sum()


def expected_profit(p, weather, rain, ad_spend, cost, price, glasses):
    lam = torch.exp(log_rate(p, features(weather, rain, ad_spend, price)))
    return price * expected_sold(lam, glasses) - cost * glasses - ad_spend, lam


def trained(path=None):
    """Веса сети спроса, подобранные fit_lemonade.py (demand_net.json), — как controls для run()."""
    import json
    from pathlib import Path
    path = Path(path or Path(__file__).with_name("demand_net.json"))
    return json.load(open(path)) if path.exists() else {}
