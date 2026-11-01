"""Подбор нейросети спроса лимонного киоска по данным и выбор решений по ней.

1. Данные: дни «настоящей» игры (lemonade/engine.py — правила Apple II 1979, покупатели — пуассоновский
   поток) при случайных контролах: погода, шанс дождя, листовки, цена, сколько стаканов. Видно,
   сколько продано и когда лимонад кончился (после этого спрос не виден).
2. Подбор весов сети по точному правдоподобию: n·log λ − λ·F(τ), где F(τ) — доля дневного потока до
   момента распродажи (F = 1, если лимонад не кончился). Adam, проверка на отложенных днях.
3. Решения: при заданной погоде и себестоимости — цена и листовки (градиентом по точному ожиданию
   прибыли через сеть) и число стаканов (перебором). Сравнение с решением, знающим правила.

    python -m models.lemonade.fit_lemonade       # из корня репозитория (папка test)
"""
import importlib.util
import json
import math
import random
from pathlib import Path

import torch

from models.lemonade.lemonade_model import WEIGHTS, LemonadeStand, day_cdf, expected_profit, features, log_rate

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lemonade_engine", HERE.parents[1] / "lemonade" / "engine.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
W_CODE = {"sunny": 0, "cloudy": 1, "hot": 2}
torch.set_num_threads(2)


def make_days(n, seed):
    """Дни игры при случайных контролах (без гроз и дорожных работ — это отдельные события)."""
    rng, out = random.Random(seed), []
    for i in range(n):
        info = G.World(seed * 10000 + i).day(1)                      # прохожие дня
        u = rng.random()
        w = "sunny" if u < 0.6 else "cloudy" if u < 0.8 else "hot"
        info.update(weather=w, rain=rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]) if w == "cloudy" else 0.0,
                    storm=False, street=None, traffic=1.0)
        signs, price, glasses = rng.randint(0, 8), rng.randint(2, 60), rng.randint(5, 150)
        _, s = G.World(seed).play(info, price, signs, glasses)
        out.append({"weather": W_CODE[w], "rain": info["rain"], "ad_spend": 15.0 * signs, "price": float(price),
                    "glasses": glasses, "sold": s["sold"],
                    "frac": day_cdf(s["out_at"]) if s["out_at"] is not None else 1.0})
    return out


def tensors(days):
    T = lambda k, dt=torch.float64: torch.tensor([d[k] for d in days], dtype=dt)  # noqa: E731
    return {"x": features(T("weather", torch.long), T("rain"), T("ad_spend"), T("price")),
            "n": T("sold"), "frac": T("frac")}


def log_lik(p, data):
    """Средний по дням log P(продажи) — пуассоновский поток, оборванный в момент распродажи."""
    lr = log_rate(p, data["x"])
    return (data["n"] * lr - torch.exp(lr) * data["frac"] - torch.lgamma(data["n"] + 1)).mean()


def fit(train, steps=3000, lr=0.02, l2=1e-3):
    p = LemonadeStand.make_params(names=WEIGHTS)
    opt = torch.optim.Adam([p[k] for k in WEIGHTS], lr=lr)
    for i in range(steps):
        loss = -log_lik(p, train) + l2 * (p["w1"] ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: p[k].detach() for k in WEIGHTS}


def true_lambda(weather, rain, ad_spend, price):
    info = {"weather": ["sunny", "cloudy", "hot"][weather], "rain": rain, "traffic": 1.0}
    return G.mean_demand(price, ad_spend / 15.0, info, G.TRUTH)


def best_decision(lam_fn, weather, rain, cost, steps=150):
    """Цена и листовки — градиентом по точному ожиданию прибыли; стаканы — перебором на каждом шаге."""
    best = None
    for p0, a0 in [(8.0, 15.0), (14.0, 45.0), (25.0, 30.0)]:          # несколько стартов
        z = torch.tensor([math.log(p0), math.log1p(a0)], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(steps):
            price, ad = torch.exp(z[0]), torch.expm1(z[1]).clamp_min(0)
            lam = lam_fn(weather, rain, ad, price)
            q = best_q(lam.detach(), price.detach(), cost)
            ep = price * _es(lam, q) - cost * q - ad
            opt.zero_grad()
            (-ep).backward()
            opt.step()
        price, ad = math.exp(z[0].item()), math.expm1(z[1].item())
        lam = lam_fn(weather, rain, torch.tensor(ad), torch.tensor(price)).item()
        q = best_q(torch.tensor(lam), torch.tensor(price), cost)
        if best is None or ep.item() > best["profit"]:
            best = {"price": price, "ad_spend": ad, "glasses": q, "profit": ep.item(), "lam": lam}
    return best


def _es(lam, q):
    from models.lemonade.lemonade_model import expected_sold
    return expected_sold(lam, q)


def best_q(lam, price, cost):
    """Число стаканов с наибольшей ожидаемой прибылью: добавляем стакан, пока P(D > Q)·цена > себестоимость."""
    q, surv, pmf, k = 0, 1.0, math.exp(-float(lam)), 0
    while float(price) * surv > cost and q < 1000:
        q += 1
        surv -= pmf
        k += 1
        pmf *= float(lam) / k
    return q


def truth_profit(weather, rain, ad_spend, price, glasses, cost):
    lam = torch.tensor(true_lambda(weather, rain, ad_spend, price), dtype=torch.float64)
    return float(price * _es(lam, glasses) - cost * glasses - ad_spend)


if __name__ == "__main__":
    train, test = tensors(make_days(300, seed=1)), tensors(make_days(100, seed=2))
    p = fit(train)
    # «идеальная» сеть = правда: log λ по формуле игры — для сравнения правдоподобий
    with torch.no_grad():
        ll_fit = log_lik(p, test).item()
        lt = torch.log(torch.tensor([true_lambda(int(w), float(r), float(a), float(pr))
                                     for w, r, a, pr in zip(test["x"][:, :3].argmax(1), test["x"][:, 3],
                                                            torch.expm1(test["x"][:, 4] * 3), test["x"][:, 5] * 20)],
                                    dtype=torch.float64))
        ll_true = (test["n"] * lt - torch.exp(lt) * test["frac"] - torch.lgamma(test["n"] + 1)).mean().item()
    print(f"log P(продажи) на отложенных 100 днях, в среднем за день: сеть {ll_fit:.3f}, правила игры {ll_true:.3f}")

    print(f"\n{'погода':10}{'листовки':>9}" + "".join(f"{f'{c}¢':>12}" for c in (5, 10, 15, 20, 30)))
    for w, rain in ((0, 0.0), (1, 0.5), (2, 0.0)):
        for ad in (0, 45):
            row = ""
            for price in (5, 10, 15, 20, 30):
                lam = torch.exp(log_rate(p, features(w, rain, ad, price))).item()
                row += f"{lam:6.1f} / {true_lambda(w, rain, ad, price):4.1f}"
            print(f"{['солнечно', 'пасм. 50%', 'жара'][w]:10}{ad:>7}¢ " + row)
    print("(в ячейке: λ сети / λ по правилам игры — среднее число покупателей за день)")

    print("\nРешения при себестоимости 4¢ — по сети и по правилам; прибыль обоих решений по правилам игры:")
    net = lambda w, r, a, pr: torch.exp(log_rate(p, features(w, r, a, pr)))  # noqa: E731
    truth = lambda w, r, a, pr: torch.as_tensor(true_lambda(w, float(r), a, pr), dtype=torch.float64) \
        if not isinstance(pr, torch.Tensor) or not pr.requires_grad else _true_torch(w, r, a, pr)  # noqa: E731

    def _true_torch(w, r, a, pr):
        th = G.TRUTH
        n1 = torch.where(pr < th["ref"], th["base"] + 0.8 * th["base"] * (th["ref"] - pr) / th["ref"],
                         th["base"] * (th["ref"] / pr) ** th["power"])
        sf = 1 + (1 - torch.exp(-th["sign"] * a / 15))
        wm = th["hot"] if w == 2 else (1 - r) if w == 1 else 1.0
        return n1 * sf * wm

    out = {}
    for w, rain in ((0, 0.0), (1, 0.5), (2, 0.0)):
        dn = best_decision(net, w, rain, 4.0)
        dt = best_decision(lambda *a: _true_torch(*a), w, rain, 4.0)
        pn = truth_profit(w, rain, dn["ad_spend"], dn["price"], dn["glasses"], 4.0)
        pt = truth_profit(w, rain, dt["ad_spend"], dt["price"], dt["glasses"], 4.0)
        name = ["солнечно", "пасмурно 50%", "жара"][w]
        print(f"  {name:13} сеть: цена {dn['price']:5.1f}¢, листовки {dn['ad_spend']:5.1f}¢, стаканов {dn['glasses']:3d} "
              f"→ {pn:6.1f}¢ | правила: {dt['price']:5.1f}¢, {dt['ad_spend']:5.1f}¢, {dt['glasses']:3d} → {pt:6.1f}¢")
        out[name] = {"net": dn, "truth": dt, "net_profit_by_rules": pn, "truth_profit": pt}

    json.dump({k: v.tolist() for k, v in p.items()}, open(HERE / "demand_net.json", "w"))
    print("\nвеса сети сохранены в models/lemonade/demand_net.json")

    # проверка: модель Magnum с подобранной сетью — среднее по 2000 прогонам против точного ожидания
    d = out["солнечно"]["net"]
    ctl = dict({k: v.tolist() for k, v in p.items()}, weather=0, rain=0.0, cost=4.0, price=d["price"],
               ad_spend=d["ad_spend"], glasses=d["glasses"])
    runs = [LemonadeStand.run(controls=ctl, seed=s, horizon=1, debug=False, record=[])["final"]["profit"] for s in range(2000)]
    m = sum(float(x) for x in runs) / len(runs)
    exact, _ = expected_profit(p, 0, 0.0, d["ad_spend"], 4.0, d["price"], d["glasses"])
    print(f"симуляция LemonadeStand (2000 дней): средняя прибыль {m:.1f}¢; точное ожидание по сети {exact.item():.1f}¢")
