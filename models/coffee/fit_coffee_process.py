"""Проверка: правдоподобие, которое движок выводит сам из разделения «эффект / критерий», восстанавливает
истинные параметры кофейни (coffee_process.py) по журналу событий.

1. Симуляция 30 дней с известными параметрами (кофейня открыта 8–22) -> журнал: приходы, концы заказов,
   уходы (с номерами клиентов). Открытия и закрытия в журнале не нужны — движок знает расписание сам.
   Начала заказов (решения when) в журнал для подбора не нужны — движок выводит их сам.
2. Проигрывание журнала один раз -> достаточные статистики (не зависят от параметров).
3. Максимум правдоподобия по 5 параметрам (Adam, затем L-BFGS, в логарифмах) + стандартные ошибки по гессиану.

    python -m models.coffee.fit_coffee_process        # из корня репозитория
"""
import math
import time

import torch

import magnum as mg
from models.coffee.coffee_process import CoffeeProcess as C

FIT = ["clients_per_hour", "order_mean", "order_std", "wait_mean", "wait_std"]
UNIT = {"clients_per_hour": (1.0, "в час"), "order_mean": (mg.SECOND, "с"), "order_std": (mg.SECOND, "с"),
        "wait_mean": (mg.SECOND, "с"), "wait_std": (mg.SECOND, "с")}


def fit(stats, start, steps=600, lr=0.05):
    raw = {k: torch.tensor(math.log(float(start[k])), dtype=torch.float64, requires_grad=True) for k in FIT}
    opt = torch.optim.Adam(raw.values(), lr=lr)
    for _ in range(steps):
        loss = -C.log_likelihood(stats, {k: torch.exp(v) for k, v in raw.items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Adam с постоянным шагом колеблется у узкого оптимума (среднее заказа известно до 0.1 %),
    # поэтому доводим квазиньютоном
    lbfgs = torch.optim.LBFGS(raw.values(), max_iter=200, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = -C.log_likelihood(stats, {k: torch.exp(v) for k, v in raw.items()})
        loss.backward()
        return loss

    lbfgs.step(closure)
    est = torch.stack([raw[k].detach() for k in FIT])
    # стандартные ошибки: обратный гессиан −log L по логарифмам параметров
    H = torch.autograd.functional.hessian(
        lambda z: -C.log_likelihood(stats, {k: torch.exp(z[i]) for i, k in enumerate(FIT)}), est)
    se_log = torch.sqrt(torch.diag(torch.linalg.inv(H)))
    return {k: float(torch.exp(est[i])) for i, k in enumerate(FIT)}, {k: float(se_log[i]) for i, k in enumerate(FIT)}


if __name__ == "__main__":
    truth = {k: float(C.params[k].default) for k in FIT}
    t0 = time.time()
    log, final = C.simulate(seed=7, horizon=30)
    t1 = time.time()
    hours = [(r[0] % 1) * 24 for r in log if r[1] == "arrive"]
    print(f"приходы с {min(hours):.2f} до {max(hours):.2f} ч (открыто {int(C.params['open_hour'].default)}–"
          f"{int(C.params['close_hour'].default)})")
    stats = C.replay([r for r in log if r[3] not in ("when", "daily")], horizon=30)
    t2 = time.time()
    start = {k: v * 2.0 for k, v in truth.items()}                 # старт вдвое от истины
    est, se = fit(stats, start)
    t3 = time.time()
    print(f"журнал 30 дней: {sum(r[3] != 'when' for r in log)} событий (обслужено {final['served']}, ушли {final['lost']}); "
          f"симуляция {t1 - t0:.1f} с, проигрывание {t2 - t1:.2f} с, подбор {t3 - t2:.1f} с")
    print(f"\n{'параметр':20}{'истина':>10}{'оценка':>10}{'±2σ':>18}")
    for k in FIT:
        u, unit = UNIT[k]
        lo, hi = est[k] * math.exp(-2 * se[k]), est[k] * math.exp(2 * se[k])
        print(f"{k:20}{truth[k] / u:>10.1f}{est[k] / u:>10.1f}{f'{lo / u:.1f}–{hi / u:.1f}':>18}  {unit}")
