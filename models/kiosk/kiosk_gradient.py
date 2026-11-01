"""Разбор: градиент ожидаемой выручки через вероятность пути (score function) на маленькой модели,
где ответ известен точно.

Киоск, один бариста, рабочий день 8 часов. Клиенты приходят с интенсивностью lam в час.
Если бариста свободен — заказ (время экспоненциальное, среднее s часов), иначе клиент уходит.
Выручка = price · обслуженные.

    python -m models.kiosk.kiosk_gradient
"""
import math

import torch

import magnum as mg
from magnum.process import Exponential, Process, rate, timer

T = 8 * mg.HOUR                      # горизонт — один рабочий день (время в днях)
PRICE = 300.0


class Kiosk(Process):
    lam = mg.param(10.0, low=0)                        # клиентов в час
    s = mg.param(0.1, low=0, low_open=True)            # среднее время заказа, часы

    order = timer(lambda m: Exponential(m.s * mg.HOUR), fires="done")

    def init(self):
        self.free = True
        self.served = 0
        self.lost = 0

    @rate(lambda m: m.lam / mg.HOUR)
    def arrive(self):
        cid = self.new_id()
        if self.free:                                  # ветка: решение по состоянию
            self.free = False
            self.served += 1
            self.start("order", cid)
        else:
            self.lost += 1

    def done(self, cid):
        self.free = True


def exact_expected_served(lam, s):
    """Точно: двухсостоянийная цепь Маркова, E[served] = ∫ lam·P(свободен в t) dt (t в часах)."""
    mu, h = 1 / s, 8.0
    a = lam + mu
    return lam * (mu * h / a + lam / a ** 2 * (1 - torch.exp(-a * h)))


def path_log_prob(stats, lam, s):
    return Kiosk.log_likelihood(stats, {"lam": lam, "s": s})


if __name__ == "__main__":
    lam0, s0 = 10.0, 0.1

    # ---------------------------------------------------- 1. один путь, по шагам
    log, final = Kiosk.simulate(seed=3, horizon=T)
    print("1. Один день (seed=3), первые события:")
    for t, ev, ent, kind in log[:8]:
        print(f"   {t / mg.HOUR * 60:7.2f} мин  {ev:7} клиент {ent}  ({kind})")
    print(f"   ... всего за день: пришло {final['served'] + final['lost']}, обслужено {final['served']}, "
          f"ушли {final['lost']}")

    stats = Kiosk.replay(log, T)
    lam = torch.tensor(lam0, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
    lp = path_log_prob(stats, lam, s)
    lp.backward()
    n_arr = sum(r[1] == "arrive" for r in log)
    d = stats["done"]["order"] / mg.HOUR
    c = stats["cens"]["order"] / mg.HOUR
    print(f"\n   log p(путь) = {lp.item():.2f}")
    print(f"   ∂/∂lam = {lam.grad.item():+.3f}   по формуле n/lam − T = {n_arr}/{lam0} − 8 = {n_arr / lam0 - 8:+.3f}")
    print(f"   ∂/∂s   = {s.grad.item():+.3f}   по формуле Σ(d − s)/s² по {len(d)} заказам "
          f"(+ незаконченный) = {float(((d - s0).sum() + c.sum()) / s0 ** 2):+.3f}")
    print(f"   выручка этого дня = {PRICE * final['served']:.0f} — от lam и s на фиксированном пути не зависит")

    # ---------------------------------------------------- 2. градиент ожидания по многим путям
    N = 4000
    runs = [Kiosk.simulate(seed=sd, horizon=T) for sd in range(N)]
    stats = [Kiosk.replay(lg, T) for lg, _ in runs]
    R = torch.tensor([PRICE * f["served"] for _, f in runs], dtype=torch.float64)

    lam = torch.tensor(lam0, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
    lp = torch.stack([path_log_prob(st, lam, s) for st in stats])
    # E[R · ∇log p]; базовая линия (средняя выручка) не меняет ожидание, но сильно снижает шум
    b = R.mean()
    est = ((R - b) * lp).mean()
    est.backward()
    g_mc = torch.tensor([lam.grad.item(), s.grad.item()])

    lam_e = torch.tensor(lam0, dtype=torch.float64, requires_grad=True)
    s_e = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
    E = PRICE * exact_expected_served(lam_e, s_e)
    E.backward()
    print(f"\n2. По {N} дням: средняя выручка {R.mean():.0f} (точно {E.item():.0f})")
    print(f"   {'':12}{'оценка':>12}{'точно':>12}")
    print(f"   {'∂E/∂lam':12}{g_mc[0]:>12.1f}{lam_e.grad.item():>12.1f}   руб. за +1 клиента в час")
    print(f"   {'∂E/∂s':12}{g_mc[1]:>12.0f}{s_e.grad.item():>12.0f}   руб. за +1 час к заказу "
          f"(≈ {s_e.grad.item() / 60:.0f} за +1 мин)")

    # без базовой линии — тот же ожидаемый градиент, но шум больше
    def spread(baseline, reps=20, n=200):
        out = []
        for r in range(reps):
            idx = slice(r * n, (r + 1) * n)
            l = torch.tensor(lam0, dtype=torch.float64, requires_grad=True)
            lpr = torch.stack([path_log_prob(st, l, torch.tensor(s0, dtype=torch.float64)) for st in stats[idx]])
            Rr = R[idx]
            ((Rr - (Rr.mean() if baseline else 0)) * lpr).mean().backward()
            out.append(l.grad.item())
        return torch.tensor(out).std().item()
    print(f"\n   разброс оценки ∂E/∂lam по 200 дням: без базовой линии {spread(False):.0f}, "
          f"с базовой линией {spread(True):.0f}")
