"""Скрытые события: когда в данных видна только часть журнала (например, в чеках — только моменты
оплат, а приходы, уходы и начала заказов не видны).

Опирается на разделение «эффект / критерий» (magnum.process):

* ``smc_day`` — фильтр частиц по одному дню. Скрытые события (приходы, таймеры терпения, расписание,
  решения when) разыгрываются по самой модели, а наблюдаемый таймер (например, «заказ» -> оплата)
  не разыгрывается: его вклад считается точно — вероятность, что ни один идущий заказ не закончился
  до следующего чека, и плотность, что какой-то закончился ровно в момент чека. Отсюда оценка
  log P(моменты чеков) и выборка полных журналов, согласованных с чеками.
* ``merge_stats`` — сложить достаточные статистики полных журналов (Process.replay) по дням.
* ``fit_em`` — максимум правдоподобия по чекам методом Monte Carlo EM: E-шаг — дорисовать скрытые
  события фильтром частиц при текущих параметрах; M-шаг — точное правдоподобие полных журналов
  (Process.log_likelihood) максимизируется по параметрам градиентом.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random as _random

import torch

from .process import Process, _View
from .queues import Queue


# ------------------------------------------------------------ частица
class _P:
    __slots__ = ("m", "t", "obs", "heap", "hid", "log", "si")


def _clone_state(st):
    return {k: (Queue(v.snapshot()) if isinstance(v, Queue) else v) for k, v in st.items()}


def _floats(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()
            out[k] = v.item() if v.dim() == 0 else v.tolist()
        else:
            out[k] = v
    return out


def smc_day(cls: type[Process], obs_times, observed, end, controls=None, n=100, rng=None, keep=1):
    """Фильтр частиц по одному дню (время в днях от полуночи, день начинается с init()).

    obs_times — отсортированные моменты наблюдаемого события ``observed`` (его вызывает таймер);
    end — конец окна наблюдения: после него события не видны (и не моделируются).
    Возвращает (оценка log P(наблюдений), список из keep полных журналов для Process.replay)."""
    rng = rng or _random.Random(0)
    tname = cls.fired_by[observed]
    pt = cls.resolve(controls)
    pf = _floats(pt)
    view = _View(pt, {})
    lpf, lsf, cdf, ppf = cls.timers[tname].dist(view).floats()
    hidden_dist = {nm: tm.dist(view) for nm, tm in cls.timers.items() if nm != tname}
    structural = {k: int(v) for k, v in pt.items() if cls.params[k].integer}
    sched = cls._schedule(structural, end)
    rates = list(cls.rates.items())
    whens = cls.whens
    effects = cls.effects
    seq = itertools.count()

    def bind(P):
        def start(name, ent):
            if name == tname:
                P.obs[ent] = [P.t, P.t]              # [начало, до какого момента учтено выживание]
            else:
                d = hidden_dist[name].sample(rng)
                tok = next(seq)
                P.hid[(name, ent)] = tok
                heapq.heappush(P.heap, (P.t + d, tok, name, ent))

        def cancel(name, ent):
            if name == tname:
                P.obs.pop(ent, None)
            else:
                P.hid.pop((name, ent), None)
        P.m.__dict__["_hooks"] = (start, cancel)

    def resolve_whens(m):
        again = True
        while again:
            again = False
            for name, cond in whens:
                if cond(m):
                    effects[name](m)
                    again = True
                    break

    def new_particle():
        P = _P()
        P.m = cls._new(cls, pf)
        P.t, P.obs, P.heap, P.hid, P.log, P.si = 0.0, {}, [], {}, None, 0
        bind(P)
        P.m.init()
        resolve_whens(P.m)
        return P

    def clone(Q):
        P = _P()
        P.m = cls._new(cls, pf)
        P.m.__dict__["_state"] = _clone_state(Q.m.__dict__["_state"])
        P.m.__dict__["_next_id"] = Q.m.__dict__["_next_id"]
        P.t, P.si, P.log = Q.t, Q.si, Q.log
        P.obs = {k: list(v) for k, v in Q.obs.items()}
        P.heap, P.hid = list(Q.heap), dict(Q.hid)
        bind(P)
        return P

    def step_nonrate(P, tn):
        """Применить ближайшее событие таймера или расписания в момент tn."""
        m = P.m
        if P.si < len(sched) and sched[P.si][0] == tn:
            effects[sched[P.si][1]](m)
            P.si += 1
        else:
            _, tok, name, ent = heapq.heappop(P.heap)
            del P.hid[(name, ent)]
            ev = cls.timers[name].fires
            P.log = ((tn, ev, ent, "timer"), P.log)
            effects[ev](m, ent)
        resolve_whens(m)

    def next_nonrate(P):
        heap = P.heap
        while heap and P.hid.get((heap[0][2], heap[0][3])) != heap[0][1]:
            heapq.heappop(heap)
        t_tim = heap[0][0] if heap else math.inf
        t_sch = sched[P.si][0] if P.si < len(sched) else math.inf
        return min(t_tim, t_sch)

    def guided_first(P, T):
        """Наблюдаемых таймеров нет, а к T чек будет — значит, на отрезке случится природное событие,
        после которого таймер запустится. Его момент τ предлагаем не по модели, а «от чека назад»:
        T − τ ~ длительность наблюдаемого таймера, урезанная до отрезка. Вес = плотность модели для
        первого природного события в τ / плотность предложения."""
        g = T - P.t
        Fg = cdf(g)
        if Fg <= 0:
            return -math.inf
        d = ppf(rng.random() * Fg)
        tau = T - d
        lw = -(lpf(d) - math.log(Fg))
        m = P.m
        while True:                                   # до τ — только таймеры и расписание, без природы
            tot = sum(float(fn(m)) for _, fn in rates)
            tn = next_nonrate(P)
            if tn >= tau:
                lw -= tot * (tau - P.t)
                P.t = tau
                break
            lw -= tot * (tn - P.t)
            P.t = tn
            step_nonrate(P, tn)
        lam = [float(fn(m)) for _, fn in rates]
        tot = sum(lam)
        if tot <= 0:
            return -math.inf
        u, acc = rng.random() * tot, 0.0
        for (e, _), l in zip(rates, lam):
            acc += l
            if u < acc:
                break
        P.log = ((tau, e, None, "rate"), P.log)
        effects[e](m)
        resolve_whens(m)
        return lw + math.log(tot)

    def advance(P, T, guide=False):
        """Скрытые события до T при условии, что наблюдаемый таймер до T не сработал;
        возвращает log веса: P(ни один наблюдаемый таймер не сработал на отрезке) и поправку предложения."""
        lw = 0.0
        m = P.m
        if guide and not P.obs:
            lw = guided_first(P, T)
            if lw == -math.inf:
                return lw
        while True:
            lam = [float(fn(m)) for _, fn in rates]
            tot = sum(lam)
            t_rate = P.t + rng.expovariate(tot) if tot > 0 else math.inf
            heap = P.heap
            while heap and P.hid.get((heap[0][2], heap[0][3])) != heap[0][1]:
                heapq.heappop(heap)
            t_tim = heap[0][0] if heap else math.inf
            t_sch = sched[P.si][0] if P.si < len(sched) else math.inf
            tn = min(t_rate, t_tim, t_sch)
            if tn >= T:
                break
            # выживание наблюдаемых таймеров до tn учитываем в конце (оно телескопируется)
            P.t = tn
            if t_sch == tn:
                effects[sched[P.si][1]](m)
                P.si += 1
            elif t_rate == tn:
                u, acc = rng.random() * tot, 0.0
                for (e, _), l in zip(rates, lam):
                    acc += l
                    if u < acc:
                        break
                P.log = ((tn, e, None, "rate"), P.log)
                effects[e](m)
            else:
                _, tok, name, ent = heapq.heappop(heap)
                del P.hid[(name, ent)]
                ev = cls.timers[name].fires
                P.log = ((tn, ev, ent, "timer"), P.log)
                effects[ev](m, ent)
            resolve_whens(m)
        P.t = T
        for v in P.obs.values():
            lw += lsf(T - v[0]) - lsf(v[1] - v[0])
            v[1] = T
        return lw

    parts = [new_particle() for _ in range(n)]
    logw = [0.0] * n
    total = 0.0

    def normalize():
        mx = max(logw)
        if mx == -math.inf:
            return -math.inf, None
        w = [math.exp(x - mx) for x in logw]
        sw = sum(w)
        return mx + math.log(sw / n), [x / sw for x in w]

    for tk in obs_times:
        for i, S in enumerate(parts):
            if logw[i] == -math.inf:
                continue
            # Скрытый путь до tk разыгрываем по модели; если наблюдаемых таймеров нет, первое природное
            # событие предлагаем «от чека назад» (иначе почти все частицы объясняли бы чек плохо).
            P = S
            lw = advance(P, tk, guide=True)
            if lw == -math.inf or not P.obs:
                logw[i] = -math.inf
                continue
            hz = [(ent, math.exp(lpf(tk - v[0]) - lsf(tk - v[0]))) for ent, v in P.obs.items()]
            sh = sum(h for _, h in hz)
            if sh <= 0:
                logw[i] = -math.inf
                continue
            u, acc = rng.random() * sh, 0.0
            for ent, h in hz:
                acc += h
                if u < acc:
                    break
            del P.obs[ent]
            P.log = ((tk, observed, ent, "timer"), P.log)
            effects[observed](P.m, ent)
            resolve_whens(P.m)
            logw[i] += lw + math.log(sh)
        inc, w = normalize()
        if w is None:
            return -math.inf, []
        total += inc
        # систематический ресэмплинг на каждом наблюдении
        u0 = rng.random() / n
        idx, c, j = [], w[0], 0
        for k in range(n):
            u = u0 + k / n
            while u > c and j < n - 1:
                j += 1
                c += w[j]
            idx.append(j)
        seen = set()
        new = []
        for j in idx:
            new.append(parts[j] if j not in seen else clone(parts[j]))
            seen.add(j)
        parts, logw = new, [0.0] * n

    for i, P in enumerate(parts):
        logw[i] = advance(P, end)
    inc, w = normalize()
    total += inc
    logs = []
    for _ in range(keep):
        u, acc = rng.random(), 0.0
        for i, wi in enumerate(w):
            acc += wi
            if u < acc:
                break
        out, node = [], parts[i].log
        while node is not None:
            out.append(node[0])
            node = node[1]
        logs.append(out[::-1])
    return total, logs


# ------------------------------------------------------------ статистики и EM
def merge_stats(stats_list):
    count, exposure, done, cens = {}, {}, {}, {}
    for st in stats_list:
        for k, v in st["count"].items():
            count[k] = count.get(k, 0) + v
        for k, v in st["exposure"].items():
            exposure[k] = exposure.get(k, 0.0) + v
        for nm in st["done"]:
            done.setdefault(nm, []).append(st["done"][nm])
            cens.setdefault(nm, []).append(st["cens"][nm])
    return {"count": count, "exposure": exposure,
            "done": {k: torch.cat(v) for k, v in done.items()},
            "cens": {k: torch.cat(v) for k, v in cens.items()}}


def maximize(cls, stats, names, start, fixed=None, steps=200):
    """M-шаг: максимум точного правдоподобия полных журналов по names (в логарифмах, L-BFGS)."""
    raw = {k: torch.log(torch.as_tensor(start[k], dtype=torch.float64).clamp_min(1e-12)).clone().requires_grad_(True)
           for k in names}
    fixed = fixed or {}
    opt = torch.optim.LBFGS(list(raw.values()), max_iter=steps, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -cls.log_likelihood(stats, {**fixed, **{k: torch.exp(v) for k, v in raw.items()}})
        loss.backward()
        return loss
    opt.step(closure)
    return {k: torch.exp(v).detach() for k, v in raw.items()}


def fit_em(cls, days, observed, end, names, start, fixed=None, iters=15, n=64, seed=0, keep=1, verbose=True):
    """Monte Carlo EM по дням: days — список отсортированных моментов наблюдаемого события (время от
    полуночи, в днях). Возвращает историю параметров и оценок log P по итерациям."""
    rng = _random.Random(seed)
    theta = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in start.items()}
    fixed = fixed or {}
    hist = []
    for it in range(iters):
        ctl = {**fixed, **theta}
        ll, sts = 0.0, []
        for obs in days:
            l, logs = smc_day(cls, obs, observed, end, ctl, n=n, rng=rng, keep=keep)
            ll += l
            sts += [cls.replay(lg, end, controls=ctl) for lg in logs]
        stats = merge_stats(sts)
        theta = maximize(cls, stats, names, theta, fixed)
        hist.append((ll, {k: v.clone() for k, v in theta.items()}))
        if verbose:
            print(f"  итерация {it + 1}: log P(чеков) ≈ {ll:.1f}  " +
                  "  ".join(f"{k}={_short(v)}" for k, v in theta.items()), flush=True)
    return hist


def _short(v):
    return f"{v.item():.4g}" if v.dim() == 0 else "[…]"


def fit_score(cls, days, observed, end, names, start, fixed=None, iters=40, lr=0.05, n=64, keep=2, seed=0,
              bounds=None, average_last=10, verbose=True):
    """Максимум правдоподобия по наблюдаемым моментам градиентным подъёмом (Adam в логарифмах).

    Градиент log P(чеков) по тождеству Фишера: ∇ log P(чеков | θ) = E[ ∇ log p(полный журнал | θ) | чеки ].
    Полные журналы берём из фильтра частиц при текущем θ, ∇ log p(полный журнал) — точный (autograd по
    Process.log_likelihood). В отличие от EM, шаг не застревает, когда скрытой информации много.
    bounds — {имя: (низ, верх)} для параметров, которые данные почти не ограничивают.
    Итог — среднее параметров (в логарифмах) по последним average_last итерациям."""
    rng = _random.Random(seed)
    fixed = fixed or {}
    bounds = bounds or {}
    raw = {k: torch.log(torch.as_tensor(start[k], dtype=torch.float64).clamp_min(1e-9)).clone().requires_grad_(True)
           for k in names}
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    hist = []
    for it in range(iters):
        theta = {k: torch.exp(v.detach()) for k, v in raw.items()}
        ctl = {**fixed, **theta}
        ll, sts = 0.0, []
        for obs in days:
            l, logs = smc_day(cls, obs, observed, end, ctl, n=n, rng=rng, keep=keep)
            ll += l
            sts += [cls.replay(lg, end, controls=ctl) for lg in logs]
        stats = merge_stats(sts)
        loss = -cls.log_likelihood(stats, {**fixed, **{k: torch.exp(v) for k, v in raw.items()}}) / keep
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k, (lo, hi) in bounds.items():
                raw[k].clamp_(math.log(lo), math.log(hi))
        hist.append((ll, theta))
        if verbose:
            print(f"  шаг {it + 1}: log P(чеков) ≈ {ll:.1f}  " +
                  "  ".join(f"{k}={_short(v)}" for k, v in theta.items()), flush=True)
    tail = hist[-average_last:]
    est = {k: torch.exp(torch.stack([torch.log(h[1][k].clamp_min(1e-12)) for h in tail]).mean(0)) for k in names}
    return est, hist
