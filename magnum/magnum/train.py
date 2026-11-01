"""Обучение параметров модели градиентными методами (torch).

Параметры модели — тензоры. Если передать в ``run(controls=...)`` тензор с ``requires_grad``:

* показатели, которые модель считает из него арифметикой (выручка = цена × количество), остаются
  тензорами с градиентом в ``result["final"]`` — производная «по траектории»;
* случайные исходы, распределение которых зависит от параметра (``rng.*``, ``choose``, ``arrival``),
  дают ``result["log_prob"]`` — логарифм вероятности выпавшей траектории.

``expected`` собирает из серии запусков оценку E[показатель], градиент которой — несмещённая оценка
градиента ожидания (pathwise + score function / REINFORCE)::

    params = Bakery.make_params(names=["price_mult"])           # тензоры, price_mult с requires_grad
    opt = torch.optim.Adam(mg.trainable(params), lr=0.01)
    for step in range(100):
        runs = [Bakery.run(controls=params, seed=step * 64 + i, horizon=28, debug=False, record=[])
                for i in range(64)]
        loss = -mg.expected(runs, lambda r: r["final"]["profit"])
        opt.zero_grad(); loss.backward(); opt.step()
        Bakery.clamp_(params)                                   # вернуть в границы параметров

Ограничения: время событий (after=..., моменты прихода) — обычные числа, поэтому через них градиент
идёт только score-путём; оценка несмещённая, но с дисперсией — нужны десятки запусков на шаг.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch


def trainable(params: dict) -> list:
    """Тензоры с requires_grad из словаря параметров — для torch.optim."""
    return [t for t in params.values() if isinstance(t, torch.Tensor) and t.requires_grad]


def expected(runs: Iterable[dict], value: Callable[[dict], object] | str = None) -> torch.Tensor:
    """Дифференцируемая оценка E[value] по серии запусков с разными сидами.

    :param runs: результаты Model.run (одни и те же controls, разные seed)
    :param value: функция результат -> число/тензор, или имя переменной из final
    :return: 0-мерный тензор; его значение — среднее value, градиент — оценка ∇E[value]
             (pathwise-часть + score function с базой «среднее по остальным запускам»)
    """
    runs = list(runs)
    if not runs:
        raise ValueError("expected: пустой список запусков")
    if isinstance(value, str):
        name = value
        value = lambda r: r["final"][name]  # noqa: E731
    f = torch.stack([torch.as_tensor(value(r), dtype=torch.float64).reshape(()) for r in runs])
    lp = torch.stack([torch.as_tensor(r["log_prob"], dtype=torch.float64).reshape(()) for r in runs])
    n = len(runs)
    fd = f.detach()
    baseline = (fd.sum() - fd) / (n - 1) if n > 1 else torch.zeros_like(fd)
    # (lp − lp.detach()) равно 0 по значению, но несёт градиент log p
    surrogate = f + (fd - baseline) * (lp - lp.detach())
    return surrogate.mean()
