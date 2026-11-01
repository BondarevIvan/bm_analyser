"""Кофейня из coffee_process.py, но поток клиентов свой в каждом часе дня (per_hour[час]),
а не одна интенсивность на весь день: в реальных чеках к обеду клиентов в разы больше, чем утром.

Время суток входит в интенсивность через состояние: событие расписания tick каждый час
увеличивает self.hour.
"""
import magnum as mg
from magnum.process import daily, rate

from models.coffee.coffee_process import CoffeeProcess


class CoffeeHours(CoffeeProcess):
    per_hour = mg.param([0.0] * 8 + [10.0] * 10 + [0.0] * 6, low=0)    # клиентов в час, по часам суток

    def init(self):
        CoffeeProcess.init(self)
        self.hour = 0

    @daily(lambda m: [h * mg.HOUR for h in range(1, 24)])
    def tick(self):
        self.hour += 1

    @rate(lambda m: m.per_hour[m.hour] / mg.HOUR if m.is_open else 0.0)
    def arrive(self):
        CoffeeProcess.arrive(self)
