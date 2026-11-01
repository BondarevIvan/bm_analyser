"""Кофейня как в coffee_queue.py (run_coffee.py), но события отделены от критериев их наступления
(magnum.process). Параметры по умолчанию — те же, что в coffee_queue.py.

Эффекты — только что меняется; критерии — когда:
  открытие       daily: каждый день в open_hour
  закрытие       daily: каждый день в close_hour
  приход         rate: clients_per_hour в час, пока открыто; иначе 0
  начало заказа  when: есть свободный мастер и очередь
  конец заказа   таймер «заказ» от начала, длительность логнормальная (среднее, разброс)
  уход           таймер «терпение» от прихода, отменяется, если заказ начался

Нормальные длительности из run_coffee заменены логнормальными с теми же средним и разбросом:
длительность не бывает отрицательной, и у неё есть плотность для правдоподобия.
После закрытия новые клиенты не приходят, а уже стоящих в очереди дообслуживают (как в coffee_queue.py).
"""
import magnum as mg
from magnum.process import LogNormal, Process, daily, rate, timer, when


class CoffeeProcess(Process):
    open_hour = mg.param(8, low=0, high=23)
    close_hour = mg.param(22, low=0, high=23)
    clients_per_hour = mg.param(15.0, low=0)
    masters = mg.param(4, low=1)
    order_mean = mg.param(mg.seconds(420), low=0, low_open=True)
    order_std = mg.param(mg.seconds(10), low=0, low_open=True)
    wait_mean = mg.param(mg.seconds(330), low=0, low_open=True)
    wait_std = mg.param(mg.seconds(30), low=0, low_open=True)

    order = timer(lambda m: LogNormal(m.order_mean, m.order_std), fires="finish")
    patience = timer(lambda m: LogNormal(m.wait_mean, m.wait_std), fires="leave")

    def init(self):
        self.is_open = False                 # время 0 — полночь
        self.queue = mg.Queue()
        self.free = int(self.masters)
        self.served = 0
        self.lost = 0

    @daily(lambda m: m.open_hour * mg.HOUR)
    def open(self):
        self.is_open = True

    @daily(lambda m: m.close_hour * mg.HOUR)
    def close(self):
        self.is_open = False

    @rate(lambda m: m.clients_per_hour / mg.HOUR if m.is_open else 0.0)
    def arrive(self):
        cid = self.new_id()
        self.queue.push(cid)
        self.start("patience", cid)

    @when(lambda m: m.free > 0 and len(m.queue) > 0)
    def start_order(self):
        cid = self.queue.pop()
        self.cancel("patience", cid)
        self.free -= 1
        self.start("order", cid)

    def finish(self, cid):
        self.free += 1
        self.served += 1

    def leave(self, cid):
        self.queue.remove(cid)
        self.lost += 1
