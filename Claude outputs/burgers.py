import heapq
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import magnum.magnum as mg
from magnum.magnum.process import (
    Process,
    LogNormal,
    daily,
    rate,
    timer,
    when,
)


ITEM_COUNTS = [1, 2, 3, 4]   # сколько позиций может быть в чеке (варианты probability_count_wishes_in_receipt)
MAX_ITEMS = len(ITEM_COUNTS)
EXACT = 0.1 * mg.SECOND      # разброс «почти фиксированного» времени операции


# ------------------------------------------------------------------ граф дорог
# Вершина 0 — точка (кухня), остальные — адреса доставки. Рёбра — дороги с временем проезда
# (параметр road_times, его можно подбирать по данным). Время до адреса — кратчайший путь.
NODES = ["Гриль 63", "Центр", "Вокзал", "Университет", "Парк", "Микрорайон"]
POINT = 0
ADDRESSES = [1, 2, 3, 4, 5]
EDGES = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)]
ROAD_MINUTES = [4.0, 6.0, 5.0, 3.0, 9.0, 7.0, 8.0, 6.0]
ADDRESS_PROBS = [0.3, 0.2, 0.2, 0.15, 0.15]


def travel_time(road_times, src, dst):
    """Время кратчайшего пути src -> dst по графу (Дейкстра).

    Путь выбирается по числам, а время — сумма тензоров road_times по рёбрам пути, поэтому
    градиент правдоподобия доходит до времени каждой дороги."""
    w = [float(x) for x in (road_times.detach() if isinstance(road_times, torch.Tensor) else road_times)]
    adj = {}
    for e, (u, v) in enumerate(EDGES):
        adj.setdefault(u, []).append((v, e))
        adj.setdefault(v, []).append((u, e))
    dist, prev = {src: 0.0}, {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, e in adj.get(u, []):
            nd = d + w[e]
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, (u, e)
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        raise ValueError(f"адрес {NODES[dst]} недостижим по графу дорог")
    total, node = 0.0, dst
    while node != src:
        node, e = prev[node]
        total = total + road_times[e]
    return total


# ------------------------------------------------------------------ API
class Request(BaseModel):
    clients_per_hour: float = 10.0

    frying_time: float = 4.0
    burger_assembly_time: float = 1.0
    packaging_time: float = 0.5          # минут на одну позицию: упаковка чека = позиций × это время

    max_burgers_at_time: int = 6
    count_assemblers: int = 2
    count_packers: int = 1

    # вероятность, что в чеке 1, 2, 3, 4 позиции (нормируется, сумма не обязана быть 1)
    probability_count_wishes_in_receipt: list[float] = Field(
        default_factory=lambda: [0.4, 0.3, 0.2, 0.1], min_length=MAX_ITEMS, max_length=MAX_ITEMS)

    # доставка
    count_deliveries: int = 5
    handover_time: float = 1.0           # минут на передачу заказа клиенту
    ride_spread: float = 0.15            # разброс времени в пути (доля от времени по графу)
    road_times: list[float] = Field(     # минут на каждую дорогу графа (порядок как в EDGES)
        default_factory=lambda: list(ROAD_MINUTES), min_length=len(EDGES), max_length=len(EDGES))
    address_probs: list[float] = Field(  # вероятности адресов (порядок как в ADDRESSES)
        default_factory=lambda: list(ADDRESS_PROBS), min_length=len(ADDRESSES), max_length=len(ADDRESSES))

    seed: int = 1
    days: int = 30
    quantile: float = 50.0


class Response(BaseModel):
    times: list[float]            # время на кухне: от прихода до упаковки, мин
    items: list[int]              # сколько позиций было в каждом заказе
    average: float                # среднее время на кухне, мин
    average_by_items: dict[int, float]   # среднее время на кухне по числу позиций, мин
    quantile: float

    delivery_times: list[float]   # от прихода заказа до передачи клиенту, мин
    delivery_average: float
    delivery_quantile: float
    courier_wait_average: float   # сколько готовый заказ ждал свободного курьера, мин


# ------------------------------------------------------------------ заказ
@dataclass
class Order:
    order_id: int = 0
    n_items: int = 1              # позиций (бургеров) в чеке
    arrived: float = 0.0          # появился на мониторе
    frying_start: float = 0.0     # первая котлета заказа на гриле
    frying_done: float = 0.0      # последняя котлета заказа готова
    assembly_start: float = 0.0   # сборщик взял первый бургер заказа
    assembly_done: float = 0.0    # собран последний бургер, заказ встал в очередь на упаковку
    packing_start: float = 0.0    # упаковщик взял заказ
    done: float = 0.0             # заказ упакован — готов к выдаче курьеру
    assembly_waits: list = field(default_factory=list)   # ожидание сборки по каждому бургеру
    # доставка
    address: int = -1             # вершина графа
    courier_left: float = 0.0     # курьер взял заказ и выехал
    at_client: float = 0.0        # доехал до адреса
    delivered: float = 0.0        # отдал заказ
    courier_back: float = 0.0     # вернулся на точку

    @property
    def total(self):
        return self.done - self.arrived

    @property
    def total_delivery(self):
        return self.delivered - self.arrived

    @property
    def wait_frying(self):
        return self.frying_start - self.arrived

    @property
    def wait_assembly(self):
        return sum(self.assembly_waits) / len(self.assembly_waits)

    @property
    def wait_packing(self):
        return self.packing_start - self.assembly_done

    @property
    def wait_courier(self):
        return self.courier_left - self.done


# ------------------------------------------------------------------ кухня
class Kitchen(Process):
    """
    В чеке 1–4 позиции (бургера). Каждый бургер жарится и собирается отдельно,
    упаковывается заказ целиком, когда собраны все его бургеры; упаковка длится
    позиций × packaging_time:

        монитор (бургеры)  --(место на гриле)-->  жарка  --> очередь на сборку
        очередь на сборку  --(свободный сборщик)-->  сборка бургера
        все бургеры заказа собраны  --> очередь на упаковку
        очередь на упаковку  --(свободный упаковщик)-->  упаковка заказа  --> ready_orders

    Выход кухни — очередь ready_orders с сущностями (номер заказа, позиций). Кто её разбирает
    (доставка, выдача в зале), кухня не знает; одна кухня просто копит в ней готовые заказы.
    """
    clients_per_hour = mg.param(25.0)

    frying_time = mg.param(4 * mg.MINUTE)
    burger_assembly_time = mg.param(1 * mg.MINUTE)
    packaging_time = mg.param(0.5 * mg.MINUTE)       # на одну позицию

    # ограниченные ресурсы
    max_burgers_at_time = mg.param(6)
    count_assemblers = mg.param(2)
    count_packers = mg.param(1)

    # вероятность, что в чеке 1, 2, 3, 4 позиции
    probability_count_wishes_in_receipt = mg.param([0.4, 0.3, 0.2, 0.1])

    # timer ждёт распределение (с .sample), а не число; разброс почти нулевой — время фиксированное
    frying = timer(lambda m: LogNormal(m.frying_time, EXACT), fires="fried")
    building = timer(lambda m: LogNormal(m.burger_assembly_time, EXACT), fires="assembled")
    # упаковка зависит от заказа: order = (номер, позиций); key — для правдоподобия важно только число позиций
    packing = timer(lambda m, order: LogNormal(order[1] * m.packaging_time, EXACT),
                    fires="packed", key=lambda order: order[1])

    def init(self):
        # очереди ожидания
        self._monitor = mg.Queue()          # бургеры, ждущие гриля
        self._assembly_queue = mg.Queue()   # бургеры, ждущие сборщика
        self._packing_queue = mg.Queue()    # заказы, ждущие упаковщика
        # кто сейчас в работе
        self._frying_slots = mg.Queue()
        self._assemblers_busy = mg.Queue()
        self._packers_busy = mg.Queue()
        # сколько бургеров заказа ещё не собрано и сколько в нём позиций
        self._left_to_assemble = {}
        self._items_in_order = {}
        # выход кухни: готовые заказы (номер, позиций)
        self.ready_orders = mg.Queue()

        self._orders = 0
        self._done = 0
        self.is_open = False

    @daily(lambda m: mg.HOUR * 10.0)
    def open(self):
        self.is_open = True

    @daily(lambda m: mg.HOUR * 20.0)
    def close(self):
        self.is_open = False

    # --- приход заказа
    @rate(lambda m: m.clients_per_hour / mg.HOUR if m.is_open else 0.0)
    def arrived_new_order(self):
        n_items = self.choose(lambda m: m.probability_count_wishes_in_receipt, values=ITEM_COUNTS)
        self._orders += 1
        order_id = self._orders
        self._left_to_assemble[order_id] = n_items
        self._items_in_order[order_id] = n_items
        for i in range(n_items):
            self._monitor.push((order_id, i))

    # --- жарка (по бургеру)
    @when(lambda m: len(m._monitor) > 0 and len(m._frying_slots) < m.max_burgers_at_time)
    def start_frying(self):
        burger = self._monitor.pop()
        self._frying_slots.push(burger)
        self.start("frying", burger)

    def fried(self, burger):
        self._frying_slots.remove(burger)      # место на гриле освободилось
        self._assembly_queue.push(burger)

    # --- сборка (по бургеру)
    @when(lambda m: len(m._assembly_queue) > 0 and len(m._assemblers_busy) < m.count_assemblers)
    def start_assembly(self):
        burger = self._assembly_queue.pop()
        self._assemblers_busy.push(burger)
        self.start("building", burger)

    def assembled(self, burger):
        self._assemblers_busy.remove(burger)   # сборщик освободился
        order_id = burger[0]
        self._left_to_assemble[order_id] -= 1
        if self._left_to_assemble[order_id] == 0:   # собран весь заказ — на упаковку
            del self._left_to_assemble[order_id]
            self._packing_queue.push((order_id, self._items_in_order.pop(order_id)))

    # --- упаковка (заказ целиком, позиций × время)
    @when(lambda m: len(m._packing_queue) > 0 and len(m._packers_busy) < m.count_packers)
    def start_packing(self):
        order = self._packing_queue.pop()
        self._packers_busy.push(order)
        self.start("packing", order)

    def packed(self, order):
        self._packers_busy.remove(order)       # упаковщик освободился
        self._done += 1
        self.ready_orders.push(order)          # готово — дальше не дело кухни


Cooking = Kitchen   # старое имя


# ------------------------------------------------------------------ доставка
class Delivery(Process):
    """
    Курьеры развозят готовые заказы из очереди ready_orders (её создаёт и наполняет кухня):

        свободный курьер + готовый заказ  -->  выехал (адрес — случайный, address_probs)
        через время пути по графу дорог  -->  у клиента
        через handover_time  -->  отдал заказ
        через время пути обратно  -->  вернулся на точку, снова свободен

    Сущность доставки — (номер заказа, позиций, адрес).
    """
    count_deliveries = mg.param(10)
    handover_time = mg.param(1 * mg.MINUTE)
    ride_spread = mg.param(0.15)                                   # разброс времени в пути, доля
    road_times = mg.param([t * mg.MINUTE for t in ROAD_MINUTES])  # время по каждой дороге (EDGES)
    address_probs = mg.param(ADDRESS_PROBS)                        # вероятности адресов (ADDRESSES)

    to_client = timer(lambda m, d: Delivery._ride(m, d), fires="arrived_at_client", key=lambda d: d[2])
    handing = timer(lambda m: LogNormal(m.handover_time, EXACT), fires="handed_over")
    to_point = timer(lambda m, d: Delivery._ride(m, d), fires="returned", key=lambda d: d[2])

    @staticmethod
    def _ride(m, d):
        t = travel_time(m.road_times, POINT, d[2])    # дороги двусторонние: туда и обратно одинаково
        return LogNormal(t, m.ride_spread * t)

    def init(self):
        self._couriers_busy = mg.Queue()    # заказы, с которыми курьер сейчас в пути
        self._delivered = 0

    @when(lambda m: len(m.ready_orders) > 0 and len(m._couriers_busy) < m.count_deliveries)
    def courier_left(self):
        order_id, n_items = self.ready_orders.pop()
        address = self.choose(lambda m: m.address_probs, values=ADDRESSES)
        d = (order_id, n_items, address)
        self._couriers_busy.push(d)          # курьер больше не свободен
        self.start("to_client", d)

    def arrived_at_client(self, d):
        self.start("handing", d)

    def handed_over(self, d):
        self._delivered += 1
        self.start("to_point", d)

    def returned(self, d):
        self._couriers_busy.remove(d)        # вернулся на точку — снова свободен


# ------------------------------------------------------------------ вся модель
class Restaurant(Kitchen, Delivery):
    def init(self):
        Kitchen.init(self)      # первой — кухня: она создаёт ready_orders
        Delivery.init(self)


# ------------------------------------------------------------------ разбор журнала
def collect_orders(log):
    """Восстановить моменты этапов каждого заказа по журналу [(t, событие, сущность, вид)].

    Решения when пишутся в журнал без сущности, поэтому очереди проигрываются здесь же:
    все очереди FIFO, значит взятый из очереди элемент — тот, что стоит в ней первым.
    Запись choose идёт сразу после события, в котором сделан выбор: после arrived_new_order —
    число позиций, после courier_left — адрес.
    """
    orders = {}
    fried_at = {}
    left = {}
    monitor, assembly_q, packing_q, ready_q = deque(), deque(), deque(), deque()
    pending = None          # кому относится следующая запись choose: ("items", oid) / ("address", oid)
    for t, event, ent, _kind in log:
        if event == "arrived_new_order":
            oid = len(orders) + 1
            orders[oid] = Order(order_id=oid, arrived=t)
            pending = ("items", oid)
        elif event == "choose":
            what, oid = pending
            if what == "items":
                n = ITEM_COUNTS[ent]
                orders[oid].n_items = n
                left[oid] = n
                monitor.extend((oid, i) for i in range(n))
            else:
                orders[oid].address = ADDRESSES[ent]
            pending = None
        elif event == "start_frying":
            oid, _ = monitor.popleft()
            if orders[oid].frying_start == 0.0:
                orders[oid].frying_start = t
        elif event == "fried":
            fried_at[ent] = t
            orders[ent[0]].frying_done = t
            assembly_q.append(ent)
        elif event == "start_assembly":
            burger = assembly_q.popleft()
            o = orders[burger[0]]
            if not o.assembly_waits:
                o.assembly_start = t
            o.assembly_waits.append(t - fried_at.pop(burger))
        elif event == "assembled":
            oid = ent[0]
            left[oid] -= 1
            if left[oid] == 0:
                orders[oid].assembly_done = t
                packing_q.append(oid)
        elif event == "start_packing":
            orders[packing_q.popleft()].packing_start = t
        elif event == "packed":
            orders[ent[0]].done = t
            ready_q.append(ent[0])
        elif event == "courier_left":
            oid = ready_q.popleft()
            orders[oid].courier_left = t
            pending = ("address", oid)
        elif event == "arrived_at_client":
            orders[ent[0]].at_client = t
        elif event == "handed_over":
            orders[ent[0]].delivered = t
        elif event == "returned":
            orders[ent[0]].courier_back = t
    return [o for o in orders.values() if o.done > 0]


def compute_quantile(arr: list[float], q: float) -> float:
    """Выборочный квантиль уровня q ∈ [0, 1] (линейная интерполяция)."""
    if not arr:
        return 0.0
    arr = sorted(arr)
    pos = min(max(q, 0.0), 1.0) * (len(arr) - 1)
    i = int(pos)
    j = min(i + 1, len(arr) - 1)
    return arr[i] + (arr[j] - arr[i]) * (pos - i)


# ------------------------------------------------------------------ бар (без изменений)
class Bar(Process):
    # Сотрудники
    count_barman = mg.param(2)
    count_waiter = mg.param(0)
    count_cooks = mg.param(2)
    count_cleaners = mg.param(1)

    ## Зарплаты
    salary_barman = mg.param(65_000)
    salary_cleaner = mg.param(90_000)
    salary_waiter = mg.param(90_000)
    salary_cook = mg.param(65_000)

    # Про посетителей
    average_receipt = mg.param(900.0)
    clients_per_hour = mg.param([
        0.0,  # Понедельник
        8.0,  # Вторник
        8.0,  # Среда
        8.0,  # Четверг
        4.0,  # Пятница
        0.0,  # Суббота
        8.0,  # Воскресенье
    ])
    CAC = mg.param(100.0)

    # наценка в процентах
    average_markup = mg.param(100.0)

    # Аренда
    rent_value = mg.param(81_900)

    def init(self):
        self.bank = 0.0
        self.day_of_week = 0
        self.day_of_month = 0
        self.is_opened = False
        self.count_clients = 0
        self.rent_need_pay = False
        self.salary_need_pay = False

    @daily(lambda m: 6 * mg.HOUR)
    def update_day(self):
        self.day_of_week = (self.day_of_week + 1) % 7
        self.day_of_month = (self.day_of_month + 1) % 30
        if self.day_of_month in [5, 23]:
            self.salary_need_pay = True
        if self.day_of_month in [28]:
            self.rent_need_pay = True

    @daily(lambda m: 20 * mg.HOUR)
    def open(self):
        self.is_opened = True

    @daily(lambda m: 4 * mg.HOUR)
    def close(self):
        self.is_opened = False

    @rate(lambda m: m.clients_per_hour[m.day_of_week] / mg.HOUR if m.is_opened else 0.0)
    def client_arrivied(self):
        self.bank += self.average_receipt
        clean_cost = self.average_receipt / (1 + self.average_markup / 100.0)
        self.bank -= clean_cost
        self.count_clients += 1
        self.bank -= self.CAC

    @when(lambda m: m.salary_need_pay)
    def salary(self):
        salaries = [self.salary_barman, self.salary_cleaner, self.salary_waiter, self.salary_cook]
        counts = [self.count_barman, self.count_cleaners, self.count_waiter, self.count_cooks]
        for salary, count in zip(salaries, counts):
            self.bank -= salary * count * 0.5
        self.salary_need_pay = False

    @when(lambda m: m.rent_need_pay)
    def rent(self):
        self.bank -= self.rent_value
        self.rent_need_pay = False


def bar_simulate():
    bank = 0.0
    clients = 0.0
    seeds = 20
    for seed in range(1, seeds + 1):
        _log, state = Bar.simulate(horizon=30 * mg.DAY, seed=seed)
        bank += float(state["bank"])
        clients += int(state["count_clients"])
    print(bank / seeds, clients / seeds)


# ------------------------------------------------------------------ прогоны
def restaurant_simulate(controls=None, seed=1, days=30, plot=True):
    log, state = Restaurant.simulate(controls=controls, horizon=days * mg.DAY, seed=seed)
    orders = collect_orders(log)
    delivered = [o for o in orders if o.delivered > 0]
    print(f"параметры: {controls or 'по умолчанию'}")
    print(f"  заказов пришло: {state['_orders']}, приготовлено: {state['_done']}, "
          f"доставлено: {state['_delivered']}")

    def avg(xs):
        return sum(xs) / len(xs) / mg.MINUTE if xs else float("nan")

    print(f"  кухня: {avg([o.total for o in orders]):.1f} мин "
          f"(ожидание жарки {avg([o.wait_frying for o in orders]):.1f}, "
          f"сборки {avg([o.wait_assembly for o in orders]):.1f}, "
          f"упаковки {avg([o.wait_packing for o in orders]):.1f})")
    print(f"  доставка: от заказа до клиента {avg([o.total_delivery for o in delivered]):.1f} мин, "
          f"ожидание курьера {avg([o.wait_courier for o in delivered]):.1f}, "
          f"в пути {avg([o.at_client - o.courier_left for o in delivered]):.1f}")
    for a in ADDRESSES:
        xs = [o.total_delivery for o in delivered if o.address == a]
        if xs:
            print(f"    {NODES[a]:<12} {len(xs):>5} заказов, {avg(xs):.1f} мин "
                  f"(по графу {float(travel_time(Restaurant.params['road_times'].default, POINT, a)) / mg.MINUTE:.0f} мин в одну сторону)")

    if plot:
        plt.hist([o.total_delivery / mg.MINUTE for o in delivered], bins=40)
        plt.xlabel("время от заказа до передачи клиенту, мин")
        plt.ylabel("заказов")
        plt.savefig("a.jpg")
    return orders


app = FastAPI()


@app.post("/cooking")
def make_request(r: Request) -> Response:   # без async: симуляция тяжёлая, FastAPI запустит её в потоке
    log, _ = Restaurant.simulate(controls={
        "clients_per_hour": r.clients_per_hour,
        "frying_time": r.frying_time * mg.MINUTE,
        "burger_assembly_time": r.burger_assembly_time * mg.MINUTE,
        "packaging_time": r.packaging_time * mg.MINUTE,
        "max_burgers_at_time": r.max_burgers_at_time,
        "count_assemblers": r.count_assemblers,
        "count_packers": r.count_packers,
        "probability_count_wishes_in_receipt": r.probability_count_wishes_in_receipt,
        "count_deliveries": r.count_deliveries,
        "handover_time": r.handover_time * mg.MINUTE,
        "ride_spread": r.ride_spread,
        "road_times": [t * mg.MINUTE for t in r.road_times],
        "address_probs": r.address_probs,
    }, horizon=r.days * mg.DAY, seed=r.seed)

    orders = collect_orders(log)
    durations = [o.total / mg.MINUTE for o in orders]
    items = [o.n_items for o in orders]

    by_items = {}
    for k in ITEM_COUNTS:
        ks = [d for d, n in zip(durations, items) if n == k]
        if ks:
            by_items[k] = sum(ks) / len(ks)

    delivered = [o for o in orders if o.delivered > 0]
    delivery = [o.total_delivery / mg.MINUTE for o in delivered]
    waits = [o.wait_courier / mg.MINUTE for o in delivered]

    return Response(
        times=durations,
        items=items,
        average=sum(durations) / len(durations) if durations else 0.0,
        average_by_items=by_items,
        quantile=compute_quantile(durations, r.quantile / 100.0),
        delivery_times=delivery,
        delivery_average=sum(delivery) / len(delivery) if delivery else 0.0,
        delivery_quantile=compute_quantile(delivery, r.quantile / 100.0),
        courier_wait_average=sum(waits) / len(waits) if waits else 0.0,
    )


@app.get("/")
async def cooking_ui():
    return FileResponse("/root/workdir/bm_analyser/models/bar/cooking.html")


if __name__ == "__main__":
    restaurant_simulate()
    restaurant_simulate({"count_deliveries": 4}, plot=False)
