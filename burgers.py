import heapq
import magnum.magnum as mg
from magnum.magnum.process import (
    Process,
    LogNormal,
    daily,
    rate,
    timer,
    when,
)
import matplotlib.pyplot as plt
import torch
from collections import deque
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ITEM_COUNTS = [1, 2, 3, 4]   # сколько позиций может быть в чеке (варианты probability_count_wishes_in_receipt)
MAX_ITEMS = len(ITEM_COUNTS)


# ------------------------------------------------------------------ граф дорог
# Вершина 0 — точка (кухня), остальные — адреса доставки. Рёбра — двусторонние дороги, у каждой
# своё время проезда (параметр road_times, его можно подбирать по данным). Время до адреса —
# кратчайший путь по графу.
NODES = ["Точка", "Центр", "Вокзал", "Университет", "Парк", "Микрорайон"]
POINT = 0
ADDRESSES = [1, 2, 3, 4, 5]
EDGES = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)]
ROAD_MINUTES = [4.0, 6.0, 5.0, 3.0, 9.0, 7.0, 8.0, 6.0]   # время по каждой дороге из EDGES, мин
ADDRESS_PROBS = [0.3, 0.2, 0.2, 0.15, 0.15]                # вероятность адреса из ADDRESSES


def travel_time(road_times, src, dst):
    """Время кратчайшего пути src -> dst (Дейкстра).

    Путь ищется по числам, а время — сумма элементов road_times по рёбрам пути: если road_times —
    тензор с градиентом, градиент правдоподобия дойдёт до времени каждой дороги."""
    rt = road_times.detach() if isinstance(road_times, torch.Tensor) else road_times
    w = [float(x) for x in rt]
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
            if d + w[e] < dist.get(v, float("inf")):
                dist[v], prev[v] = d + w[e], (u, e)
                heapq.heappush(heap, (d + w[e], v))
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
    packaging_time: float = 0.5

    max_burgers_at_time: int = 6
    count_assemblers: int = 2
    count_packers: int = 1

    # вероятность, что в чеке 1, 2, 3, 4 позиции (нормируется, сумма не обязана быть 1)
    probability_count_wishes_in_receipt: list[float] = Field(
        default_factory=lambda: [0.4, 0.3, 0.2, 0.1], min_length=MAX_ITEMS, max_length=MAX_ITEMS)

    # доставка
    count_deliveries: int = 5
    handover_time: float = 1.0           # минут на то, чтобы отдать заказ
    road_times: list[float] = Field(     # минут на каждую дорогу (порядок как в EDGES)
        default_factory=lambda: list(ROAD_MINUTES), min_length=len(EDGES), max_length=len(EDGES))
    address_probs: list[float] = Field(  # вероятности адресов (порядок как в ADDRESSES)
        default_factory=lambda: list(ADDRESS_PROBS), min_length=len(ADDRESSES), max_length=len(ADDRESSES))

    seed: int = 1
    days: int = 30


class Response(BaseModel):
    times: list[float]            # время на кухне: от прихода до упаковки, мин
    items: list[int]              # сколько позиций было в каждом заказе
    average: float                # среднее время на кухне, мин
    average_by_items: dict[int, float]   # среднее время на кухне по числу позиций, мин

    delivery_times: list[float]   # от прихода заказа до передачи клиенту, мин
    delivery_average: float
    courier_wait_average: float   # сколько упакованный заказ ждал свободного доставщика, мин


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
    done: float = 0.0             # заказ упакован — готов к выдаче доставщику
    assembly_waits: list = field(default_factory=list)   # ожидание сборки по каждому бургеру
    # доставка
    address: int = -1             # вершина графа
    courier_left: float = 0.0     # доставщик взял заказ и выехал
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


class Cooking(Process):
    """
    В чеке 1–4 позиции (бургера). Каждый бургер жарится и собирается отдельно,
    упаковывается заказ целиком, когда собраны все его бургеры:

        монитор (бургеры)  --(место на гриле)-->  жарка  --> очередь на сборку
        очередь на сборку  --(свободный сборщик)-->  сборка бургера
        все бургеры заказа собраны  --> очередь на упаковку
        очередь на упаковку  --(свободный упаковщик)-->  упаковка заказа  --> ready_orders

    Выход кухни — очередь ready_orders (номера упакованных заказов). Кто её разбирает, кухня
    не знает; одна кухня (Cooking.simulate) просто копит в ней готовые заказы.

    Число позиций в чеке выбирается при приходе заказа через self.choose с вероятностями
    probability_count_wishes_in_receipt; выбор пишется в журнал и входит в правдоподобие.

    Сущности в очередях и таймерах: бургер — (номер заказа, номер позиции), заказ — номер.
    Время событий внутри эффектов недоступно, моменты этапов восстанавливает collect_orders.
    """
    clients_per_hour = mg.param(25.0)

    frying_time = mg.param(4 * mg.MINUTE)
    burger_assembly_time = mg.param(1 * mg.MINUTE)
    packaging_time = mg.param(0.5 * mg.MINUTE)

    # ограниченные ресурсы
    max_burgers_at_time = mg.param(6)
    count_assemblers = mg.param(2)
    count_packers = mg.param(1)

    # вероятность, что в чеке 1, 2, 3, 4 позиции
    probability_count_wishes_in_receipt = mg.param([0.4, 0.3, 0.2, 0.1])

    # timer ждёт распределение (с .sample), а не число; разброс почти нулевой — время фиксированное
    frying = timer(lambda m: LogNormal(m.frying_time, 0.1 * mg.SECOND), fires="fried")
    building = timer(lambda m: LogNormal(m.burger_assembly_time, 0.1 * mg.SECOND), fires="assembled")
    packing = timer(lambda m: LogNormal(m.packaging_time, 0.1 * mg.SECOND), fires="packed")

    def init(self):
        # очереди ожидания
        self._monitor = mg.Queue()          # бургеры, ждущие гриля
        self._assembly_queue = mg.Queue()   # бургеры, ждущие сборщика
        self._packing_queue = mg.Queue()    # заказы, ждущие упаковщика
        # кто сейчас в работе
        self._frying_slots = mg.Queue()
        self._assemblers_busy = mg.Queue()
        self._packers_busy = mg.Queue()
        # сколько бургеров заказа ещё не собрано
        self._left_to_assemble = {}
        # выход кухни: упакованные заказы
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
            self._packing_queue.push(order_id)

    # --- упаковка (заказ целиком)
    @when(lambda m: len(m._packing_queue) > 0 and len(m._packers_busy) < m.count_packers)
    def start_packing(self):
        order_id = self._packing_queue.pop()
        self._packers_busy.push(order_id)
        self.start("packing", order_id)

    def packed(self, order_id):
        self._packers_busy.remove(order_id)     # упаковщик освободился
        self._done += 1
        self.ready_orders.push(order_id)        # готово — дальше не дело кухни


class Delivery(Process):
    """
    Доставщики разбирают очередь ready_orders (её создаёт и наполняет кухня):

        есть готовый заказ и свободный доставщик  -->  доставщик берёт заказ, перестаёт быть свободным
        через время пути по графу дорог          -->  на адресе, начинает отдавать заказ
        через handover_time                       -->  заказ отдан, едет обратно
        через время пути обратно                  -->  вернулся на точку, снова свободен

    Адрес выбирается, когда доставщик берёт заказ (choose с вероятностями address_probs).
    Сущность доставки — (номер заказа, адрес); время в пути зависит от адреса — таймер
    с распределением от сущности, key — правдоподобие группирует поездки по адресу.
    """
    count_deliveries = mg.param(10)
    handover_time = mg.param(1 * mg.MINUTE)
    ride_spread = mg.param(0.15)                                   # разброс времени в пути, доля
    road_times = mg.param([t * mg.MINUTE for t in ROAD_MINUTES])  # время по каждой дороге (EDGES)
    address_probs = mg.param(ADDRESS_PROBS)                        # вероятности адресов (ADDRESSES)

    to_client = timer(lambda m, d: Delivery._ride(m, d), fires="arrived_at_client", key=lambda d: d[1])
    handing = timer(lambda m: LogNormal(m.handover_time, 0.1 * mg.SECOND), fires="handed_over")
    to_point = timer(lambda m, d: Delivery._ride(m, d), fires="returned", key=lambda d: d[1])

    @staticmethod
    def _ride(m, d):
        t = travel_time(m.road_times, POINT, d[1])   # дороги двусторонние: туда и обратно одинаково
        return LogNormal(t, m.ride_spread * t)

    def init(self):
        self._couriers_busy = mg.Queue()   # доставки, с которыми доставщик сейчас не на точке
        self._delivered = 0

    @when(lambda m: len(m.ready_orders) > 0 and len(m._couriers_busy) < m.count_deliveries)
    def courier_left(self):
        order_id = self.ready_orders.pop()
        address = self.choose(lambda m: m.address_probs, values=ADDRESSES)
        d = (order_id, address)
        self._couriers_busy.push(d)        # доставщик больше не свободен
        self.start("to_client", d)

    def arrived_at_client(self, d):
        self.start("handing", d)           # отдаёт заказ

    def handed_over(self, d):
        self._delivered += 1
        self.start("to_point", d)          # едет обратно

    def returned(self, d):
        self._couriers_busy.remove(d)      # вернулся на точку — снова свободен


class Restaurant(Cooking, Delivery):
    """Кухня + доставка, связаны очередью ready_orders."""

    def init(self):
        Cooking.init(self)      # первой — кухня: она создаёт ready_orders
        Delivery.init(self)


def collect_orders(log):
    """Восстановить моменты этапов каждого заказа по журналу [(t, событие, сущность, вид)].

    Решения when пишутся в журнал без сущности, поэтому очереди проигрываются здесь же:
    все очереди FIFO, значит взятый из очереди элемент — тот, что стоит в ней первым.
    Запись choose идёт сразу после события, где сделан выбор: после arrived_new_order —
    число позиций, после courier_left — адрес.
    """
    orders = {}
    fried_at = {}
    left = {}
    monitor, assembly_q, packing_q, ready_q = deque(), deque(), deque(), deque()
    pending = None          # к чему относится следующая запись choose: ("items" | "address", номер заказа)
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
            orders[ent].done = t
            ready_q.append(ent)
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


def cooking_simulate(controls=None, seed=1, plot=True, model=None):
    model = model or Restaurant
    log, state = model.simulate(controls=controls, horizon=30 * mg.DAY, seed=seed)
    orders = collect_orders(log)
    print(f"{model.__name__}, параметры: {controls or 'по умолчанию'}")
    print(f"  заказов пришло: {state['_orders']}, готово: {state['_done']}")

    def avg(xs):
        return sum(xs) / len(xs) / mg.MINUTE

    total = [o.total for o in orders]
    print(f"  среднее время на кухне: {avg(total):.1f} мин (макс {max(total) / mg.MINUTE:.1f})")
    print(f"  ожидание: жарки {avg([o.wait_frying for o in orders]):.1f}, "
          f"сборки {avg([o.wait_assembly for o in orders]):.1f}, "
          f"упаковки {avg([o.wait_packing for o in orders]):.1f} мин")
    for k in range(1, MAX_ITEMS + 1):
        ok = [o.total for o in orders if o.n_items == k]
        if ok:
            print(f"  {k} поз.: {len(ok)} заказов ({len(ok) / len(orders):.0%}), "
                  f"в среднем {avg(ok):.1f} мин")

    delivered = [o for o in orders if o.delivered > 0]
    if delivered:
        print(f"  доставлено: {state['_delivered']}; от заказа до клиента "
              f"{avg([o.total_delivery for o in delivered]):.1f} мин, "
              f"ожидание доставщика {avg([o.wait_courier for o in delivered]):.1f} мин")
        road = model.params["road_times"].default
        for a in ADDRESSES:
            xs = [o.total_delivery for o in delivered if o.address == a]
            if xs:
                one_way = float(travel_time(road, POINT, a)) / mg.MINUTE
                print(f"    {NODES[a]:<12} {len(xs):>5} заказов, {avg(xs):5.1f} мин "
                      f"(дорога в одну сторону {one_way:.0f} мин)")

    if plot:
        plt.hist([x / mg.MINUTE for x in total], bins=40)
        plt.xlabel("время от заказа до упаковки, мин")
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
        "road_times": [t * mg.MINUTE for t in r.road_times],
        "address_probs": r.address_probs,
    }, horizon=r.days * mg.DAY, seed=r.seed)

    orders = collect_orders(log)
    durations = [o.total / mg.MINUTE for o in orders]
    items = [o.n_items for o in orders]

    by_items = {}
    for k in range(1, MAX_ITEMS + 1):
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
        delivery_times=delivery,
        delivery_average=sum(delivery) / len(delivery) if delivery else 0.0,
        courier_wait_average=sum(waits) / len(waits) if waits else 0.0,
    )


@app.get("/")
async def cooking_ui():
    return FileResponse("/root/workdir/bm_analyser/models/bar/cooking.html")


@app.get("/full")
async def full_ui():
    return FileResponse("/root/workdir/bm_analyser/models/bar/full.html")


if __name__ == "__main__":
    cooking_simulate()
    cooking_simulate({"count_deliveries": 7}, plot=False)
