# Magnum — Python API

Модель — класс-наследник `mg.Model`. Параметры объявляются через `mg.param(...)`, события — это методы, триггеры —
декораторы.

```python
import magnum as mg

class Coffee(mg.Model):
    masters = mg.param(4, low=0)               # параметр: по умолчанию 4, границы
    clients_per_hour = mg.param(15.0, low=0)

    @mg.on_start
    def init(self):
        self.clients = mg.Queue()
        self.idle = self.masters
        self.served = 0

    @mg.arrival(rate="clients_per_hour", per=mg.HOUR)
    def visit(self):
        self.clients.push(self.time)

    @mg.when(lambda m: m.idle > 0 and len(m.clients) > 0)
    def start_order(self):
        self.clients.pop()
        self.idle -= 1
        self.emit(self.order_done, after=self.rng.exponential(7 * mg.MINUTE))

    def order_done(self):
        self.idle += 1
        self.served += 1

r = Coffee.run(controls={"masters": 3}, seed=1, horizon=30)
r["final"]["served"]
```

## Время

Время модели — дни (`self.time`). Единицы: `mg.SECOND`, `mg.MINUTE`, `mg.HOUR`, `mg.DAY`, `mg.WEEK`
(`5 * mg.MINUTE`) и функции `mg.minutes(5)`, `mg.hours(2)`…
Помощники: `mg.hour_of_day(t)` → 0..23, `mg.weekday(t, start=0)` → 0..6 (0 — понедельник), `mg.day_index(t)`.

## Параметры

Каждый параметр — тензор torch: число (0-мерный тензор) или массив.

```python
price = mg.param(1.0, low=0, high=3, low_open=True)   # (0, 3], float64
staff = mg.param(4, low=1, high=10)                    # целый default → int64, только целые, не обучается
weights = mg.param(mg.array(7, 20, value=1.0), low=0)  # тензор 7×20, границы поэлементно
```

В событиях `self.price`, `self.weights` — тензоры: `self.weights[day]`, `self.weights.sum(1)`, `self.price * n`.
Перекрываются при запуске: `Model.run(controls={"price": 1.2})` — число, список, numpy-массив или тензор.
Неизвестное имя, неверная форма или выход за границы — ошибка `MagnumControlError`.
`Model.params` — словарь объявлений (`default`, `low`, `high`, `integer`, `shape`).

* Тензор с `requires_grad`, переданный в controls, попадает в модель **без копии** — градиенты дойдут до него.
* Присваивание в состояние: 0-мерный тензор без градиента становится обычным числом
  (`self.idle = self.masters` → `int`), остальные тензоры копируются — `self.idle -= 1` не испортит параметр.
* Значения в `final` и в сериях `variables` — обычные числа и списки; тензоры с градиентом остаются в `final`
  тензорами (их можно дифференцировать).

## События и триггеры

| | |
|---|---|
| `@mg.on_start` | в момент 0, до остальных событий |
| `@mg.every(mg.HOUR)` / `@mg.every("period")` | периодически: в моменты P, 2P, … |
| `@mg.arrival(rate=..., per=mg.HOUR)` | пуассоновский поток: в среднем rate событий за per |
| `@mg.when(lambda m: ...)` | пока условие истинно |
| метод без декоратора | событие без триггера — его планируют через `emit` |
| `self.emit(self.finish, n, after=3 * mg.MINUTE)` | запланировать событие с аргументами |
| `self.choose([(0.3, self.a), (0.7, None)])` | случайный выбор ветки |
| `self.print(...)` | отладочный вывод (выключается `debug=False`) |

* `rate`, `period` — число, имя атрибута модели (строка) или функция `m -> число`; пересчитываются при каждом
  срабатывании, так что интенсивность может меняться по ходу (rate 0 — поток останавливается).
* `when` — условие проверяется только после событий, поменявших переменные, которые оно читает
  (включая элементы списков и очередей). Время и `rng` в условии читать нельзя.
* Метод можно вызвать и напрямую (`self.arrive()`) — тогда это обычный вызов внутри текущего события, не отдельное событие.
* Ошибка внутри события оборачивается в `MagnumRuntimeError` с именем события и моментом времени.

## Состояние

Всё, что записано в `self.<имя>`, — переменные модели: попадают в `r["final"]` и в серии `r["variables"]`.
Атрибуты с `_` в начале — служебные, не записываются. Структуры: `mg.Queue()` (`push`, `pop`, `first`, `remove`,
`in`, `len`), `mg.Resource(capacity)` (`seize`, `release`, `free`), обычные списки (`mg.array(...)`).

Случайность — только через `self.rng`: `random()`, `uniform(a, b)`, `normal(mu, sigma)`, `exponential(mean)`,
`bernoulli(p)`, `poisson(lam)`, `categorical(weights)`, `choice(items, weights=None)`.
Один и тот же `seed` даёт один и тот же результат.

## Запуск

```python
r = Model.run(controls=None, seed=0, horizon=365, sampling="1d", record=None, max_events=10_000_000, debug=True)
```

* `sampling` — `"1d"`, `"6h"`, число (дни) или `"on_change"`; `record=["profit", "served"]` — какие переменные писать.
* `debug=False` — отключить `print`.
* результат: `model, seed, horizon, sampling, controls, variables{имя: {time, value}}, final, event_counts`.
* `mg.run_many(Model, seeds=range(100), processes=8, horizon=28)` — серия запусков, параллельно по процессам
  (класс модели должен лежать в импортируемом модуле, не в `__main__` Jupyter).

## Подмодели

Модель можно включить в другую модель — в общем времени, с общим `rng`:

```python
class Network(mg.Model):
    center = mg.sub(CoffeeQueue, count_masters=4)   # значения по умолчанию для параметров подмодели
    mall = mg.sub(CoffeeQueue)

    @mg.every(mg.DAY)
    def report(self):
        self.total = self.center.count_served + self.mall.count_served

    @mg.when(lambda m: len(m.mall.clients) > 10)    # when на переменные подмодели
    def redirect(self):
        self.mall.clients.pop()
        self.emit(self.center.register_client)        # событие подмодели

Network.run(controls={"mall.count_masters": 3})
```

* Параметры подмоделей — параметры внешней модели под именами `mall.count_masters` (один набор controls
  на всю программу, оптимизатор видит все). `Network.params` содержит их все.
* Переменные и события подмодели в результате — `final["mall.count_served"]`, `event_counts["mall.start_order"]`,
  `record=["mall.clients"]`.
* `on_start` подмоделей выполняются раньше `on_start` внешней модели; условия `when` проверяются после всех `on_start`.
* `self.emit("name")` строкой — событие той модели, из которой вызван emit; `self.emit("mall.name")` — подмодели.
* Методы подмодели можно вызывать напрямую: `items = self.checks.issue(day)`.
* Подмодели могут быть вложенными (`north.center.count_masters`); саму подмодель по-прежнему можно запускать отдельно.

Пример: `demo_bakery/bakery_checks.py` — пекарня, где генератор чеков подключён подмоделью.

## Обучение

Параметры можно обучать градиентными методами torch.

```python
params = Model.make_params(names=["price"])          # тензоры; price — с requires_grad
opt = torch.optim.Adam(mg.trainable(params), lr=0.01)
...
Model.clamp_(params)                                 # после opt.step(): вернуть в границы
```

**Правдоподобие данных.** Когда модель описывает, как получаются наблюдения (например, состав чека), логарифм
вероятности реальных данных пишется формулой от параметров-тензоров, и его максимизируют напрямую —
пример `demo_bakery/fit_checks.py` (веса генератора чеков по 6,6 тыс. реальных чеков, несколько секунд).

**Градиент ожидания показателя.** `result["log_prob"]` — логарифм вероятности выпавшей траектории по обучаемым
параметрам: в него входят `rng.*`, `choose` и потоки `arrival`, распределение которых зависит от
тензора с `requires_grad`. `mg.expected(runs, "profit")` собирает из серии запусков оценку E[profit],
градиент которой — несмещённая оценка ∇E[profit] (по траектории + score function):

```python
runs = [Model.run(controls=params, seed=s, horizon=28, debug=False, record=[]) for s in range(64)]
loss = -mg.expected(runs, lambda r: r["final"]["profit"])
loss.backward()
```

Оценка несмещённая, но для длинных траекторий с тысячами случайных исходов дисперсия большая (на пекарне
за 4 недели — в разы больше самого градиента при 16 запусках). Для таких показателей пока надёжнее перебор
или конечные разности на общих сидах; score-оценка хороша для коротких траекторий и правдоподобия.

Исходы по элементам одной таблицы (`self.rng.bernoulli(self._p[d, h])`, `self.rng.categorical(self.w[k])`)
копятся счётчиками, поэтому граф и обратный проход не растут с числом событий; таблицу выгодно посчитать один раз
в `@mg.on_start`.

## Тесты моделей

Обычный pytest: запустить модель с нужными `controls` и проверить свойства результата —
см. `models/coffee/tests/test_coffee_queue.py`.
