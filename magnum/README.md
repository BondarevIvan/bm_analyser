# Magnum

Библиотека на Python для моделирования бизнеса как случайного потока событий: приходы клиентов, очереди,
смены персонала, чеки, деньги. Модель запускается на много сидов — получается распределение прибыли и других
показателей, по которому можно сравнивать и подбирать параметры (график, цены, число сотрудников).
Параметры модели — тензоры torch, поэтому их можно обучать градиентными методами (см. «Обучение» в документации).

## Установка

    pip install -e magnum          # из папки, где лежит каталог magnum/ (зависимость — torch)

## Пример

```python
import magnum as mg

class Coffee(mg.Model):
    masters = mg.param(4, low=0)                # параметр модели: по умолчанию и границы
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
print(r["final"]["served"])

runs = mg.run_many(Coffee, seeds=range(1000), processes=8, horizon=30, debug=False)
```

Время модели — дни; единицы `mg.SECOND`, `mg.MINUTE`, `mg.HOUR`, `mg.DAY`, `mg.WEEK`.

## Документация

* [docs/python_api.md](docs/python_api.md) — параметры, события и триггеры, состояние, случайные величины,
  запуск и запись переменных, подмодели (`mg.sub`), обучение параметров, тесты моделей.
* `examples/summator.py` — минимальный пример.

## Тесты

    cd magnum && python -m pytest -q
