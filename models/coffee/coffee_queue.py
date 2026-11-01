"""Кофейня с очередью.

Клиенты приходят пуассоновским потоком в часы работы, ждут в очереди не дольше своего терпения,
свободный бариста (idle_masters — обычная переменная) берёт следующего. Время — дни.

    from models.coffee.coffee_queue import CoffeeQueue
    r = CoffeeQueue.run(seed=1, horizon=30, debug=False)
    r["final"]["count_served"], r["final"]["count_lost"]
"""
import magnum as mg


class CoffeeQueue(mg.Model):
    name = "Кофейня"

    open_hour = mg.param(8, low=0, high=24)
    close_hour = mg.param(22, low=0, high=24)
    average_duration_order = mg.param(420.0, low=0)              # с, приготовление заказа
    average_count_clients_per_1h = mg.param(15.0, low=0)         # клиентов в час
    average_wait = mg.param(330.0, low=0)                        # с, терпение клиента
    std_wait = mg.param(30.0, low=0)
    count_masters = mg.param(4, low=0)

    # ---------------------------------------------------------------- вспомогательное
    def is_open(self):
        hour = (self.time - mg.day_index(self.time)) * 24
        return self.open_hour <= hour <= self.close_hour

    def time_to_open_as_days(self, tolerance=0.0001):
        """Сколько дней до ближайшего открытия (0, если открыто)."""
        frac = self.time - mg.day_index(self.time)
        hour = frac * 24
        if hour >= self.close_hour:
            return 1 - frac + self.open_hour / 24 + tolerance
        if hour <= self.open_hour:
            return self.open_hour / 24 - frac + tolerance
        return 0.0

    # ---------------------------------------------------------------- события
    @mg.on_start
    def init(self):
        self.count_served = 0
        self.count_processed = 0
        self.count_lost = 0
        self.last_client_id = 0
        self.clients = mg.Queue()
        self.idle_masters = self.count_masters
        self.emit(self.visit_client_generator)

    def visit_client_generator(self):
        delay = self.rng.exponential(1 / self.average_count_clients_per_1h) * mg.HOUR
        if self.is_open():
            self.register_client()
            self.emit(self.visit_client_generator, after=delay)
        else:
            self.emit(self.visit_client_generator, after=self.time_to_open_as_days() * mg.DAY + delay)

    def register_client(self):
        self.last_client_id += 1
        self.clients.push(self.last_client_id)
        wait = max(0.0, self.rng.normal(self.average_wait, self.std_wait))
        self.emit(self.client_leave, self.last_client_id, after=wait * mg.SECOND)

    def client_leave(self, client_id):
        if self.clients.remove(client_id):
            self.count_lost += 1

    @mg.when(lambda m: m.idle_masters > 0 and len(m.clients) > 0)
    def start_order(self):
        self.clients.pop()
        self.idle_masters -= 1
        self.count_served += 1
        duration = max(0.0, self.rng.normal(self.average_duration_order, 10))   # max — защита от after < 0
        self.emit(self.order_done, after=duration * mg.SECOND)

    def order_done(self):
        self.idle_masters += 1
        self.count_processed += 1


if __name__ == "__main__":
    r = CoffeeQueue.run(seed=1, horizon=30, debug=False)
    f = r["final"]
    print(f"за 30 дней: пришло {f['last_client_id']}, обслужено {f['count_served']}, ушли {f['count_lost']}")
