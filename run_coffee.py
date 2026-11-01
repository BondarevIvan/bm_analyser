import magnum.magnum as mgn


def compute_hours(time: float):
    seconds = 86400 * (time - int(time))
    return int(seconds // 3600)


def pretty_time(time: float):
    seconds = 86400 * (time - int(time))
    hours = int(seconds // 3600)
    minute = int((seconds // 60) % 60)
    int_seconds = int((seconds // 1) % 60)
    return f"День {int(time):03d}. {hours:02d}:{minute:02d}:{int_seconds:02d}"


class CoffeeQueue(mgn.Model):

    average_wait_time = mgn.param(mgn.minutes(3))
    std_wait_time = mgn.param(mgn.seconds(30))
    count_masters = mgn.param(1)

    average_order = mgn.param(mgn.minutes(2), low=0, high=mgn.minutes(10))
    std_order = mgn.param(mgn.seconds(20), low=0, high=mgn.minutes(5))

    clients_per_hour = mgn.param(15)
    time_to_make_order = mgn.param(mgn.seconds(10))

    @mgn.on_start
    def init(self):
        self.queue = mgn.Queue()
        self.free_masters = self.count_masters
        self._id_client = 0

    def free_one_master(self):
        self.free_masters += 1

    def client_choose(self, id_client: int):
        self.print("Клиент", id_client, "сделал заказ", f"({pretty_time(self.time)})")
        self.free_masters -= 1
        self.emit(self.free_one_master, after=max(0.0, self.rng.normal(self.average_order, self.std_order)))

    @mgn.when(lambda model: len(model.queue) > 0 and model.free_masters > 0)
    def start_new_order(self):
        id_client = self.queue.pop()
        self.print("Клиент", id_client, "делает заказ", f"({pretty_time(self.time)})")
        self.emit(self.client_choose, id_client, after=self.time_to_make_order)

    def client_leave(self, id_client: int):
        if self.queue.remove(id_client):
            self.print("Клиент не дождался", id_client, f"({pretty_time(self.time)})")

    @mgn.arrival(rate="clients_per_hour", per=mgn.HOUR)
    def arrive_client(self):
        # 08:00-21:00
        if not (20 >= compute_hours(self.time) >= 8):
            return
        self._id_client += 1
        self.print("Клиент", self._id_client, "пришел", f"({pretty_time(self.time)})")
        self.queue.push(self._id_client)
        wait = max(0.0, self.rng.normal(self.average_wait_time, self.std_wait_time))
        self.emit(self.client_leave, self._id_client, after=wait)


if __name__ == "__main__":
    result = CoffeeQueue.run(horizon=30, debug=True, sampling="on_change")
