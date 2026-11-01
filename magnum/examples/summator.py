"""Простой пример: каждый день к сумме прибавляется value; с вероятностью p_reset value обнуляется.

    python examples/summator.py
"""
import magnum as mg


class Summator(mg.Model):
    value = mg.param(1.0, low=-10, high=100)
    p_reset = mg.param(0.001, low=0, high=1)

    @mg.on_start
    def init(self):
        self.summa = 0.0

    @mg.every(mg.DAY)
    def accumulate(self):
        self.summa += self.value
        self.choose([(self.p_reset, lambda: self.emit(self.set_value, 0.0))])

    def set_value(self, new_value):
        self.value = new_value


if __name__ == "__main__":
    r = Summator.run(seed=1, horizon=365)
    print("сумма за год:", r["final"]["summa"], "| value в конце:", r["final"]["value"])
