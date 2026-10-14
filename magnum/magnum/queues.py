"""Встроенные структуры состояния: очередь (queue) и ресурс (resource).

Обе хранятся среди глобальных переменных модели под своим именем. В результат (variables / final) попадают
в «плоском» виде: очередь — список элементов от головы к хвосту, ресурс — число занятых единиц.
"""
from __future__ import annotations

from collections import deque


class Queue:
    """FIFO-очередь чисел.

    push / pop / remove / contains / size — за O(1) (remove — ленивое удаление: элемент помечается
    и выбрасывается, когда дойдёт до головы). Если одинаковых значений несколько, remove убирает самое раннее.
    """

    __slots__ = ("_items", "_alive", "_removed", "_size")

    def __init__(self):
        self._items: deque = deque()
        self._alive: dict = {}     # значение -> сколько таких в очереди
        self._removed: dict = {}   # значение -> сколько ранних вхождений помечены удалёнными
        self._size = 0

    def __len__(self):
        return self._size

    def push(self, x) -> None:
        self._items.append(x)
        self._alive[x] = self._alive.get(x, 0) + 1
        self._size += 1

    def contains(self, x) -> bool:
        return self._alive.get(x, 0) > 0

    def remove(self, x) -> bool:
        n = self._alive.get(x, 0)
        if n == 0:
            return False
        if n == 1:
            del self._alive[x]
        else:
            self._alive[x] = n - 1
        self._removed[x] = self._removed.get(x, 0) + 1
        self._size -= 1
        return True

    def _clean_head(self):
        items, removed = self._items, self._removed
        while items:
            x = items[0]
            r = removed.get(x)
            if not r:
                return
            items.popleft()
            if r == 1:
                del removed[x]
            else:
                removed[x] = r - 1

    def first(self):
        self._clean_head()
        if not self._items:
            raise IndexError("очередь пуста")
        return self._items[0]

    def pop(self):
        x = self.first()
        self._items.popleft()
        n = self._alive[x]
        if n == 1:
            del self._alive[x]
        else:
            self._alive[x] = n - 1
        self._size -= 1
        return x

    def clear(self) -> None:
        self._items.clear()
        self._alive.clear()
        self._removed.clear()
        self._size = 0

    def snapshot(self) -> list:
        """Элементы от головы к хвосту (без удалённых)."""
        removed = dict(self._removed)
        out = []
        for x in self._items:
            r = removed.get(x)
            if r:
                removed[x] = r - 1
                continue
            out.append(x)
        return out

    def __repr__(self):
        return f"queue{self.snapshot()}"


class Resource:
    """Ресурс из capacity одинаковых единиц (мастера, кассы, столы)."""

    __slots__ = ("capacity", "busy")

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.busy = 0

    @property
    def free(self) -> int:
        return self.capacity - self.busy

    def snapshot(self) -> float:
        return float(self.busy)

    def __repr__(self):
        return f"resource(busy={self.busy}, capacity={self.capacity})"


STATEFUL = (Queue, Resource)


def plain(v):
    """Значение для результата/снимка: массивы копируются, очередь и ресурс разворачиваются."""
    t = type(v)
    if t is list:
        return [plain(x) for x in v]
    if t is Queue or t is Resource:
        return v.snapshot()
    return v
