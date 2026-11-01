"""Структуры состояния модели: очередь (Queue) и ресурс (Resource).

В результат моделирования (variables / final) попадают в «плоском» виде: очередь — список элементов
от головы к хвосту, ресурс — число занятых единиц.

Изменения очереди и ресурса видны движку (триггер when, запись sampling="on_change"): после каждой
изменяющей операции вызывается ``_notify``, который движок назначает, когда объект кладут в состояние модели.
"""
from __future__ import annotations

from collections import deque

import torch


class Queue:
    """FIFO-очередь.

    push / pop / remove / contains / len — за O(1) (remove — ленивое удаление: элемент помечается
    и выбрасывается, когда дойдёт до головы). Если одинаковых значений несколько, remove убирает самое раннее.
    Элементы должны быть хешируемыми (числа, строки, кортежи).
    """

    __slots__ = ("_items", "_alive", "_removed", "_size", "_notify")

    def __init__(self, items=()):
        self._items: deque = deque()
        self._alive: dict = {}     # значение -> сколько таких в очереди
        self._removed: dict = {}   # значение -> сколько ранних вхождений помечены удалёнными
        self._size = 0
        self._notify = None
        for x in items:
            self._push(x)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, x):
        return self.contains(x)

    def __iter__(self):
        return iter(self.snapshot())

    def _changed(self):
        if self._notify is not None:
            self._notify()

    def _push(self, x):
        self._items.append(x)
        self._alive[x] = self._alive.get(x, 0) + 1
        self._size += 1

    def push(self, x) -> None:
        """Добавить в хвост. Тензор-число кладётся как обычное число (тензоры не хешируются по значению)."""
        if isinstance(x, torch.Tensor):
            x = x.item()
        self._push(x)
        self._changed()

    def contains(self, x) -> bool:
        if isinstance(x, torch.Tensor):
            x = x.item()
        return self._alive.get(x, 0) > 0

    def remove(self, x) -> bool:
        """Убрать x (самое раннее вхождение), если он есть. Возвращает, был ли он в очереди."""
        if isinstance(x, torch.Tensor):
            x = x.item()
        n = self._alive.get(x, 0)
        if n == 0:
            return False
        if n == 1:
            del self._alive[x]
        else:
            self._alive[x] = n - 1
        self._removed[x] = self._removed.get(x, 0) + 1
        self._size -= 1
        self._changed()
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
        """Голова очереди без извлечения; IndexError, если пуста."""
        self._clean_head()
        if not self._items:
            raise IndexError("очередь пуста")
        return self._items[0]

    def pop(self):
        """Взять из головы; IndexError, если пуста."""
        x = self.first()
        self._items.popleft()
        n = self._alive[x]
        if n == 1:
            del self._alive[x]
        else:
            self._alive[x] = n - 1
        self._size -= 1
        self._changed()
        return x

    def clear(self) -> None:
        had = self._size > 0
        self._items.clear()
        self._alive.clear()
        self._removed.clear()
        self._size = 0
        if had:
            self._changed()

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

    __slots__ = ("capacity", "busy", "_notify")

    def __init__(self, capacity: int):
        if isinstance(capacity, torch.Tensor):
            capacity = capacity.item()
        if isinstance(capacity, bool) or capacity < 0 or capacity != int(capacity):
            raise ValueError(f"resource: capacity должна быть целым числом >= 0, получено {capacity!r}")
        self.capacity = int(capacity)
        self.busy = 0
        self._notify = None

    @property
    def free(self) -> int:
        return self.capacity - self.busy

    def seize(self) -> None:
        """Занять единицу; ошибка, если свободных нет."""
        if self.busy >= self.capacity:
            raise RuntimeError(f"resource: нет свободных единиц (занято {self.busy} из {self.capacity})")
        self.busy += 1
        if self._notify is not None:
            self._notify()

    def release(self) -> None:
        """Освободить единицу; ошибка, если ни одна не занята."""
        if self.busy <= 0:
            raise RuntimeError("resource: ни одна единица не занята")
        self.busy -= 1
        if self._notify is not None:
            self._notify()

    def snapshot(self) -> float:
        return float(self.busy)

    def __repr__(self):
        return f"resource(busy={self.busy}, capacity={self.capacity})"


STATEFUL = (Queue, Resource)


def plain(v):
    """Значение для результата/снимка: списки копируются, очередь и ресурс разворачиваются."""
    t = type(v)
    if t is float or t is int:
        return v
    if isinstance(v, torch.Tensor):
        return v.detach().tolist()
    if isinstance(v, list):
        return [plain(x) for x in v]
    if t is Queue or t is Resource:
        return v.snapshot()
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return tuple(plain(x) for x in v)
    return v
