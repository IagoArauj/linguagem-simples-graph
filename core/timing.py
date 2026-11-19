"""Medição compartilhada de backoff sem chamadas de modelo ativas."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from threading import Lock


class RetryTiming:
    """Conta tempo de parede, não a soma das esperas de threads paralelas."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._lock = Lock()
        self._active = 0
        self._waiting = 0
        self._last = clock()
        self._wait_seconds = 0.0

    def _update(self, *, active: int = 0, waiting: int = 0) -> None:
        with self._lock:
            now = self._clock()
            if self._waiting and not self._active:
                self._wait_seconds += now - self._last
            self._last = now
            self._active += active
            self._waiting += waiting

    @property
    def retry_wait_seconds(self) -> float:
        self._update()
        with self._lock:
            return self._wait_seconds

    @contextmanager
    def invocation(self) -> Iterator[None]:
        self._update(active=1)
        try:
            yield
        finally:
            self._update(active=-1)

    @contextmanager
    def backoff(self) -> Iterator[None]:
        # Transição atômica: uma chamada em espera deixa de estar ativa.
        self._update(active=-1, waiting=1)
        try:
            yield
        finally:
            self._update(active=1, waiting=-1)


@contextmanager
def track_document_retries(clients: Iterable[object]) -> Iterator[RetryTiming]:
    """Instrumenta clientes de um documento, inclusive nas threads do grafo.

    Os clientes não devem ser compartilhados por documentos simultâneos.
    O executor CLI processa documentos sequencialmente, mas os ramos usam o
    mesmo contador protegido por lock. A referência anterior é sempre restaurada.
    """
    from core.providers.base import RetryingModelClient

    timing = RetryTiming()
    previous = []
    seen: set[int] = set()
    try:
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            if not isinstance(client, RetryingModelClient):
                raise TypeError("Contabilização de backoff exige RetryingModelClient.")
            previous.append((client, client.retry_timing))
            client.retry_timing = timing
        yield timing
    finally:
        for client, old_timing in previous:
            client.retry_timing = old_timing
