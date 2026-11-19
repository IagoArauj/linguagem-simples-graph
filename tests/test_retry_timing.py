from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from core.providers.base import SimulatedModelClient, TransientError
from core.timing import RetryTiming, track_document_retries


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_overlapping_backoffs_count_wall_time_only():
    clock = Clock()
    timing = RetryTiming(clock)
    with timing.invocation(), timing.invocation():
        with timing.backoff():
            clock.advance(10)  # Outro ramo ainda está trabalhando.
            with timing.backoff():
                clock.advance(60)
            clock.advance(20)  # Outro ramo voltou a trabalhar.
    assert timing.retry_wait_seconds == 60


def test_new_active_call_interrupts_discount():
    clock = Clock()
    timing = RetryTiming(clock)
    with timing.invocation():
        with timing.backoff():
            clock.advance(5)
            with timing.invocation():
                clock.advance(30)
            clock.advance(7)
    assert timing.retry_wait_seconds == 12
    clock.advance(100)
    assert timing.retry_wait_seconds == 12


@pytest.mark.parametrize("fails", [False, True])
def test_retries_measure_actual_sleep_and_cleanup_after_failure(fails):
    clock = Clock()
    timing = RetryTiming(clock)
    script = [TransientError("429"), TransientError("429")]
    script.append(TransientError("429") if fails else "ok")
    client = SimulatedModelClient(
        script=script,
        technical_retries=2,
        retry_backoff_seconds=60,
        sleep=lambda delay: clock.advance(delay + 0.25),
    )
    client.retry_timing = timing
    if fails:
        with pytest.raises(TransientError):
            client.invoke([])
    else:
        assert client.invoke([]).content == "ok"
    assert timing.retry_wait_seconds == 180.5
    clock.advance(50)
    assert timing.retry_wait_seconds == 180.5


def test_no_backoff_means_zero():
    client = SimulatedModelClient(script=["ok"])
    with track_document_retries([client, client]) as timing:
        assert client.invoke([]).content == "ok"
    assert timing.retry_wait_seconds == 0
    assert client.retry_timing is None


def test_per_document_reset_and_restore_on_exception():
    client = SimulatedModelClient()
    previous = RetryTiming()
    client.retry_timing = previous
    with pytest.raises(RuntimeError):
        with track_document_retries([client]) as first:
            assert client.retry_timing is first
            raise RuntimeError("aborted")
    assert client.retry_timing is previous
    with track_document_retries([client]) as second:
        assert first is not second
        assert second.retry_wait_seconds == 0
    assert client.retry_timing is previous


def test_wait_overlapping_request_in_other_thread_is_not_discounted():
    clock = Clock()
    timing = RetryTiming(clock)
    sleeping = Event()
    release = Event()

    def sleeper():
        with timing.invocation(), timing.backoff():
            sleeping.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with timing.invocation():
            future = executor.submit(sleeper)
            try:
                assert sleeping.wait(5)
                clock.advance(60)
            finally:
                release.set()
            future.result(timeout=5)
    assert timing.retry_wait_seconds == 0


def test_interrupted_sleep_does_not_leave_pending_wait():
    clock = Clock()
    timing = RetryTiming(clock)

    def interrupted_sleep(delay):
        clock.advance(3)
        raise RuntimeError("interrupted")

    client = SimulatedModelClient(
        script=[TransientError("429")], technical_retries=1,
        retry_backoff_seconds=60, sleep=interrupted_sleep,
    )
    client.retry_timing = timing
    with pytest.raises(RuntimeError):
        client.invoke([])
    assert timing.retry_wait_seconds == 3
    clock.advance(100)
    assert timing.retry_wait_seconds == 3
