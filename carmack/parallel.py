"""Bounded, order-preserving parallel mapping over batches.

The one driver here hands batches to an executor it does not own, keeps a bounded number
of them in flight, and yields their results in submission order. That makes the output of
a parallel stage a function of its input alone, independent of how many workers are
running or of the order in which they happen to finish.
"""

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from itertools import islice

log = logging.getLogger(__name__)


def map_batches_in_order[BatchT, ResultT](
    executor: Executor,
    worker: Callable[[BatchT], ResultT],
    batches: Iterable[BatchT],
    n_workers: int,
) -> Iterator[ResultT]:
    """Apply a worker to each batch in parallel, yielding results in submission order.

    At most ``max(n_workers * 2, 2)`` batches are in flight at once, and the batch source
    is pulled no further ahead than that, so resident memory stays proportional to the
    window rather than to the length of the input however long the caller spends folding
    each result. The replacement batch is submitted before its predecessor's result is
    yielded, so no worker sits idle while the caller does that folding.

    Three contracts bind callers:

    1. The opening window is submitted when this function is called, not when the iterator
       it returns is first advanced. A caller that has to control the moment its pool
       starts its workers -- a process pool launches them on the first submit -- can
       therefore place that moment by placing the call.
    2. Results are yielded in the order their batches were submitted, never in the order
       the work finished. Output therefore does not depend on the worker count or on
       scheduling, so identical input gives identical bytes on every run.
    3. The returned iterator must be consumed inside the block that keeps the executor
       alive. This driver never creates, enters, exits or shuts down the pool, so a caller
       that abandons the iterator early leaves futures still queued that are its own to
       drain when it closes the pool.

    Leaving the pool's lifecycle wholly to the caller is deliberate. A caller writing to
    compressed output streams has to be free to shut its pool down before it closes those
    streams: workers inherit the write end of each compressor's input pipe, so a
    compressor still held open by a live worker never sees end of file and the run stops
    making progress. Only the caller can order those two shutdowns, so only the caller may
    own the pool.

    A worker exception is not caught. It surfaces in place of the failing batch's result,
    once every earlier batch has been yielded, so even the partial output written before a
    failure is the same on every run.

    Args:
        executor: Live executor to submit work to, created and shut down by the caller.
        worker: Callable applied to one batch. Bind any further arguments with
            ``functools.partial`` so it stays a single-argument callable.
        batches: Batches to map over, consumed lazily and exactly once.
        n_workers: Width of the caller's pool, which sizes the in-flight window.

    Returns:
        An iterator over the worker's result for each batch, in the order the batches were
        submitted.
    """

    batch_iter = iter(batches)
    max_in_flight = max(n_workers * 2, 2)
    log.debug(f"Mapping batches in order with max in-flight {max_in_flight}")
    pending: deque[Future[ResultT]] = deque(
        executor.submit(worker, batch) for batch in islice(batch_iter, max_in_flight)
    )

    def drain() -> Iterator[ResultT]:
        while pending:
            result = pending.popleft().result()
            for batch in islice(batch_iter, 1):
                pending.append(executor.submit(worker, batch))
            yield result

    return drain()
