"""Unit tests for the bounded, in-order batch mapping driver.

The driver hands batches to an executor it does not own, keeps a bounded number of them
in flight, and yields their results in submission order rather than completion order.
These tests pin all three of those properties, plus the window policy itself, without
depending on how fast any particular worker happens to run.
"""

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from assertpy import assert_that

import carmack.parallel
from carmack.parallel import map_batches_in_order

# Worker count for the tests that drive a real thread pool. The driver keeps twice this
# many batches in flight, which is also the number of threads the pool is given and the
# number of parties each rendezvous waits for.
REAL_POOL_WORKERS = 2
REAL_POOL_WINDOW = 4

# Whole windows of batches, so every rendezvous gets exactly the number of parties it
# waits for and the run ends with no partial window left over.
REAL_POOL_WINDOW_COUNT = 3
REAL_POOL_BATCHES = REAL_POOL_WINDOW * REAL_POOL_WINDOW_COUNT

# Step between the graded sleeps that order completions within a window. Long enough that
# the grading survives ordinary scheduling jitter, short enough that the whole test costs
# a fraction of a second.
SLEEP_UNIT_S = 0.02

# Cap on how long a batch waits for the rest of its window to arrive. A driver that fills
# the window never comes close to it; one that does not fails here rather than hanging the
# suite, which is the point of putting a timeout on the rendezvous at all.
BARRIER_TIMEOUT_S = 5.0


def expected_window(n_workers: int) -> int:
    """Number of batches the driver is expected to keep in flight.

    Restated here rather than imported so the tests pin the policy instead of agreeing
    with whatever the module under test happens to do.

    Args:
        n_workers: Worker count handed to the driver.

    Returns:
        The maximum number of batches that may be submitted but not yet consumed.
    """

    return max(n_workers * 2, 2)


def double(batch: int) -> int:
    """Map a batch to a result that is distinguishable from the batch itself.

    Args:
        batch: Batch value to transform.

    Returns:
        Twice the batch value.
    """

    return batch * 2


class SubmitLimitExceeded(RuntimeError):
    """Raised when a fake executor is asked to take more submissions than allowed."""


class WorkerFailure(RuntimeError):
    """Raised by a worker that has been told to fail on a particular batch."""


class FailingWorker:
    """A worker that succeeds on every batch except one."""

    def __init__(self, failing_batch: int) -> None:
        """Store the batch to fail on.

        Args:
            failing_batch: Batch value that raises instead of returning a result.
        """

        self.failing_batch = failing_batch

    def __call__(self, batch: int) -> int:
        """Transform a batch, or fail if it is the chosen one.

        Args:
            batch: Batch value to transform.

        Returns:
            Twice the batch value.

        Raises:
            WorkerFailure: If the batch is the one this worker was told to fail on.
        """

        if batch == self.failing_batch:
            raise WorkerFailure(f"batch {batch} failed")
        return double(batch)


class RecordingFuture:
    """A future stand-in that runs its work on submission and records its retrieval.

    The work runs eagerly and its outcome is held until the result is asked for, which is
    how a real future looks from the driver's side: a worker's exception surfaces where
    the result is retrieved, never where the batch was submitted.
    """

    def __init__(
        self,
        index: int,
        worker: Callable[[Any], Any],
        batch: Any,
        executor: "RecordingExecutor",
    ) -> None:
        """Run the worker and hold its outcome.

        Args:
            index: Submission position of this future, counted from zero.
            worker: Callable applied to the batch.
            batch: Batch handed to the worker.
            executor: Executor stand-in to report the eventual retrieval to.
        """

        self.index = index
        self.executor = executor
        self.value: Any = None
        self.error: Exception | None = None
        try:
            self.value = worker(batch)
        except Exception as error:
            self.error = error

    def result(self, timeout: float | None = None) -> Any:
        """Return the outcome of the submitted work, recording the retrieval.

        Args:
            timeout: Accepted for signature compatibility and ignored, because the work
                has already run.

        Returns:
            The value the worker returned.

        Raises:
            Exception: Whatever the worker raised, re-raised in the caller's frame.
        """

        self.executor.record_retrieval(self.index)
        if self.error is not None:
            raise self.error
        return self.value


class RecordingExecutor:
    """An executor stand-in that records everything the driver does to it.

    Submissions run eagerly, so nothing read back here depends on timing: the tests see
    the order batches were submitted in, the order their results were retrieved in, how
    many were outstanding at the busiest moment, and whether the driver ever tried to shut
    the pool down.
    """

    def __init__(self, submit_limit: int | None = None) -> None:
        """Start with nothing submitted.

        Args:
            submit_limit: Number of submissions to accept before raising, or None for no
                limit. A limit turns a driver that resubmits the same batch for ever into
                a failing test rather than a hanging one.
        """

        self.submit_limit = submit_limit
        self.submitted_batches: list[Any] = []
        self.result_order: list[int] = []
        self.outstanding = 0
        self.peak_outstanding = 0
        self.shutdown_calls = 0

    def submit(self, worker: Callable[[Any], Any], batch: Any) -> RecordingFuture:
        """Accept a batch and run its worker straight away.

        Args:
            worker: Callable to apply to the batch.
            batch: Batch to hand to the worker.

        Returns:
            A future stand-in holding the outcome of the call.

        Raises:
            SubmitLimitExceeded: If more submissions arrive than the limit allows.
        """

        if self.submit_limit is not None and len(self.submitted_batches) >= self.submit_limit:
            raise SubmitLimitExceeded(f"more than {self.submit_limit} batches submitted")

        index = len(self.submitted_batches)
        self.submitted_batches.append(batch)
        self.outstanding += 1
        self.peak_outstanding = max(self.peak_outstanding, self.outstanding)
        return RecordingFuture(index, worker, batch, self)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Record a shutdown request that the driver must never make.

        Args:
            wait: Accepted for signature compatibility and ignored.
            cancel_futures: Accepted for signature compatibility and ignored.
        """

        self.shutdown_calls += 1

    def record_retrieval(self, index: int) -> None:
        """Note that the result of the future at a submission position has been taken.

        Args:
            index: Submission position of the future whose result was retrieved.
        """

        self.result_order.append(index)
        self.outstanding -= 1


class CountingBatches:
    """A batch source that counts how many batches have been pulled out of it."""

    def __init__(self, batches: Iterable[Any]) -> None:
        """Wrap a sequence of batches.

        Args:
            batches: Batches to hand out one at a time.
        """

        self.batches = batches
        self.pulled = 0

    def __iter__(self) -> Iterator[Any]:
        """Yield batches, counting each one as it leaves.

        Yields:
            The next batch.
        """

        for batch in self.batches:
            self.pulled += 1
            yield batch


class ReverseCompletionWorker:
    """A worker that makes each window of batches complete in reverse submission order.

    Every batch first rendezvouses on a barrier sized to the in-flight window, so a whole
    window is running before any of it finishes, and then sleeps for a span graded by the
    batch's position within its window. The last batch of a window therefore finishes
    first and the first finishes last, which is the scheduling a completion-order drain
    would emit and an in-order drain must not. The barrier carries a timeout so a driver
    that never fills the window fails the test instead of hanging it.
    """

    def __init__(self, window: int) -> None:
        """Prepare the rendezvous and the completion log.

        Args:
            window: Number of batches expected to be in flight at once.
        """

        self.window = window
        self.barrier = threading.Barrier(window, timeout=BARRIER_TIMEOUT_S)
        self.lock = threading.Lock()
        self.completions: list[int] = []

    def __call__(self, batch: int) -> int:
        """Wait for the rest of the window, then finish after a graded delay.

        Args:
            batch: Batch value to transform.

        Returns:
            Twice the batch value.
        """

        self.barrier.wait()
        time.sleep((self.window - 1 - batch % self.window) * SLEEP_UNIT_S)
        with self.lock:
            self.completions.append(batch)
        return double(batch)


class TestMapBatchesInOrderResultOrder:
    """Tests that results leave the driver in the order their batches went in."""

    def test_map_batches_in_order_yields_results_in_input_order(self) -> None:
        """Every result comes out at the position of the batch that produced it."""

        batches = list(range(10))
        executor = RecordingExecutor()

        results = list(map_batches_in_order(executor, double, iter(batches), 3))

        assert_that(results).is_equal_to([double(batch) for batch in batches])
        assert_that(executor.submitted_batches).is_equal_to(batches)

    def test_map_batches_in_order_retrieves_results_in_submission_order(self) -> None:
        """Results are collected strictly left to right, oldest submission first."""

        batches = list(range(10))
        executor = RecordingExecutor()

        list(map_batches_in_order(executor, double, iter(batches), 3))

        assert_that(executor.result_order).is_equal_to(list(range(len(batches))))


class TestMapBatchesInOrderCompletionOrder:
    """Tests that output order does not follow the order work happens to finish in."""

    def test_map_batches_in_order_yields_input_order_when_completions_are_reversed(
        self,
    ) -> None:
        """A real pool whose windows finish backwards still yields batches in order.

        This is the property the whole driver exists for: with work completing in the
        reverse of submission order, a drain that emits whatever finished first would
        produce reversed output, so identical input would give different bytes at
        different worker counts.
        """

        worker = ReverseCompletionWorker(REAL_POOL_WINDOW)
        batches = list(range(REAL_POOL_BATCHES))

        with ThreadPoolExecutor(max_workers=REAL_POOL_WINDOW) as executor:
            results = list(
                map_batches_in_order(executor, worker, iter(batches), REAL_POOL_WORKERS)
            )

        assert_that(results).is_equal_to([double(batch) for batch in batches])

        # Proof the test was not vacuous: the pool really did finish the work in some
        # order other than the one it was handed. The exact permutation is left
        # unasserted because it is a property of the scheduler, not of the driver.
        assert_that(worker.completions).is_length(len(batches))
        assert_that(worker.completions).is_not_equal_to(batches)


class TestMapBatchesInOrderWindow:
    """Tests that the number of batches in flight stays inside the window policy."""

    @pytest.mark.parametrize("n_workers", [1, 2, 5])
    def test_map_batches_in_order_keeps_submissions_within_the_window(
        self, n_workers: int
    ) -> None:
        """Outstanding submissions peak at the window and never exceed it."""

        window = expected_window(n_workers)
        batch_count = window * 3 + 1
        executor = RecordingExecutor()

        list(map_batches_in_order(executor, double, iter(range(batch_count)), n_workers))

        assert_that(executor.peak_outstanding).is_equal_to(window)
        assert_that(executor.submitted_batches).is_length(batch_count)

    @pytest.mark.parametrize("n_workers", [1, 2, 5])
    def test_map_batches_in_order_reads_no_further_ahead_than_the_window(
        self, n_workers: int
    ) -> None:
        """The batch source is never pulled more than a window ahead of the consumer.

        This is what keeps resident memory proportional to the window rather than to the
        length of the input, however long the caller spends folding each result.
        """

        window = expected_window(n_workers)
        batch_count = window * 3 + 1
        source = CountingBatches(range(batch_count))
        executor = RecordingExecutor()

        yielded = 0
        for _ in map_batches_in_order(executor, double, source, n_workers):
            yielded += 1
            assert_that(source.pulled - yielded).is_less_than_or_equal_to(window)

        assert_that(yielded).is_equal_to(batch_count)
        assert_that(source.pulled).is_equal_to(batch_count)


class TestMapBatchesInOrderExecutorLifecycle:
    """Tests that the driver leaves the pool's lifecycle entirely to its caller."""

    def test_map_batches_in_order_never_shuts_the_executor_down(self) -> None:
        """Draining every batch leaves the executor running.

        The caller shuts its pool down by leaving the block that owns it, and that has to
        happen before the caller closes its compressed output streams: workers inherit
        the write end of each compressor's input pipe, so a compressor still held open by
        a live worker never sees end of file and the run stops making progress. The driver
        may therefore have no lifecycle role at all -- no shutdown, and by extension no
        entering or exiting of a pool it did not create.
        """

        executor = RecordingExecutor()

        list(map_batches_in_order(executor, double, iter(range(6)), 2))

        assert_that(executor.shutdown_calls).is_equal_to(0)

    def test_map_batches_in_order_never_shuts_the_executor_down_when_abandoned(self) -> None:
        """A generator abandoned part way through still shuts nothing down.

        Outstanding futures are the caller's to drain when it closes its pool, so an early
        exit from the loop must not reach for the pool either.
        """

        executor = RecordingExecutor()
        driver = map_batches_in_order(executor, double, iter(range(6)), 2)

        first = next(driver)
        driver.close()

        assert_that(first).is_equal_to(double(0))
        assert_that(executor.shutdown_calls).is_equal_to(0)


class TestMapBatchesInOrderBatchSource:
    """Tests covering how the driver consumes the batches it is given."""

    def test_map_batches_in_order_handles_fewer_batches_than_the_window(self) -> None:
        """An input that runs out before the window fills drains cleanly."""

        batches = [0, 1, 2]
        executor = RecordingExecutor()

        results = list(map_batches_in_order(executor, double, iter(batches), 4))

        assert_that(results).is_equal_to([double(batch) for batch in batches])
        assert_that(executor.submitted_batches).is_equal_to(batches)
        assert_that(executor.peak_outstanding).is_equal_to(len(batches))
        assert_that(executor.result_order).is_equal_to([0, 1, 2])

    @pytest.mark.parametrize("batches", [[], iter([])], ids=["list", "iterator"])
    def test_map_batches_in_order_with_no_batches_yields_and_submits_nothing(
        self, batches: Iterable[Any]
    ) -> None:
        """An empty input produces no results and touches the executor not at all."""

        executor = RecordingExecutor()

        results = list(map_batches_in_order(executor, double, batches, 2))

        assert_that(results).is_empty()
        assert_that(executor.submitted_batches).is_empty()
        assert_that(executor.shutdown_calls).is_equal_to(0)

    def test_map_batches_in_order_submits_each_batch_of_a_list_exactly_once(self) -> None:
        """A list input is walked once, with no batch submitted twice.

        Taking a slice of a list starts from the front every time, so a driver that does
        not first turn its input into a single iterator resubmits the leading batch for
        ever and never finishes. The executor refuses to take more submissions than there
        are batches so that failure shows up as a raised error rather than a hung test.
        """

        batches = [1, 2, 3, 4, 5, 6, 7]
        executor = RecordingExecutor(submit_limit=len(batches))

        results = list(map_batches_in_order(executor, double, batches, 2))

        assert_that(executor.submitted_batches).is_equal_to(batches)
        assert_that(results).is_equal_to([double(batch) for batch in batches])


class TestMapBatchesInOrderWorkerFailure:
    """Tests that a failing batch fails the run at its own position and nowhere else."""

    @pytest.mark.parametrize("failing_batch", [0, 1, 3, 5])
    def test_map_batches_in_order_raises_after_yielding_the_batches_before_the_failure(
        self, failing_batch: int
    ) -> None:
        """The worker's exception arrives once every earlier batch has been yielded.

        Because the failure lands at a fixed position rather than wherever the doomed
        batch happened to finish, even the partial output written before it is the same
        every run.
        """

        batches = list(range(6))
        executor = RecordingExecutor()
        worker = FailingWorker(failing_batch)

        yielded = []
        with pytest.raises(WorkerFailure) as caught:
            for result in map_batches_in_order(executor, worker, iter(batches), 2):
                yielded.append(result)

        assert_that(str(caught.value)).contains(f"batch {failing_batch} failed")
        assert_that(yielded).is_equal_to([double(batch) for batch in batches[:failing_batch]])


class TestParallelModuleSurface:
    """Tests on the shape of the module itself rather than on any single run."""

    def test_map_batches_in_order_submits_the_first_window_when_called(self) -> None:
        """Calling the driver submits the opening window, before any result is taken.

        The caller forks its pool on that first submit, and has to do so before it starts
        anything that runs a thread of its own, because forking a multi-threaded process
        can leave an inherited lock held for ever in the child.
        """

        executor = RecordingExecutor()

        map_batches_in_order(executor, double, iter(range(10)), REAL_POOL_WORKERS)

        assert_that(executor.submitted_batches).is_length(expected_window(REAL_POOL_WORKERS))
        assert_that(executor.submitted_batches).is_equal_to([0, 1, 2, 3])
        assert_that(executor.result_order).is_empty()

    def test_parallel_exports_no_underscore_prefixed_name(self) -> None:
        """Nothing in the module is named as though it were private."""

        private_names = sorted(
            name
            for name in vars(carmack.parallel)
            if name.startswith("_") and not name.startswith("__")
        )

        assert_that(private_names).is_empty()
