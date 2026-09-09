"""Tests for the prepare-reads reporting module.

``PrepareStats`` is the reconciling value object for a prepare-reads run: every
input read is dispatched to exactly one output arm - the scRNA (``TGIDX=NONE``)
arm, or one scTIP target bucket - and the same object renders the plain-text
``prepare_stats.txt`` report. Unlike the assign-targets stage this stage never
filters, so ``total_reads`` is the denominator throughout, not just for the
unmatched count: a per-target line is a fraction of every read the run saw, not
a fraction of the reads that matched. These tests pin the value object's
contract (frozen, safe division, the derived ``matched_written`` total), the
reconciling invariant (``unmatched_written + sum(target_written.values()) ==
total_reads``) across the degenerate and mixed cases the design calls out, and
the report's shape: the volatile run-detail header that ``strip_report_run_details``
must keep working against, every section heading, the counts and their
percentages, the literal statement of the reconciling invariant, and the
per-target distribution sorted by target name.

``PrepareCounts`` is the mutable accumulator that feeds it: one batch of reads
is tallied into one of these, batches are folded together as they drain, and
the run's totals are rendered as a frozen ``PrepareStats`` at the end. Its
tests pin the properties that make batching invisible to the statistics - an
additive fold, an untouched argument, order independence, and a partition of a
read set tallying to the same totals as a single pass over it - plus the field
mapping and plain-dict rendering that ``to_stats`` performs.
"""

import dataclasses
from collections import Counter
from collections.abc import Sequence
from itertools import permutations

import pytest
from assertpy import assert_that

from carmack import __version__ as carmack_version
from carmack.prepare_reads.prepare_reporting import PrepareCounts, PrepareStats
from tests.utils import strip_report_run_details

# Three scTIP target buckets, seen three, one and two times respectively, supplied
# out of lexical order so the report's sorting is tested.
TARGET_WRITTEN = {"targetC": 2, "targetA": 3, "targetB": 1}
UNMATCHED_WRITTEN = 4
TOTAL_READS = 10  # UNMATCHED_WRITTEN + sum(TARGET_WRITTEN.values())

RUN_DETAIL_LINES = 2
VERSION_PREFIX = "# Carmack version:"
GENERATED_PREFIX = "# Report generated at:"

REPORT_HEADING = "# Prepare Reads Stats"
FIRST_COUNT_LINE = "Total reads:"

# The literal reconciling formula the report must state, in the field names the
# module exposes, so a reader of the report can check the invariant by eye.
INVARIANT_FORMULA = "unmatched_written + sum(target_written.values()) == total_reads"

# The two scalar tallies the fold has to sum termwise.
SCALAR_FIELD_NAMES = ("total", "unmatched")

# The accumulator has exactly one counter, unlike assign-targets' three.
COUNTER_FIELD_NAMES = ("target_counts",)
COUNTER_KEYS = (("target_counts", "targetA"),)

# How the accumulator's field names map onto PrepareStats'. Spelled out here so the
# rename to_stats performs is pinned by the tests rather than read off the module.
STATS_FIELD_NAMES = {
    "total": "total_reads",
    "unmatched": "unmatched_written",
    "target_counts": "target_written",
}

# The one PrepareStats field that must arrive as a plain dict. Counter is a dict
# subclass, so a leaked Counter would satisfy an isinstance check; this is asserted
# on its exact type instead.
STATS_DICT_FIELD_NAMES = ("target_written",)

# One read's contribution to a batch's tallies: its outcome, then the target bucket
# a matched read was written to.
type ReadOutcome = tuple[str, str | None]

# A read set covering both outcomes, with every counter key seen more than once and
# ordered so that reversing the set populates the counter in a different order. Its
# tallies are exactly TARGET_WRITTEN and UNMATCHED_WRITTEN above.
READS: list[ReadOutcome] = [
    ("unmatched", None),
    ("matched", "targetA"),
    ("unmatched", None),
    ("matched", "targetB"),
    ("unmatched", None),
    ("matched", "targetA"),
    ("matched", "targetC"),
    ("unmatched", None),
    ("matched", "targetC"),
    ("matched", "targetA"),
]

# Where the read set is cut into the two batches the fold tests use. Cut here, every
# counter key the second batch touches is also touched by the first, and that overlap
# is what lets those tests tell an additive fold from a replacing one.
FOLD_SPLIT = 5
FIRST_BATCH_READS = READS[:FOLD_SPLIT]
SECOND_BATCH_READS = READS[FOLD_SPLIT:]

# Partitions of the read set into batches, as batch sizes. The single batch is today's
# serial loop; the others are batchings a pool could produce, including one read per
# batch, uneven batches, and an empty batch mid-run.
READ_PARTITIONS = [(10,), (1,) * 10, (5, 5), (2, 3, 5), (4, 0, 6), (3, 3, 4)]

# The partition whose batches are folded in every possible order.
PERMUTED_PARTITION = (3, 3, 4)


def make_stats(**overrides: object) -> PrepareStats:
    """Build a ``PrepareStats`` whose default field values reconcile.

    The defaults satisfy the invariant the producer must satisfy: the unmatched
    count plus the per-target distribution's total equals ``total_reads``.

    Args:
        **overrides: Field values replacing the reconciling defaults.

    Returns:
        A freshly constructed PrepareStats.
    """
    values: dict[str, object] = {
        "total_reads": TOTAL_READS,
        "unmatched_written": UNMATCHED_WRITTEN,
        "target_written": dict(TARGET_WRITTEN),
    }
    values.update(overrides)
    return PrepareStats(**values)


def make_empty_stats() -> PrepareStats:
    """Build a ``PrepareStats`` for a run that processed no reads."""
    return PrepareStats(total_reads=0, unmatched_written=0, target_written={})


def tally(reads: Sequence[ReadOutcome]) -> PrepareCounts:
    """Tally a sequence of read outcomes the way one worker's batch would.

    Args:
        reads: Read outcomes making up the batch.

    Returns:
        The batch's tallies, with the target counter populated in the order the
        reads are given, so insertion order is under the caller's control.
    """
    counts = PrepareCounts()
    for outcome, target in reads:
        counts.total += 1
        if outcome == "unmatched":
            counts.unmatched += 1
        else:
            counts.target_counts[target] += 1
    return counts


def split(reads: Sequence[ReadOutcome], sizes: Sequence[int]) -> list[list[ReadOutcome]]:
    """Split reads into consecutive batches of the given sizes.

    Args:
        reads: Read outcomes to split.
        sizes: Batch sizes, which must sum to the number of reads.

    Returns:
        One list of read outcomes per size, in input order.
    """
    batches = []
    start = 0
    for size in sizes:
        batches.append(list(reads[start : start + size]))
        start += size
    return batches


def outcome_sum(counts: PrepareCounts) -> int:
    """Return the unmatched tally plus every target tally, which must equal ``total``.

    Args:
        counts: Tallies to reconcile.

    Returns:
        The sum of the unmatched tally and every value in the target counter.
    """
    return counts.unmatched + sum(counts.target_counts.values())


class TestPrepareStatsConstruction:
    """Field access on a freshly constructed PrepareStats."""

    def test_fields_are_set_from_constructor(self) -> None:
        """Test that every constructor argument is readable back unchanged."""
        stats = make_stats()

        assert_that(stats.total_reads).is_equal_to(TOTAL_READS)
        assert_that(stats.unmatched_written).is_equal_to(UNMATCHED_WRITTEN)
        assert_that(stats.target_written).is_equal_to(dict(TARGET_WRITTEN))

    def test_target_written_accepts_an_empty_mapping(self) -> None:
        """Test that a scRNA-only chemistry's empty target map is a legal value."""
        stats = PrepareStats(total_reads=5, unmatched_written=5, target_written={})

        assert_that(stats.target_written).is_equal_to({})


class TestPrepareStatsValueObject:
    """The dataclass contract: immutability, safe division and the derived total."""

    @pytest.mark.parametrize(
        "field_name,new_value",
        [
            ("total_reads", 99),
            ("unmatched_written", 0),
            ("target_written", {}),
        ],
    )
    def test_stats_are_frozen(self, field_name: str, new_value: object) -> None:
        """Test that assigning to any field of PrepareStats raises FrozenInstanceError."""
        stats = make_stats()

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(stats, field_name, new_value)

    @pytest.mark.parametrize(
        "numerator,denominator,expected",
        [
            (1, 2, 0.5),
            (3, 3, 1.0),
            (0, 5, 0.0),
            (0, 0, 0.0),
            (5, 0, 0.0),
        ],
    )
    def test_fraction_returns_quotient_or_zero(
        self, numerator: int, denominator: int, expected: float
    ) -> None:
        """Test that fraction divides, and returns 0.0 rather than raising on a zero denominator."""
        assert_that(PrepareStats.fraction(numerator, denominator)).is_equal_to(expected)

    def test_fraction_is_callable_on_the_class(self) -> None:
        """Test that fraction is a staticmethod, usable without an instance."""
        assert_that(PrepareStats.fraction(1, 4)).is_equal_to(0.25)

    @pytest.mark.parametrize(
        "target_written,expected",
        [
            ({}, 0),
            ({"targetA": 5}, 5),
            ({"targetA": 3, "targetB": 1, "targetC": 2}, 6),
        ],
    )
    def test_matched_written_sums_target_written(
        self, target_written: dict[str, int], expected: int
    ) -> None:
        """Test that matched_written is derived from target_written, not stored separately.

        Accessed as an attribute rather than called, so a method-shaped
        implementation (returning a bound method object) fails this comparison
        against a plain int.
        """
        stats = make_stats(target_written=target_written, unmatched_written=0)

        assert_that(stats.matched_written).is_equal_to(expected)

    @pytest.mark.parametrize(
        "total_reads,unmatched_written,target_written",
        [
            (0, 0, {}),
            (5, 5, {}),
            (5, 0, {"targetA": 5}),
            (10, 4, {"targetA": 3, "targetB": 1, "targetC": 2}),
        ],
        ids=["zero_reads", "unmatched_only", "matched_only_single_bucket", "mixed_multi_bucket"],
    )
    def test_outcome_counts_reconcile_to_total_reads(
        self, total_reads: int, unmatched_written: int, target_written: dict[str, int]
    ) -> None:
        """Test the reconciling invariant the producer must satisfy in every shape of run.

        The dataclass does not enforce it; this documents it as the contract,
        across the degenerate zero-reads case, a scRNA-only chemistry, a single
        matched bucket, and a mixed multi-bucket run.
        """
        stats = PrepareStats(
            total_reads=total_reads,
            unmatched_written=unmatched_written,
            target_written=target_written,
        )

        assert_that(stats.unmatched_written + stats.matched_written).is_equal_to(stats.total_reads)


class TestPrepareStatsReportRunDetails:
    """The volatile run-detail header that golden comparisons strip."""

    def test_report_opens_with_version_and_timestamp_lines(self) -> None:
        """Test that the first two lines carry the exact volatile prefixes."""
        lines = make_stats().get_report().splitlines()

        assert_that(lines[0]).is_equal_to(f"{VERSION_PREFIX} {carmack_version}")
        assert_that(lines[1]).starts_with(f"{GENERATED_PREFIX} ")

    def test_strip_report_run_details_removes_exactly_those_two_lines(self) -> None:
        """Test that stripping the run details leaves every other line untouched."""
        report = make_stats().get_report()
        lines = report.splitlines(keepends=True)

        assert_that(strip_report_run_details(report)).is_equal_to(
            "".join(lines[RUN_DETAIL_LINES:])
        )


class TestPrepareStatsReportSections:
    """Section headings and the literal statement of the reconciling invariant."""

    @pytest.mark.parametrize("heading", [REPORT_HEADING, "# Target Distribution"])
    def test_report_carries_every_section_heading(self, heading: str) -> None:
        """Test that each section of the report is present and labelled."""
        assert_that(make_stats().get_report()).contains(heading)

    def test_target_distribution_heading_present_even_with_no_targets(self) -> None:
        """Test that a scRNA-only run still renders the (empty) target section heading."""
        report = make_stats(target_written={}, unmatched_written=TOTAL_READS).get_report()

        assert_that(report).contains("# Target Distribution")

    def test_report_states_the_reconciling_invariant_literally(self) -> None:
        """Test that the report literally states the invariant, not just the numbers.

        A reader of the report text must be able to see the reconciling formula
        by eye, in the field names the module exposes, rather than infer it from
        the numbers alone.
        """
        assert_that(make_stats().get_report()).contains(INVARIANT_FORMULA)

    def test_invariant_statement_precedes_the_counts(self) -> None:
        """Test that the invariant is read before the numbers it reconciles."""
        report = make_stats().get_report()

        assert_that(report.index(INVARIANT_FORMULA)).is_less_than(report.index(FIRST_COUNT_LINE))

    def test_report_does_not_call_written_reads_rejected(self) -> None:
        """Test that dispatch wording is used: no read outcome is labelled rejected.

        This stage never filters - every read is written to exactly one arm -
        so nothing in the counts section should be described as rejected.
        """
        assert_that(make_stats().get_report()).does_not_contain("Rejected")


class TestPrepareStatsReportCounts:
    """The total, unmatched and matched counts and their percentages of total_reads."""

    @pytest.mark.parametrize(
        "expected_line",
        [
            "Total reads: 10",
            "Unmatched (scRNA arm): 4 (40.00%)",
            "Matched (scTIP arms): 6 (60.00%)",
        ],
    )
    def test_report_renders_counts_against_total_reads(self, expected_line: str) -> None:
        """Test that every top-level count renders with its percentage of total_reads."""
        assert_that(make_stats().get_report()).contains(expected_line)

    def test_report_handles_zero_reads_without_error(self) -> None:
        """Test that a run with no reads renders rather than raising ZeroDivisionError."""
        report = make_empty_stats().get_report()

        assert_that(report).contains("Total reads: 0")
        assert_that(report).contains("Unmatched (scRNA arm): 0 (0.00%)")
        assert_that(report).contains("Matched (scTIP arms): 0 (0.00%)")
        assert_that(report).contains("# Target Distribution")


class TestPrepareStatsReportDistributions:
    """The per-target distribution, rendered as a fraction of total_reads."""

    @pytest.mark.parametrize(
        "expected_entry",
        ["\ttargetA\t3 (30.00%)", "\ttargetB\t1 (10.00%)", "\ttargetC\t2 (20.00%)"],
    )
    def test_target_distribution_renders_against_total_reads(self, expected_entry: str) -> None:
        """Test that per-target counts render as a percentage of total_reads, not matched_written.

        Unlike assign-targets' target_section(), this stage never filters, so
        every target line's denominator is the same total_reads the unmatched
        count uses, not the sum of matched reads alone.
        """
        assert_that(make_stats().get_report()).contains(expected_entry)

    def test_target_distribution_is_sorted(self) -> None:
        """Test that targets appear in sorted order, not insertion order."""
        report = make_stats().get_report()

        assert_that(report.index("\ttargetA\t")).is_less_than(report.index("\ttargetB\t"))
        assert_that(report.index("\ttargetB\t")).is_less_than(report.index("\ttargetC\t"))

    def test_target_distribution_entries_omitted_when_target_written_is_empty(self) -> None:
        """Test that no target rows render for a scRNA-only chemistry."""
        report = make_stats(target_written={}, unmatched_written=TOTAL_READS).get_report()
        section = report.split("# Target Distribution")[1]

        assert_that(section.strip()).is_equal_to("")


class TestPrepareCountsConstruction:
    """Field access on a PrepareCounts built with explicit, non-default values."""

    def test_explicit_values_are_stored(self) -> None:
        """Test that constructor arguments are readable back unchanged."""
        counts = PrepareCounts(total=5, unmatched=2, target_counts=Counter({"targetA": 3}))

        assert_that(counts.total).is_equal_to(5)
        assert_that(counts.unmatched).is_equal_to(2)
        assert_that(counts.target_counts).is_equal_to(Counter({"targetA": 3}))


class TestPrepareCountsDefaults:
    """The accumulator's starting state and its mutable-default contract."""

    @pytest.mark.parametrize("field_name", SCALAR_FIELD_NAMES)
    def test_default_scalar_tallies_start_at_zero(self, field_name: str) -> None:
        """Test that a fresh accumulator has counted nothing before any batch is folded in."""
        assert_that(getattr(PrepareCounts(), field_name)).is_equal_to(0)

    @pytest.mark.parametrize("field_name", COUNTER_FIELD_NAMES)
    def test_default_counters_start_empty(self, field_name: str) -> None:
        """Test that a fresh accumulator's target distribution starts with no keys at all."""
        assert_that(getattr(PrepareCounts(), field_name)).is_empty()

    @pytest.mark.parametrize("field_name,key", COUNTER_KEYS)
    def test_default_counters_are_not_shared_between_instances(
        self, field_name: str, key: object
    ) -> None:
        """Test that every accumulator owns its counter, so batches cannot alias.

        A mutable default shared between instances would make every batch's
        tallies the same object, and one batch's counts would show up in
        another's before anything had been folded.
        """
        first = PrepareCounts()
        second = PrepareCounts()

        getattr(first, field_name)[key] += 1

        assert_that(getattr(second, field_name)).is_empty()
        assert_that(getattr(first, field_name)).is_not_same_as(getattr(second, field_name))


class TestPrepareCountsFold:
    """``add``: the once-per-batch fold the parent applies to each batch's tallies."""

    @pytest.mark.parametrize("field_name", SCALAR_FIELD_NAMES)
    def test_add_sums_each_scalar_tally(self, field_name: str) -> None:
        """Test that folding adds the batch's scalar tallies onto the running totals."""
        totals = tally(FIRST_BATCH_READS)
        batch = tally(SECOND_BATCH_READS)
        expected = getattr(totals, field_name) + getattr(batch, field_name)

        totals.add(batch)

        assert_that(getattr(totals, field_name)).is_equal_to(expected)

    @pytest.mark.parametrize("field_name", COUNTER_FIELD_NAMES)
    def test_add_is_additive_on_counter_keys_seen_in_both_batches(self, field_name: str) -> None:
        """Test that folding adds counter values for a shared key rather than replacing them.

        This is the whole difference between ``Counter.update`` and
        ``dict.update``: replacing would silently discard every count an earlier
        batch recorded for a key a later batch also saw. The two batches
        deliberately overlap on the counter, so no replacing fold can satisfy
        this, which a partition into disjoint keys would let pass.
        """
        totals = tally(FIRST_BATCH_READS)
        batch = tally(SECOND_BATCH_READS)
        before = dict(getattr(totals, field_name))
        folded_in = dict(getattr(batch, field_name))
        shared = set(before) & set(folded_in)
        assert_that(shared).is_not_empty()

        totals.add(batch)

        assert_that(getattr(totals, field_name)).is_equal_to(Counter(before) + Counter(folded_in))
        for key in shared:
            assert_that(getattr(totals, field_name)[key]).is_greater_than(
                max(before[key], folded_in[key])
            )

    def test_add_leaves_the_folded_batch_untouched(self) -> None:
        """Test that folding does not corrupt the batch it read its tallies from."""
        batch = tally(SECOND_BATCH_READS)

        tally(FIRST_BATCH_READS).add(batch)

        assert_that(batch).is_equal_to(tally(SECOND_BATCH_READS))

    def test_add_folds_in_place_and_returns_nothing(self) -> None:
        """Test that the fold mutates the receiver itself rather than returning a new total."""
        totals = tally(FIRST_BATCH_READS)
        held_elsewhere = totals
        before = totals.total

        result = totals.add(tally(SECOND_BATCH_READS))

        assert_that(result).is_none()
        assert_that(held_elsewhere).is_same_as(totals)
        assert_that(held_elsewhere.total).is_greater_than(before)

    def test_folding_an_empty_batch_changes_nothing(self) -> None:
        """Test that an empty batch is the identity of the fold, as a drained pool yields."""
        totals = tally(READS)

        totals.add(PrepareCounts())

        assert_that(totals).is_equal_to(tally(READS))

    def test_folding_into_a_fresh_accumulator_yields_the_batch(self) -> None:
        """Test that the first fold of a run reproduces that batch's tallies exactly."""
        totals = PrepareCounts()

        totals.add(tally(READS))

        assert_that(totals).is_equal_to(tally(READS))

    @pytest.mark.parametrize("order", list(permutations(range(len(PERMUTED_PARTITION)))))
    def test_fold_order_does_not_change_the_totals(self, order: tuple[int, ...]) -> None:
        """Test that the totals do not depend on which batch was folded first.

        This is what keeps the statistics independent of which worker happened
        to finish first.
        """
        batches = split(READS, PERMUTED_PARTITION)
        totals = PrepareCounts()

        for index in order:
            totals.add(tally(batches[index]))

        assert_that(totals).is_equal_to(tally(READS))

    @pytest.mark.parametrize("sizes", READ_PARTITIONS)
    def test_outcome_invariant_holds_on_every_batch(self, sizes: tuple[int, ...]) -> None:
        """Test that each batch's own tallies reconcile, so a batch-local slip is visible here."""
        for batch in split(READS, sizes):
            counts = tally(batch)

            assert_that(outcome_sum(counts)).is_equal_to(counts.total)

    @pytest.mark.parametrize("sizes", READ_PARTITIONS)
    def test_outcome_invariant_holds_on_the_fold(self, sizes: tuple[int, ...]) -> None:
        """Test that the folded totals reconcile, so a fold-local slip is visible separately."""
        totals = PrepareCounts()

        for batch in split(READS, sizes):
            totals.add(tally(batch))

        assert_that(outcome_sum(totals)).is_equal_to(totals.total)
        assert_that(totals.total).is_equal_to(len(READS))

    @pytest.mark.parametrize("sizes", READ_PARTITIONS)
    def test_folding_any_partition_equals_the_single_pass_tally(
        self, sizes: tuple[int, ...]
    ) -> None:
        """Test that how the reads were batched cannot be seen in the totals.

        Every partition of one read set has to fold to what a single pass over
        the whole set tallies, which is what makes batching invisible to the
        statistics.
        """
        assert_that(sum(sizes)).is_equal_to(len(READS))
        totals = PrepareCounts()

        for batch in split(READS, sizes):
            totals.add(tally(batch))

        assert_that(totals).is_equal_to(tally(READS))


class TestPrepareCountsToStats:
    """``to_stats``: handing the accumulated tallies off as the run's frozen value object."""

    @pytest.mark.parametrize("counts_field,stats_field", list(STATS_FIELD_NAMES.items()))
    def test_to_stats_maps_every_tally_onto_its_stats_field(
        self, counts_field: str, stats_field: str
    ) -> None:
        """Test that each tally arrives in the PrepareStats field it is named for."""
        counts = tally(READS)

        stats = counts.to_stats()

        assert_that(getattr(stats, stats_field)).is_equal_to(getattr(counts, counts_field))

    @pytest.mark.parametrize("stats_field", STATS_DICT_FIELD_NAMES)
    def test_to_stats_returns_plain_dicts_not_counters(self, stats_field: str) -> None:
        """Test that no Counter leaks into the frozen value object.

        Asserted on the exact type, because Counter is a dict subclass and an
        isinstance check would not catch the leak.
        """
        stats = tally(READS).to_stats()

        assert_that(type(getattr(stats, stats_field)) is dict).is_true()

    @pytest.mark.parametrize("field_name", COUNTER_FIELD_NAMES)
    def test_reversing_the_reads_populates_the_counter_in_a_different_order(
        self, field_name: str
    ) -> None:
        """Test the premise of the order-independence test: the insertion orders do differ."""
        forward = getattr(tally(READS), field_name)
        reverse = getattr(tally(list(reversed(READS))), field_name)

        assert_that(list(reverse)).is_not_equal_to(list(forward))
        assert_that(reverse).is_equal_to(forward)

    def test_to_stats_report_is_insertion_order_independent(self) -> None:
        """Test that the order keys were first counted in cannot reach the report.

        Two accumulators holding equal counts, populated in different orders,
        must render the same report. This is what proves the worker count
        cannot leak into the stats output.
        """
        forward = tally(READS).to_stats().get_report()
        reverse = tally(list(reversed(READS))).to_stats().get_report()

        assert_that(strip_report_run_details(reverse)).is_equal_to(
            strip_report_run_details(forward)
        )

    def test_to_stats_on_a_fresh_accumulator_renders_the_zero_report(self) -> None:
        """Test that a run that processed no reads renders the empty report, not an error."""
        report = PrepareCounts().to_stats().get_report()

        assert_that(strip_report_run_details(report)).is_equal_to(
            strip_report_run_details(make_empty_stats().get_report())
        )
        assert_that(report).contains("Total reads: 0")
        assert_that(report).contains("# Target Distribution")
