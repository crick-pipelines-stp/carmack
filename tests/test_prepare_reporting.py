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

The same object renders ``detected_targets.txt``, the machine-readable list of
the arms and buckets a run actually wrote a read into, and those tests pin what
a downstream consumer of that file depends on: bare tokens and nothing else, the
``NONE`` sentinel first and only when a read was unmatched, an undetected target
omitted whether it is absent from the tallies or present with a zero count, and
an ordering that agrees with the report's own distribution.

The same value object also renders MultiQC custom content payloads for the run:
``to_mqc_general_stats`` reduces the unmatched and matched counts to the two
percentages of ``total_reads`` a generalstats table needs, reusing ``fraction``
so those percentages can never drift from what ``get_report`` already prints;
``to_mqc_target_distribution`` renders the same per-target distribution as a
bargraph, adding the unmatched arm as one more category so every read the run
saw is accounted for in one chart. Both are keyed by a caller-supplied prefix
and carry the shared Carmack parent identifiers from ``carmack.mqc_report``.
These tests pin the reuse of ``fraction``, the zero-reads edge case, the
prefix keying, JSON-serializability, and -- for the distribution -- that no
read is double-counted or dropped and that the target categories stay sorted
by name.

They also pin what makes each payload addressable in a MultiQC run. A payload
carrying no ``id`` is filed under the cleaned filename MultiQC falls back to, so
an N-sample run renders N separate one-sample sections instead of one section
with N bars, and a writer naming a payload's output file from its own id has
nothing to name the file with. A generalstats payload carrying no ``pconfig``
leaves MultiQC to guess its headers from the raw data keys, so the columns
render as bare ``pct_matched``/``pct_unmatched`` with no title, no percent
suffix and no colour scale beside three other stages' fully configured columns.
The bargraph's ``section_name`` is pinned to differ from assign-targets', read
from that stage's own builder rather than copied, because the two become
sibling sections under one parent and must be tellable apart.

``PrepareCounts`` is the mutable accumulator that feeds it: one batch of reads
is tallied into one of these, batches are folded together as they drain, and
the run's totals are rendered as a frozen ``PrepareStats`` at the end. Its
tests pin the properties that make batching invisible to the statistics - an
additive fold, an untouched argument, order independence, and a partition of a
read set tallying to the same totals as a single pass over it - plus the field
mapping and plain-dict rendering that ``to_stats`` performs.
"""

import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from itertools import permutations
from typing import Any

import pytest
from assertpy import assert_that

from carmack import __version__ as carmack_version
from carmack.assign_targets.assign_reporting import AssignStats
from carmack.assign_targets.target_assigner import NO_TARGET
from carmack.barcode.extraction_dataclasses import MatchMethod
from carmack.barcode.extraction_reporting import ExtractionStats, OverallStats, PerBarcodeStats
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME
from carmack.prepare_reads.prepare_reporting import PrepareCounts, PrepareStats
from carmack.umi.umi_reporting import UmiExtractionStats
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

# Two distinct prefixes, used to prove the mqc payloads key off whatever prefix
# is passed in rather than a name the module hardcodes.
MQC_PREFIX_A = "SK588"
MQC_PREFIX_B = "SK661"

GENERALSTATS_PLOT_TYPE = "generalstats"
BARGRAPH_PLOT_TYPE = "bargraph"

# The MultiQC module id each payload must declare. Pinned as literals because a
# downstream consumer keys off them, and because the id is what a writer names the
# payload's output file from.
PREPARE_GENERAL_STATS_ID = "carmack_prepare_general_stats"
PREPARE_TARGET_DISTRIBUTION_ID = "carmack_prepare_target_distribution"

# The two builders under test, and the id each must declare.
MQC_BUILDER_IDS = {
    "to_mqc_general_stats": PREPARE_GENERAL_STATS_ID,
    "to_mqc_target_distribution": PREPARE_TARGET_DISTRIBUTION_ID,
}

# The stage token the sibling stages spell bare - extraction, umi, tgidx, none of them
# carrying a _stats suffix - so this stage's ids are carmack_prepare_*, and the token
# they must never drift to.
STAGE_TOKEN_PREFIX = f"{CARMACK_PARENT_ID}_prepare_"
REJECTED_STAGE_TOKEN = "prepare_stats"

# The generalstats data columns, each of which pconfig must configure a header for.
GENERAL_STATS_COLUMNS = ("pct_unmatched", "pct_matched")

# Every setting a generalstats header must carry for MultiQC to render the column as a
# titled, bounded, colour-scaled percentage rather than guess it from the data key.
PCONFIG_COLUMN_KEYS = ("title", "description", "min", "max", "suffix", "format", "scale")

# The bounds and formatting that make a column read as a percentage, as every sibling
# stage's percentage columns spell them.
PCONFIG_PERCENTAGE_SETTINGS = {"min": 0, "max": 100, "suffix": "%", "format": "{:,.2f}"}

# How a bargraph's pconfig id is derived from the payload id, and the y-axis label every
# sibling stage's bargraph carries.
PLOT_ID_SUFFIX = "_plot"
BARGRAPH_YLAB = "Reads"


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


def build_mqc_payload(builder_name: str) -> dict[str, Any]:
    """Build one of the two MultiQC payloads by the name of the builder that renders it.

    Lets the properties common to both payloads - the module id, its namespacing,
    its uniqueness - be parametrized over the builders rather than written twice.

    Args:
        builder_name: Name of the ``PrepareStats`` builder to call.

    Returns:
        The payload that builder renders for the default reconciling stats.
    """
    return getattr(make_stats(), builder_name)(MQC_PREFIX_A)


def general_stats_header(column: str) -> dict[str, Any]:
    """Return the header settings the generalstats pconfig declares for one data column.

    MultiQC takes a generalstats pconfig as a list of single-key dicts, one per
    column, so a column's settings are found by searching that list rather than
    indexed straight out of a mapping.

    Args:
        column: Data column whose header settings are wanted.

    Returns:
        The settings mapping declared for that column, or an empty mapping when
        pconfig declares no entry for it.
    """
    for entry in build_mqc_payload("to_mqc_general_stats")["pconfig"]:
        if column in entry:
            return entry[column]
    return {}


def assign_target_distribution_section_name() -> str:
    """Return assign-targets' own target-distribution section heading, from its own builder.

    Read out of the sibling module rather than copied here, so the assertion that
    the two headings differ keeps holding however either stage rewords its own.

    Returns:
        The ``section_name`` assign-targets' target distribution payload carries.
    """
    stats = AssignStats(
        total_reads=0,
        matched=0,
        unmatched_no_match=0,
        unmatched_no_left_anchor_pos=0,
        unmatched_short_window=0,
        target_counts={},
        edit_distance_counts={},
    )
    return str(stats.to_mqc_target_distribution(MQC_PREFIX_A)["section_name"])


def sibling_mqc_payload_ids() -> set[str]:
    """Collect the module id of every MultiQC payload the other carmack stages render.

    Each sibling stats object is populated richly enough that none of its builders
    suppress their payload, so the result is the whole id space this stage's own
    ids have to stay clear of. Gathered by calling the builders rather than listing
    the ids here, so a sibling stage renaming or adding a payload is reflected
    without this file being edited.

    Returns:
        Every module id the barcode, UMI and assign-targets payloads declare.
    """
    siblings = (
        ExtractionStats(
            overall=OverallStats(total_reads=1, perfect=1, corrok=0, fail=0, top_10_barcodes=[]),
            per_barcode=[
                PerBarcodeStats(
                    bc_name="BC1",
                    method=MatchMethod.EXACTMATCH,
                    attempts=1,
                    success=1,
                    fail=0,
                    edit_distance_dist=Counter({0: 1}),
                    reads_w_ambiguous_match=0,
                    spacer_present=0,
                )
            ],
            bc_names=["BC1"],
        ),
        UmiExtractionStats(
            total_reads=1,
            accepted=1,
            missing_left_anchor=0,
            truncated=0,
            umi_length=12,
            homopolymer_base="G",
            homopolymer_run_counts={4: 1},
        ),
        AssignStats(
            total_reads=1,
            matched=1,
            unmatched_no_match=0,
            unmatched_no_left_anchor_pos=0,
            unmatched_short_window=0,
            target_counts={"targetA": 1},
            edit_distance_counts={0: 1},
            homopolymer_base="G",
            homopolymer_run_counts={4: 1},
        ),
    )
    ids = set()
    for stats in siblings:
        for name in dir(stats):
            if not name.startswith("to_mqc_"):
                continue
            payload = getattr(stats, name)(MQC_PREFIX_A)
            if payload is not None:
                ids.add(payload["id"])
    return ids


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


class TestPrepareStatsDetectedTargets:
    """The machine-readable list of arms and buckets that received reads."""

    def test_every_seen_arm_and_target_is_listed_once(self) -> None:
        """Test that the unmatched arm and each target seen appear exactly once each."""
        lines = make_stats().get_detected_targets().splitlines()

        assert_that(lines).is_equal_to([NO_TARGET, "targetA", "targetB", "targetC"])

    def test_the_unmatched_arm_is_listed_first(self) -> None:
        """Test that the scRNA arm's sentinel precedes every target bucket.

        Pinned separately from the sorted order below because the sentinel does
        not sort into the targets: it leads them regardless of how they are named.
        """
        lines = make_stats(target_written={"AAAA": 1}).get_detected_targets().splitlines()

        assert_that(lines).is_equal_to([NO_TARGET, "AAAA"])

    def test_targets_are_sorted_by_name_not_insertion_order(self) -> None:
        """Test that targets render sorted, so two runs of one chemistry diff cleanly."""
        lines = make_stats().get_detected_targets().splitlines()

        assert_that(lines[1:]).is_equal_to(sorted(lines[1:]))

    def test_target_order_agrees_with_the_report_distribution(self) -> None:
        """Test that the two files order the same targets the same way, row for row."""
        stats = make_stats()
        detected = [
            line for line in stats.get_detected_targets().splitlines() if line != NO_TARGET
        ]
        distribution = [
            line.split("\t")[1]
            for line in stats.get_report().split("# Target Distribution")[1].splitlines()
            if line.startswith("\t")
        ]

        assert_that(detected).is_equal_to(distribution)

    def test_unmatched_arm_omitted_when_every_read_matched(self) -> None:
        """Test that a run with no unmatched read does not claim the scRNA arm."""
        stats = make_stats(unmatched_written=0, total_reads=sum(TARGET_WRITTEN.values()))

        assert_that(stats.get_detected_targets().splitlines()).does_not_contain(NO_TARGET)

    def test_scrna_only_run_lists_the_unmatched_arm_alone(self) -> None:
        """Test that a chemistry with no target index yields exactly the one sentinel."""
        stats = make_stats(target_written={}, unmatched_written=TOTAL_READS)

        assert_that(stats.get_detected_targets()).is_equal_to(f"{NO_TARGET}\n")

    def test_zero_count_target_is_treated_as_undetected(self) -> None:
        """Test that a target tallied at zero is omitted, like one absent altogether.

        A bucket no read reached is undetected however it came to be recorded, so
        an explicit zero must not name an empty output file as one worth fanning
        out over.
        """
        stats = make_stats(target_written={"targetA": 3, "targetB": 0}, total_reads=7)

        assert_that(stats.get_detected_targets().splitlines()).is_equal_to([NO_TARGET, "targetA"])

    def test_a_run_that_wrote_nothing_renders_an_empty_string(self) -> None:
        """Test that no read written means no token, rather than a placeholder line."""
        assert_that(make_empty_stats().get_detected_targets()).is_equal_to("")

    def test_every_line_is_a_bare_newline_terminated_token(self) -> None:
        """Test that the file carries tokens only -- no header, comments, counts or blanks.

        This is the whole contract a consumer reads the file under, so it is
        asserted on the rendered text rather than inferred from the token list.
        """
        rendered = make_stats().get_detected_targets()

        assert_that(rendered).ends_with("\n")
        for line in rendered.splitlines():
            assert_that(line).is_equal_to(line.strip())
            assert_that(line).is_not_empty()
            assert_that(line).does_not_contain("#", "\t", "%", " ")

    def test_listed_tokens_are_the_sentinel_or_a_target_name(self) -> None:
        """Test that nothing but the sentinel and the run's own target keys is emitted."""
        stats = make_stats()

        tokens = set(stats.get_detected_targets().splitlines())

        assert_that(tokens).is_subset_of({NO_TARGET, *stats.target_written})


class TestPrepareStatsMqcGeneralStats:
    """to_mqc_general_stats: the generalstats payload reducing counts to two percentages."""

    def test_plot_type_is_generalstats(self) -> None:
        """Test that the payload declares itself a generalstats table."""
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["plot_type"]).is_equal_to(GENERALSTATS_PLOT_TYPE)

    def test_payload_carries_the_shared_carmack_parent_identifiers(self) -> None:
        """Test that the payload attaches to the shared Carmack parent module."""
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    @pytest.mark.parametrize("prefix", [MQC_PREFIX_A, MQC_PREFIX_B])
    def test_data_section_is_keyed_by_the_given_prefix(self, prefix: str) -> None:
        """Test that the data section is keyed by whatever prefix is passed in, not a fixed name."""
        payload = make_stats().to_mqc_general_stats(prefix)

        assert_that(list(payload["data"].keys())).is_equal_to([prefix])

    def test_data_section_contains_exactly_the_two_percentage_fields(self) -> None:
        """Test that the per-prefix section carries pct_unmatched and pct_matched, and nothing else."""
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(set(payload["data"][MQC_PREFIX_A].keys())).is_equal_to(
            {"pct_unmatched", "pct_matched"}
        )

    def test_pct_unmatched_reuses_the_fraction_staticmethod(self) -> None:
        """Test that pct_unmatched is unmatched_written / total_reads * 100, via fraction.

        Computed independently here via the same staticmethod the design calls
        for reuse of, so an implementation that reinvents the percentage math
        cannot drift from it without this test catching the difference.
        """
        stats = make_stats()
        expected = PrepareStats.fraction(stats.unmatched_written, stats.total_reads) * 100

        payload = stats.to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]["pct_unmatched"]).is_equal_to(expected)

    def test_pct_matched_reuses_the_fraction_staticmethod(self) -> None:
        """Test that pct_matched is matched_written / total_reads * 100, via fraction."""
        stats = make_stats()
        expected = PrepareStats.fraction(stats.matched_written, stats.total_reads) * 100

        payload = stats.to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]["pct_matched"]).is_equal_to(expected)

    def test_percentages_reconcile_with_get_report_for_the_same_stats(self) -> None:
        """Test that the payload's percentages are exactly what get_report() already prints.

        Cross-checked against the report's own ``.2%``-formatted text, rather
        than fraction() a second time, so a percentage that reuses fraction()
        but scales it differently from get_report() would still be caught.
        """
        stats = make_stats()
        payload = stats.to_mqc_general_stats(MQC_PREFIX_A)
        report = stats.get_report()

        unmatched_text = f"({payload['data'][MQC_PREFIX_A]['pct_unmatched']:.2f}%)"
        matched_text = f"({payload['data'][MQC_PREFIX_A]['pct_matched']:.2f}%)"

        assert_that(report).contains(unmatched_text)
        assert_that(report).contains(matched_text)

    def test_zero_reads_payload_does_not_raise_and_is_zero(self) -> None:
        """Test that a run with no reads renders 0.0 percentages rather than raising."""
        payload = make_empty_stats().to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]["pct_unmatched"]).is_equal_to(0.0)
        assert_that(payload["data"][MQC_PREFIX_A]["pct_matched"]).is_equal_to(0.0)

    def test_payload_is_json_serializable(self) -> None:
        """Test that the payload round-trips through json.dumps with no lingering Counter or dataclass."""
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        reloaded = json.loads(json.dumps(payload))

        assert_that(reloaded).is_equal_to(payload)

    def test_payload_declares_its_own_module_id(self) -> None:
        """Test that the payload names its own MultiQC module rather than leaving it to be guessed.

        MultiQC's custom-content parser falls back to the cleaned filename when a
        payload carries no id, so the module id varies with the sample and an
        N-sample run renders N separate one-sample tables instead of one table with
        N rows. The literal is pinned because a downstream consumer keys off it.
        """
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        assert_that(payload["id"]).is_equal_to(PREPARE_GENERAL_STATS_ID)

    def test_pconfig_is_a_list_of_single_column_headers(self) -> None:
        """Test that pconfig is a list of one-key dicts, the shape a generalstats pconfig takes.

        MultiQC reads one header definition per list entry, keyed by the data column
        it configures, and every sibling stage spells its generalstats pconfig this
        way. A mapping in its place would configure nothing.
        """
        pconfig = make_stats().to_mqc_general_stats(MQC_PREFIX_A)["pconfig"]

        assert_that(pconfig).is_instance_of(list)
        for entry in pconfig:
            assert_that(entry).is_instance_of(dict)
            assert_that(entry).is_length(1)

    def test_pconfig_configures_exactly_the_columns_the_data_carries(self) -> None:
        """Test that every data column gets a header, and no header names a column that is absent.

        An unconfigured column is one MultiQC guesses from the raw data key, so it
        lands in the General Statistics table as a bare ``pct_matched`` with no
        title, no percent suffix and no colour scale, beside three other stages'
        fully configured columns.
        """
        payload = make_stats().to_mqc_general_stats(MQC_PREFIX_A)

        configured = [next(iter(entry)) for entry in payload["pconfig"]]

        assert_that(configured).is_length(len(GENERAL_STATS_COLUMNS))
        assert_that(set(configured)).is_equal_to(set(GENERAL_STATS_COLUMNS))
        assert_that(set(configured)).is_equal_to(set(payload["data"][MQC_PREFIX_A]))

    @pytest.mark.parametrize("column", GENERAL_STATS_COLUMNS)
    @pytest.mark.parametrize("setting", PCONFIG_COLUMN_KEYS)
    def test_every_column_header_carries_every_setting(self, column: str, setting: str) -> None:
        """Test that each column's header declares the full set of settings the siblings declare.

        Asserted setting by setting rather than on the whole mapping, so a header
        missing one of them names which one, and so a header carrying extra settings
        a future chart needs is not failed for it.
        """
        assert_that(general_stats_header(column)).contains_key(setting)

    @pytest.mark.parametrize("column", GENERAL_STATS_COLUMNS)
    def test_every_column_header_is_titled_and_described_in_words(self, column: str) -> None:
        """Test that each column carries human prose rather than repeating the raw data key.

        The title is the column heading a reader sees in the General Statistics
        table, and the description is its tooltip; leaving them to MultiQC's fallback
        is what this payload does today by carrying no pconfig at all.
        """
        header = general_stats_header(column)

        assert_that(header["title"]).is_instance_of(str)
        assert_that(str(header["title"]).strip()).is_not_empty()
        assert_that(header["title"]).is_not_equal_to(column)
        assert_that(header["description"]).is_instance_of(str)
        assert_that(str(header["description"]).strip()).is_not_empty()

    @pytest.mark.parametrize("column", GENERAL_STATS_COLUMNS)
    @pytest.mark.parametrize("setting,value", list(PCONFIG_PERCENTAGE_SETTINGS.items()))
    def test_every_column_header_renders_a_bounded_percentage(
        self, column: str, setting: str, value: object
    ) -> None:
        """Test that each column is bounded 0-100 and formatted as a suffixed, two-decimal percentage.

        The data section already carries percentages of ``total_reads``, so the
        header has to say so; without it MultiQC renders them as bare unsuffixed
        floats auto-scaled to whatever range the run happened to produce, which is
        not comparable with the neighbouring stages' percentage columns.
        """
        assert_that(general_stats_header(column)).contains_entry({setting: value})

    @pytest.mark.parametrize("column", GENERAL_STATS_COLUMNS)
    def test_every_column_header_names_a_colour_scale(self, column: str) -> None:
        """Test that each column asks for a colour scale, so the table reads at a glance.

        The neighbouring stages' percentage columns are all scaled; an unscaled
        column beside them reads as a column nobody thought worth looking at.
        """
        scale = general_stats_header(column)["scale"]

        assert_that(scale).is_instance_of(str)
        assert_that(str(scale).strip()).is_not_empty()


class TestPrepareStatsMqcTargetDistribution:
    """to_mqc_target_distribution: the bargraph payload over target_written plus the unmatched arm."""

    def test_plot_type_is_bargraph(self) -> None:
        """Test that the payload declares itself a bargraph."""
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["plot_type"]).is_equal_to(BARGRAPH_PLOT_TYPE)

    def test_payload_carries_the_shared_carmack_parent_identifiers(self) -> None:
        """Test that the payload attaches to the shared Carmack parent module."""
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    @pytest.mark.parametrize("prefix", [MQC_PREFIX_A, MQC_PREFIX_B])
    def test_data_section_is_keyed_by_the_given_prefix(self, prefix: str) -> None:
        """Test that the data section is keyed by whatever prefix is passed in, not a fixed name."""
        payload = make_stats().to_mqc_target_distribution(prefix)

        assert_that(list(payload["data"].keys())).is_equal_to([prefix])

    @pytest.mark.parametrize("target,expected_count", list(TARGET_WRITTEN.items()))
    def test_per_target_counts_equal_target_written_exactly(
        self, target: str, expected_count: int
    ) -> None:
        """Test that each target's category count in the payload matches target_written exactly.

        TARGET_WRITTEN is supplied out of lexical order (C, A, B), so
        parametrizing over it also exercises a payload built from an unsorted
        target_written.
        """
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A][target]).is_equal_to(expected_count)

    def test_unmatched_category_equals_unmatched_written_exactly(self) -> None:
        """Test that the unmatched/scRNA-arm category equals unmatched_written exactly."""
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A][NO_TARGET]).is_equal_to(UNMATCHED_WRITTEN)

    def test_target_categories_are_sorted_by_name(self) -> None:
        """Test that the per-target categories are ordered by name, not insertion order.

        TARGET_WRITTEN is supplied out of lexical order (C, A, B); a payload
        that merely echoed dict iteration order would fail this.
        """
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)
        categories = payload["data"][MQC_PREFIX_A]

        target_names = [name for name in categories if name != NO_TARGET]

        assert_that(target_names).is_equal_to(sorted(TARGET_WRITTEN))

    def test_sum_of_every_category_count_equals_total_reads(self) -> None:
        """Test that no read is double-counted or dropped across the categories.

        This stage never filters, so the categories must partition every read
        the run saw exactly once.
        """
        stats = make_stats()

        payload = stats.to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(sum(payload["data"][MQC_PREFIX_A].values())).is_equal_to(stats.total_reads)

    def test_zero_count_target_is_still_included_as_a_category(self) -> None:
        """Test that a target tallied at zero still gets its own category, at zero.

        Unlike get_detected_targets, the bargraph mirrors target_section()'s
        own undistinguishing sort: every key in target_written becomes a
        category, whether or not any read landed in it.
        """
        stats = make_stats(
            target_written={"targetA": 3, "targetB": 0}, unmatched_written=4, total_reads=7
        )

        payload = stats.to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]).is_equal_to(
            {"targetA": 3, "targetB": 0, NO_TARGET: 4}
        )

    def test_scrna_only_chemistry_yields_the_unmatched_category_alone(self) -> None:
        """Test that a chemistry with no target index renders one category: the unmatched arm."""
        stats = make_stats(target_written={}, unmatched_written=TOTAL_READS)

        payload = stats.to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]).is_equal_to({NO_TARGET: TOTAL_READS})

    def test_zero_reads_payload_does_not_raise(self) -> None:
        """Test that a run with no reads renders rather than raising."""
        payload = make_empty_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["data"][MQC_PREFIX_A]).is_equal_to({NO_TARGET: 0})

    def test_payload_is_json_serializable(self) -> None:
        """Test that the payload round-trips through json.dumps with no lingering Counter or dataclass."""
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        reloaded = json.loads(json.dumps(payload))

        assert_that(reloaded).is_equal_to(payload)

    def test_payload_declares_its_own_module_id(self) -> None:
        """Test that the payload names its own MultiQC module rather than leaving it to be guessed.

        Without an id MultiQC falls back to the cleaned filename, so an N-sample run
        renders N separate one-sample bargraphs instead of one chart carrying every
        sample's bars. The literal is pinned because a downstream consumer keys off
        it, and because a writer names this payload's output file from it.
        """
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["id"]).is_equal_to(PREPARE_TARGET_DISTRIBUTION_ID)

    def test_payload_names_and_describes_its_own_section(self) -> None:
        """Test that the bargraph carries both a section heading and a prose description.

        A bargraph payload renders as a MultiQC section of its own; carrying neither
        leaves it under whatever heading the parser derived from the filename, with
        nothing on the page saying what the bars count or against what denominator.
        """
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["section_name"]).is_instance_of(str)
        assert_that(str(payload["section_name"]).strip()).is_not_empty()
        assert_that(payload["description"]).is_instance_of(str)
        assert_that(str(payload["description"]).strip()).is_not_empty()

    def test_section_name_differs_from_the_assign_targets_distribution_section(self) -> None:
        """Test that this section is tellable apart from assign-targets' own target distribution.

        Both hang off the same Carmack parent, and both are "the target
        distribution" for their stage over different denominators - every read the
        run saw here, matched reads only there - so two identically headed sections
        leave a reader no way to know whose numbers are in front of them.
        Assign-targets' heading is read from its own builder rather than copied here,
        so the two stay distinct however either stage rewords its own.
        """
        payload = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)

        assert_that(payload["section_name"]).is_not_equal_to(
            assign_target_distribution_section_name()
        )

    def test_pconfig_is_a_mapping_naming_the_plot_after_the_payload(self) -> None:
        """Test that pconfig is a dict whose id is the payload id plus the plot suffix.

        A bargraph pconfig is a mapping, not the list a generalstats pconfig is, and
        every sibling stage derives its plot id from its payload id this way so the
        plot and the section it sits in stay separately addressable.
        """
        pconfig = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)["pconfig"]

        assert_that(pconfig).is_instance_of(dict)
        assert_that(pconfig["id"]).is_equal_to(f"{PREPARE_TARGET_DISTRIBUTION_ID}{PLOT_ID_SUFFIX}")

    def test_pconfig_titles_the_plot_and_labels_the_y_axis_in_reads(self) -> None:
        """Test that the plot carries a title and says its bars are counted in reads.

        The categories are raw read counts, the same ones target_section() renders,
        so the axis is labelled for reads exactly as every sibling stage's bargraph
        labels it; an unlabelled axis reads as a fraction just as easily.
        """
        pconfig = make_stats().to_mqc_target_distribution(MQC_PREFIX_A)["pconfig"]

        assert_that(pconfig["title"]).is_instance_of(str)
        assert_that(str(pconfig["title"]).strip()).is_not_empty()
        assert_that(pconfig["ylab"]).is_equal_to(BARGRAPH_YLAB)


class TestPrepareStatsMqcPayloadIdentity:
    """The module ids both payloads declare, and the namespace and file names they key."""

    @pytest.mark.parametrize("builder_name", list(MQC_BUILDER_IDS))
    def test_payload_id_is_namespaced_under_the_carmack_parent(self, builder_name: str) -> None:
        """Test that each id sits under the shared parent id and carries a stem after it.

        A writer names each payload's output file from the payload's own id, so an
        id that is missing, or bare once the parent prefix is taken off it, leaves
        nothing to derive a filename stem from.
        """
        payload_id = str(build_mqc_payload(builder_name)["id"])

        assert_that(payload_id).starts_with(f"{CARMACK_PARENT_ID}_")
        assert_that(payload_id[len(CARMACK_PARENT_ID) + 1 :]).is_not_empty()

    @pytest.mark.parametrize("builder_name", list(MQC_BUILDER_IDS))
    def test_payload_id_uses_the_bare_stage_token(self, builder_name: str) -> None:
        """Test that the stage token is the bare ``prepare``, as the siblings' tokens are bare.

        The other stages spell theirs ``extraction``, ``umi`` and ``tgidx``, none of
        which carries a ``_stats`` suffix, and a downstream consumer keys off these
        strings literally, so the token must not drift to ``prepare_stats``.
        """
        payload_id = str(build_mqc_payload(builder_name)["id"])

        assert_that(payload_id).starts_with(STAGE_TOKEN_PREFIX)
        assert_that(payload_id).does_not_contain(REJECTED_STAGE_TOKEN)

    def test_the_two_payloads_declare_different_ids(self) -> None:
        """Test that the generalstats table and the bargraph are separate MultiQC modules.

        They render as separate sections and are written to separately named files
        derived from these ids, so one id shared between them would collapse both
        into a single module and leave one file overwriting the other.
        """
        ids = [build_mqc_payload(builder_name)["id"] for builder_name in MQC_BUILDER_IDS]

        assert_that(set(ids)).is_length(len(MQC_BUILDER_IDS))

    @pytest.mark.parametrize("builder_name", list(MQC_BUILDER_IDS))
    def test_payload_id_collides_with_no_other_stage(self, builder_name: str) -> None:
        """Test that neither id is already claimed by the barcode, UMI or assign-targets payloads.

        Every carmack payload lands in one MultiQC run under one parent, so an id
        reused across stages has one stage's numbers silently displace another's.
        """
        payload_id = build_mqc_payload(builder_name)["id"]

        assert_that(sibling_mqc_payload_ids()).does_not_contain(payload_id)


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
