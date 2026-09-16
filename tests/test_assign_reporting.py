"""Tests for the assign-targets reporting module.

``AssignStats`` is the reconciling value object for a target-index assignment
run: every input read lands in exactly one of ``matched``,
``unmatched_no_match``, ``unmatched_no_left_anchor_pos`` or ``unmatched_short_window``,
and the same object renders the plain-text ``tgidx_stats.txt`` report. These
tests pin the value object's contract (frozen, safe division, the derived
run-length denominator) and the report's shape: the volatile run-detail header
that ``strip_report_run_details`` must keep working against, every section
heading, the counts and their percentages, the three distributions with their
stated denominators, and the denominator note that stops the matched fraction
being read as a modality fraction of the library.

``AssignCounts`` is the mutable accumulator that feeds it: one batch of reads is
tallied into one of these, batches are folded together as they drain, and the
run's totals are rendered as a frozen ``AssignStats`` at the end. Its tests pin
the properties that make batching invisible to the statistics - an additive
fold, an untouched argument, order independence, and a partition of a read set
tallying to the same totals as a single pass over it - plus the field mapping and
plain-dict rendering that ``to_stats`` performs.
"""

import dataclasses
import re
from collections import Counter
from collections.abc import Sequence
from itertools import permutations

import pytest
from assertpy import assert_that

from carmack import __version__ as carmack_version
from carmack.assign_targets.assign_reporting import AssignCounts, AssignStats
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME
from tests.utils import strip_report_run_details

# A whitelist entry seen three times, one seen twice, one seen once. They are
# deliberately supplied out of lexical order so the report's sorting is tested.
TARGET_COUNTS = {"GGGGCCCC": 2, "ACGTACGT": 3, "TTTTAAAA": 1}
# Both distributions are deliberately out of ascending key order. The linegraph
# payloads they feed have to come out sorted by x whatever order they went in, so
# tidying these into ascending order would make those assertions pass on an
# unsorted payload and stop testing the thing they exist to test.
EDIT_DISTANCE_COUNTS = {2: 1, 0: 3, 1: 2}
RUN_COUNTS = {5: 1, 3: 6, 4: 2}

# The same two distributions as MultiQC has to receive them: ascending [x, y] pairs.
EDIT_DISTANCE_PAIRS = [[0, 3], [1, 2], [2, 1]]
RUN_PAIRS = [[3, 6], [4, 2], [5, 1]]

RUN_DETAIL_LINES = 2
VERSION_PREFIX = "# Carmack version:"
GENERATED_PREFIX = "# Report generated at:"
TIMESTAMP_PATTERN = re.compile(r"^# Report generated at: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

# The denial the denominator note has to make. Matched against the note with its
# wrapping normalised away, so the assertion is about the claim the note makes
# and not about the column the prose happens to wrap at. ``\bnot\b`` rather than
# a bare substring, so "note", "nothing" and "cannot" do not satisfy it.
DENIAL_PATTERN = re.compile(r"\bnot\b.*\blibrary\b")
REPORT_HEADING = "# Target Index Assignment Stats"
FIRST_COUNT_LINE = "Total reads:"

# The anchor homopolymer base handed to to_stats, matching the base the AssignStats
# fixtures above render with.
ANCHOR_BASE = "G"

# The five scalar tallies the fold has to sum termwise.
SCALAR_FIELD_NAMES = ("total", "matched", "no_left_anchor_pos", "short_window", "no_match")

# One legal key per counter, so a test that writes into a counter writes a key of that
# counter's own type rather than one key forced to stand for all three.
COUNTER_KEYS = (
    ("target_counts", "ACGTACGT"),
    ("edit_distance_counts", 0),
    ("run_counts", 3),
)
COUNTER_FIELD_NAMES = tuple(field_name for field_name, _ in COUNTER_KEYS)

# How the accumulator's field names map onto AssignStats'. Spelled out here so the
# rename to_stats performs is pinned by the tests rather than read off the module.
STATS_FIELD_NAMES = {
    "total": "total_reads",
    "matched": "matched",
    "no_match": "unmatched_no_match",
    "no_left_anchor_pos": "unmatched_no_left_anchor_pos",
    "short_window": "unmatched_short_window",
    "target_counts": "target_counts",
    "edit_distance_counts": "edit_distance_counts",
    "run_counts": "homopolymer_run_counts",
}

# The three AssignStats fields that must arrive as plain dicts. Counter is a dict
# subclass, so a leaked Counter would satisfy an isinstance check; these are asserted
# on their exact type instead.
STATS_DICT_FIELD_NAMES = ("target_counts", "edit_distance_counts", "homopolymer_run_counts")

# One read's contribution to a batch's tallies: its outcome, then the whitelist entry
# and edit distance a matched read was called at, and the anchor run length measured
# for any read whose anchor position tag was present.
type ReadOutcome = tuple[str, str | None, int | None, int | None]

# A read set covering all four outcomes, with every counter key seen more than once and
# ordered so that reversing the set populates all three counters in a different order.
READS: list[ReadOutcome] = [
    ("matched", "ACGTACGT", 0, 3),
    ("matched", "GGGGCCCC", 1, 4),
    ("no_left_anchor_pos", None, None, None),
    ("matched", "ACGTACGT", 1, 3),
    ("short_window", None, None, 5),
    ("no_match", None, None, 4),
    ("matched", "TTTTAAAA", 2, 3),
    ("no_match", None, None, 3),
    ("matched", "ACGTACGT", 0, 5),
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
READ_PARTITIONS = [(9,), (1,) * 9, (4, 5), (2, 3, 4), (4, 0, 5), (3, 3, 3)]

# The partition whose batches are folded in every possible order.
PERMUTED_PARTITION = (3, 3, 3)


def make_stats(**overrides: object) -> AssignStats:
    """Build an ``AssignStats`` whose default field values reconcile.

    The defaults satisfy every invariant the producer must satisfy: the four
    outcome counts sum to ``total_reads``, the target and edit-distance
    distributions each sum to ``matched``, and the run-length distribution sums
    to the reads whose anchor run could be measured at all.

    Args:
        **overrides: Field values replacing the reconciling defaults.

    Returns:
        A freshly constructed AssignStats.
    """
    values: dict[str, object] = {
        "total_reads": 10,
        "matched": 6,
        "unmatched_no_match": 2,
        "unmatched_no_left_anchor_pos": 1,
        "unmatched_short_window": 1,
        "target_counts": dict(TARGET_COUNTS),
        "edit_distance_counts": dict(EDIT_DISTANCE_COUNTS),
        "homopolymer_base": "G",
        "homopolymer_run_counts": dict(RUN_COUNTS),
    }
    values.update(overrides)
    return AssignStats(**values)


def make_empty_stats() -> AssignStats:
    """Build an ``AssignStats`` for a run that processed no reads."""
    return AssignStats(
        total_reads=0,
        matched=0,
        unmatched_no_match=0,
        unmatched_no_left_anchor_pos=0,
        unmatched_short_window=0,
        target_counts={},
        edit_distance_counts={},
    )


def tally(reads: Sequence[ReadOutcome]) -> AssignCounts:
    """Tally a sequence of read outcomes the way one worker's batch would.

    Args:
        reads: Read outcomes making up the batch.

    Returns:
        The batch's tallies, with each counter populated in the order the reads
        are given, so insertion order is under the caller's control.
    """
    counts = AssignCounts()
    for outcome, target, edit_distance, run_length in reads:
        counts.total += 1
        setattr(counts, outcome, getattr(counts, outcome) + 1)
        if run_length is not None:
            counts.run_counts[run_length] += 1
        if target is not None:
            counts.target_counts[target] += 1
            counts.edit_distance_counts[edit_distance] += 1
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


def outcome_sum(counts: AssignCounts) -> int:
    """Return the four outcome tallies summed, which must equal ``total``.

    Args:
        counts: Tallies to reconcile.

    Returns:
        The sum of the matched and the three unmatched tallies.
    """
    return counts.matched + counts.no_match + counts.no_left_anchor_pos + counts.short_window


class TestAssignStatsValueObject:
    """The dataclass contract: immutability, safe division and the derived denominator."""

    @pytest.mark.parametrize(
        "field_name,new_value",
        [
            ("total_reads", 99),
            ("matched", 0),
            ("unmatched_no_match", 7),
            ("homopolymer_base", "T"),
        ],
    )
    def test_stats_are_frozen(self, field_name: str, new_value: object) -> None:
        """Test that assigning to any field of AssignStats raises FrozenInstanceError."""
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
        assert_that(AssignStats.fraction(numerator, denominator)).is_equal_to(expected)

    def test_fraction_is_callable_on_the_class(self) -> None:
        """Test that fraction is a staticmethod, usable without an instance."""
        assert_that(AssignStats.fraction(1, 4)).is_equal_to(0.25)

    @pytest.mark.parametrize(
        "run_counts,expected",
        [
            ({}, 0),
            ({3: 1}, 1),
            ({3: 6, 4: 2, 5: 1}, 9),
        ],
    )
    def test_reads_with_measured_run_sums_the_counter(
        self, run_counts: dict[int, int], expected: int
    ) -> None:
        """Test that the run-length denominator is derived from the counter, never stored."""
        stats = make_stats(homopolymer_run_counts=run_counts)

        assert_that(stats.reads_with_measured_run).is_equal_to(expected)

    def test_outcome_counts_reconcile_to_total_reads(self) -> None:
        """Test the reconciling invariant the producer must satisfy.

        The dataclass does not enforce it; this documents it as the contract.
        """
        stats = make_stats()

        assert_that(
            stats.matched
            + stats.unmatched_no_match
            + stats.unmatched_no_left_anchor_pos
            + stats.unmatched_short_window
        ).is_equal_to(stats.total_reads)


class TestAssignStatsReportRunDetails:
    """The volatile run-detail header that golden comparisons strip."""

    def test_report_opens_with_version_and_timestamp_lines(self) -> None:
        """Test that the first two lines carry the exact volatile prefixes."""
        lines = make_stats().get_report().splitlines()

        assert_that(lines[0]).is_equal_to(f"{VERSION_PREFIX} {carmack_version}")
        assert_that(lines[1]).starts_with(f"{GENERATED_PREFIX} ")
        assert_that(TIMESTAMP_PATTERN.match(lines[1])).is_not_none()

    def test_strip_report_run_details_removes_exactly_those_two_lines(self) -> None:
        """Test that stripping the run details leaves every other line untouched."""
        report = make_stats().get_report()
        lines = report.splitlines(keepends=True)

        assert_that(strip_report_run_details(report)).is_equal_to(
            "".join(lines[RUN_DETAIL_LINES:])
        )


class TestAssignStatsReportSections:
    """Section headings and the load-bearing denominator note."""

    @pytest.mark.parametrize(
        "heading",
        [
            "# Target Index Assignment Stats",
            "# Target Distribution",
            "# Edit Distance Distribution",
            "# Anchor G-run Length Distribution",
        ],
    )
    def test_report_carries_every_section_heading(self, heading: str) -> None:
        """Test that each section of the report is present and labelled."""
        assert_that(make_stats().get_report()).contains(heading)

    def test_denominator_note_names_the_umi_stats_report(self) -> None:
        """Test that the note points the reader at umi_stats.txt for the run total."""
        report = make_stats().get_report()

        assert_that(report).contains("umi_stats.txt")
        assert_that(report).contains("r1_umi.fastq.gz")

    def test_denominator_note_says_the_fraction_is_not_the_library_fraction(self) -> None:
        """Test that the note explicitly denies the matched fraction is a library fraction.

        The denial is asserted over the whole note with its line wrapping
        normalised away, never over one physical line. What the note claims is
        the subject of this test; the column its prose happens to wrap at is
        not, and coupling the two would make rewrapping the note a test failure.
        """
        note = " ".join(self.denominator_note(make_stats().get_report()).lower().split())

        assert_that(note).contains("library")
        assert_that(DENIAL_PATTERN.search(note)).is_not_none()

    def test_denominator_note_precedes_the_counts(self) -> None:
        """Test that the caveat is read before the number it qualifies."""
        report = make_stats().get_report()

        assert_that(report.index("umi_stats.txt")).is_less_than(report.index(FIRST_COUNT_LINE))

    @staticmethod
    def denominator_note(report: str) -> str:
        """Return the denominator note alone, between the heading and the counts.

        Args:
            report: The full rendered report.

        Returns:
            The note text, excluding the section heading above it and the first
            count line below it.
        """
        return report.split(REPORT_HEADING)[1].split(FIRST_COUNT_LINE)[0]


class TestAssignStatsReportCounts:
    """The outcome counts and their percentages of total_reads."""

    @pytest.mark.parametrize(
        "expected_line",
        [
            "Total reads: 10",
            "Matched: 6 (60.00%)",
            "Unmatched (no_match): 2 (20.00%)",
            "Unmatched (no_left_anchor_pos): 1 (10.00%)",
            "Unmatched (short_window): 1 (10.00%)",
        ],
    )
    def test_report_renders_counts_against_total_reads(self, expected_line: str) -> None:
        """Test that every outcome count renders with its percentage of total_reads."""
        assert_that(make_stats().get_report()).contains(expected_line)

    def test_report_does_not_call_unmatched_reads_rejected(self) -> None:
        """Test that assignment wording is used: nothing here is rejected or dropped."""
        assert_that(make_stats().get_report()).does_not_contain("Rejected")

    def test_report_handles_zero_reads_without_error(self) -> None:
        """Test that a run with no reads renders rather than raising ZeroDivisionError."""
        report = make_empty_stats().get_report()

        assert_that(report).contains("Total reads: 0")
        assert_that(report).contains("Matched: 0 (0.00%)")
        assert_that(report).contains("# Target Distribution")
        assert_that(report).contains("# Edit Distance Distribution")


class TestAssignStatsReportDistributions:
    """The target, edit-distance and anchor-run distributions."""

    @pytest.mark.parametrize(
        "expected_entry",
        ["\tACGTACGT\t3 (50.00%)", "\tGGGGCCCC\t2 (33.33%)", "\tTTTTAAAA\t1 (16.67%)"],
    )
    def test_target_distribution_renders_against_matched(self, expected_entry: str) -> None:
        """Test that per-target counts render as a percentage of matched reads."""
        assert_that(make_stats().get_report()).contains(expected_entry)

    def test_target_distribution_is_sorted(self) -> None:
        """Test that targets appear in sorted order, not insertion order."""
        report = make_stats().get_report()

        assert_that(report.index("\tACGTACGT\t")).is_less_than(report.index("\tGGGGCCCC\t"))
        assert_that(report.index("\tGGGGCCCC\t")).is_less_than(report.index("\tTTTTAAAA\t"))

    @pytest.mark.parametrize(
        "expected_entry", ["\t0\t3 (50.00%)", "\t1\t2 (33.33%)", "\t2\t1 (16.67%)"]
    )
    def test_edit_distance_distribution_renders_against_matched(self, expected_entry: str) -> None:
        """Test that edit-distance counts render as a percentage of matched reads."""
        assert_that(make_stats().get_report()).contains(expected_entry)

    def test_edit_distance_distribution_is_sorted(self) -> None:
        """Test that edit distances appear in ascending order."""
        report = make_stats().get_report()
        section = report.split("# Edit Distance Distribution")[1]

        assert_that(section.index("\t0\t")).is_less_than(section.index("\t1\t"))
        assert_that(section.index("\t1\t")).is_less_than(section.index("\t2\t"))

    @pytest.mark.parametrize(
        "expected_entry", ["\t3\t6 (66.67%)", "\t4\t2 (22.22%)", "\t5\t1 (11.11%)"]
    )
    def test_run_length_distribution_renders_against_measured_reads(
        self, expected_entry: str
    ) -> None:
        """Test that run lengths render against reads_with_measured_run, not total_reads."""
        assert_that(make_stats().get_report()).contains(expected_entry)

    def test_run_length_distribution_is_sorted(self) -> None:
        """Test that run lengths appear in ascending order."""
        report = make_stats().get_report()
        section = report.split("-run Length Distribution")[1]

        assert_that(section.index("\t3\t")).is_less_than(section.index("\t4\t"))
        assert_that(section.index("\t4\t")).is_less_than(section.index("\t5\t"))

    def test_anchor_run_section_omitted_when_counter_empty(self) -> None:
        """Test that no anchor-run section is emitted when no run was ever measured."""
        report = make_stats(homopolymer_run_counts={}).get_report()

        assert_that(report).does_not_contain("-run Length Distribution")

    def test_anchor_run_heading_falls_back_when_base_unknown(self) -> None:
        """Test that a missing homopolymer base yields a sensible label, never 'None'."""
        report = make_stats(homopolymer_base=None).get_report()

        assert_that(report).contains("# Anchor homopolymer-run Length Distribution")
        assert_that(report).does_not_contain("None-run")


class TestAssignStatsMqcReporting:
    """MultiQC custom-content payloads: general stats, breakdown, target
    distribution, edit distance and anchor run.

    ``to_mqc_target_distribution`` is the odd one out among the three
    distribution payloads: it is never suppressed, so an empty run still gets a
    plot with an empty ``data`` mapping, where ``to_mqc_edit_distance`` and
    ``to_mqc_anchor_run`` return ``None`` on an empty counter. Those two ``None``
    checks are independent of one another, which the dedicated tests below pin
    directly rather than leaving to be inferred from the single-field defaults.
    """

    SAMPLE_PREFIX = "SK123"

    # ===== to_mqc_general_stats =====

    def test_to_mqc_general_stats_has_generalstats_plot_type_and_id(self) -> None:
        """Test that to_mqc_general_stats returns a generalstats payload with the expected id."""
        payload = make_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("generalstats")
        assert_that(payload["id"]).is_equal_to("carmack_tgidx_general_stats")

    def test_to_mqc_general_stats_computes_percentages(self) -> None:
        """Test that total_reads=10, matched=6, no_match=2, no_left_anchor_pos=1, short_window=1 render as 60/20/10/10."""
        payload = make_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_matched"]).is_equal_to(60.0)
        assert_that(data["pct_no_match"]).is_equal_to(20.0)
        assert_that(data["pct_no_left_anchor_pos"]).is_equal_to(10.0)
        assert_that(data["pct_short_window"]).is_equal_to(10.0)

    def test_to_mqc_general_stats_on_zero_reads_returns_zero_percentages(self) -> None:
        """Test that the zero-guarded fraction() helper keeps a zero-read run from raising."""
        payload = make_empty_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_matched"]).is_equal_to(0.0)
        assert_that(data["pct_no_match"]).is_equal_to(0.0)
        assert_that(data["pct_no_left_anchor_pos"]).is_equal_to(0.0)
        assert_that(data["pct_short_window"]).is_equal_to(0.0)

    def test_to_mqc_general_stats_attributes_its_columns_with_a_namespace(self) -> None:
        """Test that to_mqc_general_stats names carmack as its columns' source via ``namespace``."""
        payload = make_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload).contains_entry({"namespace": CARMACK_PARENT_NAME})
        assert_that(payload).does_not_contain_key("parent_id")
        assert_that(payload).does_not_contain_key("parent_name")

    # ===== to_mqc_breakdown =====

    def test_to_mqc_breakdown_has_bargraph_plot_type_and_parent(self) -> None:
        """Test that to_mqc_breakdown returns a bargraph payload naming carmack's shared parent section."""
        payload = make_stats().to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("bargraph")
        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    def test_to_mqc_breakdown_data_matches_outcome_counts(self) -> None:
        """Test that the breakdown data holds the raw matched/unmatched counts for the prefix."""
        payload = make_stats().to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(
            {"matched": 6, "no_match": 2, "no_left_anchor_pos": 1, "short_window": 1}
        )

    # ===== to_mqc_target_distribution =====

    def test_to_mqc_target_distribution_has_bargraph_plot_type(self) -> None:
        """Test that to_mqc_target_distribution returns a bargraph payload."""
        payload = make_stats().to_mqc_target_distribution(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("bargraph")

    def test_to_mqc_target_distribution_data_matches_target_counts_exactly(self) -> None:
        """Test that the payload mirrors target_counts verbatim, with no filtering or sorting applied."""
        payload = make_stats().to_mqc_target_distribution(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(dict(TARGET_COUNTS))

    def test_to_mqc_target_distribution_is_never_none_even_when_empty(self) -> None:
        """Test that an empty run still gets a target-distribution plot, unlike edit distance and anchor run."""
        payload = make_empty_stats().to_mqc_target_distribution(self.SAMPLE_PREFIX)

        assert_that(payload).is_not_none()
        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to({})

    # ===== to_mqc_edit_distance =====

    def test_to_mqc_edit_distance_has_linegraph_plot_type(self) -> None:
        """Test that to_mqc_edit_distance returns a linegraph payload when a distribution exists."""
        payload = make_stats().to_mqc_edit_distance(self.SAMPLE_PREFIX)

        assert_that(payload).is_not_none()
        assert_that(payload["plot_type"]).is_equal_to("linegraph")

    def test_to_mqc_edit_distance_data_matches_edit_distance_counts(self) -> None:
        """Test that the edit-distance data holds every count, as pairs ascending by x.

        The counts go in out of ascending order, so this also pins the sort:
        MultiQC reads a ``data`` mapping's keys as strings and re-sorts them
        lexically, and only the pair shape keeps the axis numeric.
        """
        payload = make_stats().to_mqc_edit_distance(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(EDIT_DISTANCE_PAIRS)

    def test_to_mqc_edit_distance_data_is_a_list_of_pairs_not_a_mapping(self) -> None:
        """Test that the data is a list by type, which is how MultiQC tells the shapes apart.

        MultiQC branches on ``isinstance(x_to_y[0], list)``, so a mapping -- or
        a list of tuples -- takes the string-keyed path instead, and the chart
        it draws is wrong rather than absent.
        """
        payload = make_stats().to_mqc_edit_distance(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data).is_instance_of(list)
        assert_that(data[0]).is_type_of(list)

    def test_to_mqc_edit_distance_returns_none_when_empty(self) -> None:
        """Test that a run with no edit-distance data returns None rather than an empty plot."""
        assert_that(make_empty_stats().to_mqc_edit_distance(self.SAMPLE_PREFIX)).is_none()

    # ===== to_mqc_anchor_run =====

    def test_to_mqc_anchor_run_has_linegraph_plot_type(self) -> None:
        """Test that to_mqc_anchor_run returns a linegraph payload when a run distribution exists."""
        payload = make_stats().to_mqc_anchor_run(self.SAMPLE_PREFIX)

        assert_that(payload).is_not_none()
        assert_that(payload["plot_type"]).is_equal_to("linegraph")

    def test_to_mqc_anchor_run_data_matches_run_counts(self) -> None:
        """Test that the anchor-run data holds the run-length distribution as ascending pairs.

        The run counts go in out of ascending order, so the ordering here comes
        from the payload builder's sort rather than from the order the fixture
        happened to list them in.
        """
        payload = make_stats().to_mqc_anchor_run(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(RUN_PAIRS)

    def test_to_mqc_anchor_run_data_is_a_list_of_pairs_not_a_mapping(self) -> None:
        """Test that the data is a list by type, not a mapping MultiQC would key by string."""
        payload = make_stats().to_mqc_anchor_run(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data).is_instance_of(list)
        assert_that(data[0]).is_type_of(list)

    def test_to_mqc_anchor_run_orders_double_digit_run_lengths_numerically(self) -> None:
        """Test that a run length of 10 or more sorts after 9, not between 1 and 2.

        Anchor runs are the distribution that reaches double digits in a real
        run, and a lexical sort puts 10, 11 and 12 immediately after 1. That is
        the zigzag the rendered report showed, so the regression needs a
        distribution that straddles the boundary and arrives shuffled.
        """
        stats = make_stats(homopolymer_run_counts={12: 1, 3: 40, 10: 4, 9: 11, 11: 2})

        payload = stats.to_mqc_anchor_run(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(
            [[3, 40], [9, 11], [10, 4], [11, 2], [12, 1]]
        )

    def test_to_mqc_anchor_run_returns_none_when_run_counts_are_empty(self) -> None:
        """Test that make_empty_stats' default homopolymer_run_counts ({}) yields None."""
        assert_that(make_empty_stats().to_mqc_anchor_run(self.SAMPLE_PREFIX)).is_none()

    # ===== Independence of the edit-distance and anchor-run None-checks =====

    def test_edit_distance_none_does_not_suppress_a_populated_anchor_run(self) -> None:
        """Test that an empty edit-distance counter does not blank out a populated anchor-run counter."""
        stats = make_stats(edit_distance_counts={}, homopolymer_run_counts=RUN_COUNTS)

        assert_that(stats.to_mqc_edit_distance(self.SAMPLE_PREFIX)).is_none()
        assert_that(stats.to_mqc_anchor_run(self.SAMPLE_PREFIX)).is_not_none()

    def test_anchor_run_none_does_not_suppress_a_populated_edit_distance(self) -> None:
        """Test that an empty anchor-run counter does not blank out a populated edit-distance counter."""
        stats = make_stats(edit_distance_counts=EDIT_DISTANCE_COUNTS, homopolymer_run_counts={})

        assert_that(stats.to_mqc_anchor_run(self.SAMPLE_PREFIX)).is_none()
        assert_that(stats.to_mqc_edit_distance(self.SAMPLE_PREFIX)).is_not_none()

    # ===== Shared prefix-keying contract =====

    @pytest.mark.parametrize("prefix", ["SK123", "another_sample_prefix"])
    def test_mqc_payloads_are_keyed_by_the_given_prefix(self, prefix: str) -> None:
        """Test that every to_mqc_* payload's data dict is keyed by exactly the prefix supplied."""
        stats = make_stats()

        general_stats_payload = stats.to_mqc_general_stats(prefix)
        breakdown_payload = stats.to_mqc_breakdown(prefix)
        target_distribution_payload = stats.to_mqc_target_distribution(prefix)
        edit_distance_payload = stats.to_mqc_edit_distance(prefix)
        anchor_run_payload = stats.to_mqc_anchor_run(prefix)

        assert_that(list(general_stats_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(breakdown_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(target_distribution_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(edit_distance_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(anchor_run_payload["data"].keys())).is_equal_to([prefix])


class TestAssignCountsDefaults:
    """The accumulator's starting state and its mutable-default contract."""

    @pytest.mark.parametrize("field_name", SCALAR_FIELD_NAMES)
    def test_default_scalar_tallies_start_at_zero(self, field_name: str) -> None:
        """Test that a fresh accumulator has counted nothing before any batch is folded in."""
        assert_that(getattr(AssignCounts(), field_name)).is_equal_to(0)

    @pytest.mark.parametrize("field_name", COUNTER_FIELD_NAMES)
    def test_default_counters_start_empty(self, field_name: str) -> None:
        """Test that a fresh accumulator's distributions start with no keys at all."""
        assert_that(getattr(AssignCounts(), field_name)).is_empty()

    @pytest.mark.parametrize("field_name,key", COUNTER_KEYS)
    def test_default_counters_are_not_shared_between_instances(
        self, field_name: str, key: object
    ) -> None:
        """Test that every accumulator owns its counters, so batches cannot alias.

        A mutable default shared between instances would make every batch's
        tallies the same object, and one batch's counts would show up in another's
        before anything had been folded.
        """
        first = AssignCounts()
        second = AssignCounts()

        getattr(first, field_name)[key] += 1

        assert_that(getattr(second, field_name)).is_empty()
        assert_that(getattr(first, field_name)).is_not_same_as(getattr(second, field_name))


class TestAssignCountsFold:
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
        deliberately overlap on every counter, so no replacing fold can satisfy
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

        totals.add(AssignCounts())

        assert_that(totals).is_equal_to(tally(READS))

    def test_folding_into_a_fresh_accumulator_yields_the_batch(self) -> None:
        """Test that the first fold of a run reproduces that batch's tallies exactly."""
        totals = AssignCounts()

        totals.add(tally(READS))

        assert_that(totals).is_equal_to(tally(READS))

    @pytest.mark.parametrize("order", list(permutations(range(len(PERMUTED_PARTITION)))))
    def test_fold_order_does_not_change_the_totals(self, order: tuple[int, ...]) -> None:
        """Test that the totals do not depend on which batch was folded first.

        This is what keeps the statistics independent of which worker happened to
        finish first.
        """
        batches = split(READS, PERMUTED_PARTITION)
        totals = AssignCounts()

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
        totals = AssignCounts()

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
        totals = AssignCounts()

        for batch in split(READS, sizes):
            totals.add(tally(batch))

        assert_that(totals).is_equal_to(tally(READS))


class TestAssignCountsToStats:
    """``to_stats``: handing the accumulated tallies off as the run's frozen value object."""

    @pytest.mark.parametrize("counts_field,stats_field", list(STATS_FIELD_NAMES.items()))
    def test_to_stats_maps_every_tally_onto_its_stats_field(
        self, counts_field: str, stats_field: str
    ) -> None:
        """Test that each tally arrives in the AssignStats field it is named for."""
        counts = tally(READS)

        stats = counts.to_stats(ANCHOR_BASE)

        assert_that(getattr(stats, stats_field)).is_equal_to(getattr(counts, counts_field))

    @pytest.mark.parametrize("base", [ANCHOR_BASE, None])
    def test_to_stats_carries_the_homopolymer_base_through(self, base: str | None) -> None:
        """Test that the chemistry's anchor base is passed to the stats unchanged."""
        stats = tally(READS).to_stats(base)

        assert_that(stats.homopolymer_base).is_equal_to(base)

    @pytest.mark.parametrize("stats_field", STATS_DICT_FIELD_NAMES)
    def test_to_stats_returns_plain_dicts_not_counters(self, stats_field: str) -> None:
        """Test that no Counter leaks into the frozen value object.

        Asserted on the exact type, because Counter is a dict subclass and an
        isinstance check would not catch the leak.
        """
        stats = tally(READS).to_stats(ANCHOR_BASE)

        assert_that(type(getattr(stats, stats_field)) is dict).is_true()

    @pytest.mark.parametrize("field_name", COUNTER_FIELD_NAMES)
    def test_reversing_the_reads_populates_the_counters_in_a_different_order(
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
        must render the same report. This is what proves the worker count cannot
        leak into the stats output.
        """
        forward = tally(READS).to_stats(ANCHOR_BASE).get_report()
        reverse = tally(list(reversed(READS))).to_stats(ANCHOR_BASE).get_report()

        assert_that(strip_report_run_details(reverse)).is_equal_to(
            strip_report_run_details(forward)
        )

    def test_to_stats_on_a_fresh_accumulator_renders_the_zero_report(self) -> None:
        """Test that a run that processed no reads renders the empty report, not an error."""
        report = AssignCounts().to_stats(None).get_report()

        assert_that(strip_report_run_details(report)).is_equal_to(
            strip_report_run_details(make_empty_stats().get_report())
        )
        assert_that(report).contains("Total reads: 0")
        assert_that(report).does_not_contain("-run Length Distribution")
