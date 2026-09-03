"""Tests for the assign-targets reporting module.

``AssignStats`` is the reconciling value object for a target-index assignment
run: every input read lands in exactly one of ``matched``,
``unmatched_no_match``, ``unmatched_no_umi_pos`` or ``unmatched_short_window``,
and the same object renders the plain-text ``tgidx_stats.txt`` report. These
tests pin the value object's contract (frozen, safe division, the derived
run-length denominator) and the report's shape: the volatile run-detail header
that ``strip_report_run_details`` must keep working against, every section
heading, the counts and their percentages, the three distributions with their
stated denominators, and the denominator note that stops the matched fraction
being read as a modality fraction of the library.
"""

import dataclasses
import re

import pytest
from assertpy import assert_that

from carmack import __version__ as carmack_version
from carmack.assign_targets.assign_reporting import AssignStats
from tests.utils import strip_report_run_details

# A whitelist entry seen three times, one seen twice, one seen once. They are
# deliberately supplied out of lexical order so the report's sorting is tested.
TARGET_COUNTS = {"GGGGCCCC": 2, "ACGTACGT": 3, "TTTTAAAA": 1}
EDIT_DISTANCE_COUNTS = {2: 1, 0: 3, 1: 2}
RUN_COUNTS = {5: 1, 3: 6, 4: 2}

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
        "unmatched_no_umi_pos": 1,
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
        unmatched_no_umi_pos=0,
        unmatched_short_window=0,
        target_counts={},
        edit_distance_counts={},
    )


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
            + stats.unmatched_no_umi_pos
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
            "Unmatched (no_umi_pos): 1 (10.00%)",
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
