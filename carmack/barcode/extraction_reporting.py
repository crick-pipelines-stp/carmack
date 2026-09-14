"""
Data structures and logic for reporting statistics on barcode extraction results.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version
from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.read_component import ReadComponentType
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME


@dataclass(frozen=True)
class OverallStats:
    """Overall read statistics for a barcode extraction run."""

    total_reads: int
    perfect: int
    corrok: int  # Corrected
    fail: int

    top_10_barcodes: list[tuple[str, int]]


@dataclass(frozen=True)
class PerBarcodeStats:
    """Statistics for a specific barcode component and matching method."""

    bc_name: str
    method: MatchMethod
    attempts: int
    success: int
    fail: int
    edit_distance_dist: Counter[int] | None
    reads_w_ambiguous_match: int  # Number of reads where this method produced an ambiguous match (multiple candidates tied)
    spacer_present: int  # Number of reads where at least one spacer was present (if checked for)


@dataclass(frozen=True)
class ExtractionStats:
    """
    Aggregated statistics for a barcode extraction run, including overall stats and per-barcode
    component stats.

    Attributes:
        overall: OverallStats object summarizing total reads, perfect matches, corrected matches, failed matches.
        per_barcode: List of PerBarcodeStats objects, one for each barcode component and matching method.
        bc_names: List of barcode component names included in the report.
    """

    overall: OverallStats
    per_barcode: list[PerBarcodeStats]
    bc_names: list[str]

    def get_report(self, include_run_details: bool = True) -> str:
        """
        Generate a human-readable report summarising the barcode extraction statistics.

        Args:
            include_run_details: Whether to include details about the run (Carmack version, report
            generation time) at the top of the report.

        Returns:
            Formatted string report summarising overall and per-barcode extraction statistics.
        """
        report = ""
        if include_run_details:
            report += self.get_run_details() + "\n"
        report += "# Overall Barcode Extraction Stats\n"
        report += f"Total reads: {self.overall.total_reads}\n"
        report += f"Perfect matches: {self.overall.perfect} ({self.overall.perfect / self.overall.total_reads:.2%})\n"
        report += f"Corrected matches: {self.overall.corrok} ({self.overall.corrok / self.overall.total_reads:.2%})\n"
        report += f"Failed matches: {self.overall.fail} ({self.overall.fail / self.overall.total_reads:.2%})\n"
        report += "Top 10 barcode fractions:\n"
        for barcode, count in self.overall.top_10_barcodes:
            report += f"\t{barcode}\t{count / self.overall.total_reads:.2}\n"

        # Add per-barcode stats
        report += "\n# Per-Barcode Component Stats\n"

        # Sort by method for consistent reporting
        method_order = {member: i for i, member in enumerate(MatchMethod)}

        for bc_name in self.bc_names:
            bc_stats_list = [s for s in self.per_barcode if s.bc_name == bc_name]

            if not bc_stats_list:
                continue

            bc_stats_list.sort(key=lambda x: method_order[x.method])

            report += f"\n## {bc_name}\n"
            for s in bc_stats_list:
                report += "\n"
                report += self.get_per_barcode_section(s)

        return report

    def get_per_barcode_section(self, bc_stats: PerBarcodeStats) -> str:
        """
        Get a formatted report section for a specific barcode component and matching method.

        Args:
            bc_stats: PerBarcodeStats object containing statistics for a specific barcode component and matching method.

        Returns:
            Formatted string summarising the statistics for this barcode component and method.
        """
        section = f"Matching method: {bc_stats.method.value}\n"
        section += f"Reads checked: {bc_stats.attempts}\n"
        section += f"Successful matches: {bc_stats.success} ({bc_stats.success / bc_stats.attempts:.2%})\n"
        section += f"Failed matches: {bc_stats.fail} ({bc_stats.fail / bc_stats.attempts:.2%})\n"
        if bc_stats.edit_distance_dist is not None:
            section += "Edit distance distribution (distance - count):\n"
            for ed, count in sorted(bc_stats.edit_distance_dist.items()):
                section += f"\t{ed}\t{count} ({count / bc_stats.success:.2%})\n"
        section += f"Ambiguous matches (reads): {bc_stats.reads_w_ambiguous_match} ({bc_stats.reads_w_ambiguous_match / bc_stats.attempts:.2%})\n"
        if bc_stats.spacer_present > 0:
            section += f"Matches with spacers present: {bc_stats.spacer_present}\n"
        return section

    def get_run_details(self) -> str:
        """
        Get details about the barcode extraction run, including Carmack version and report generation time.

        Returns:
            Formatted string with run details.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        details = f"# Carmack version: {carmack_version}\n"
        details += f"# Report generated at: {run_time}\n"
        return details

    def to_mqc_general_stats(self, prefix: str) -> dict[str, object]:
        """
        Build a MultiQC "generalstats" custom-content payload summarising this run.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload with a single row of headline percentages
            (perfect, corrected, failed, ambiguous) for this sample. ``namespace`` is
            what attributes those columns to Carmack: the custom-content parser
            branches on the generalstats plot type and returns before it reads
            ``parent_id``, so the parent keys that nest this stage's chart sections are
            inert here, and a namespace left unset falls back to the raw payload id.
        """
        pct_perfect = 100 * self.overall.perfect / self.overall.total_reads
        pct_corrected = 100 * self.overall.corrok / self.overall.total_reads
        pct_failed = 100 * self.overall.fail / self.overall.total_reads
        pct_ambiguous = (
            100
            * sum(s.reads_w_ambiguous_match for s in self.per_barcode)
            / self.overall.total_reads
        )

        return {
            "id": "carmack_extraction_general_stats",
            "plot_type": "generalstats",
            "namespace": CARMACK_PARENT_NAME,
            "pconfig": [
                {
                    "pct_perfect": {
                        "title": "% Perfect",
                        "description": "Percentage of reads where every barcode component matched the whitelist exactly.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "RdYlGn",
                    }
                },
                {
                    "pct_corrected": {
                        "title": "% Corrected",
                        "description": "Percentage of reads where at least one barcode component required error correction.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlGnBu",
                    }
                },
                {
                    "pct_failed": {
                        "title": "% Failed",
                        "description": "Percentage of reads that failed to match at least one barcode component.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlOrRd",
                    }
                },
                {
                    "pct_ambiguous": {
                        "title": "% Ambiguous",
                        "description": "Percentage of reads with at least one ambiguous (tied) barcode match.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlOrRd",
                    }
                },
            ],
            "data": {
                prefix: {
                    "pct_perfect": pct_perfect,
                    "pct_corrected": pct_corrected,
                    "pct_failed": pct_failed,
                    "pct_ambiguous": pct_ambiguous,
                }
            },
        }

    def to_mqc_breakdown(self, prefix: str) -> dict[str, object]:
        """
        Build a MultiQC "bargraph" custom-content payload of raw match-outcome counts.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload nested under carmack's shared parent section,
            with one bar per sample split into perfect/corrected/failed read counts.
        """
        return {
            "id": "carmack_extraction_breakdown",
            "plot_type": "bargraph",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "section_name": "Barcode Extraction Breakdown",
            "description": "Read counts broken down by barcode extraction outcome.",
            "pconfig": {
                "id": "carmack_extraction_breakdown_plot",
                "title": "Barcode Extraction: Match Outcomes",
                "ylab": "Reads",
            },
            "data": {
                prefix: {
                    "perfect": self.overall.perfect,
                    "corrected": self.overall.corrok,
                    "failed": self.overall.fail,
                }
            },
        }

    def to_mqc_edit_distance(self, prefix: str) -> dict[str, object] | None:
        """
        Build a MultiQC "linegraph" custom-content payload of the combined edit distance distribution.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload with the edit distance distribution summed
            across every barcode component and matching method, or None if none of the
            per-barcode entries carry any edit distance data.
        """
        combined: Counter[int] = Counter()
        for bc_stats in self.per_barcode:
            if bc_stats.edit_distance_dist:
                combined.update(bc_stats.edit_distance_dist)

        if not combined:
            return None

        return {
            "id": "carmack_extraction_edit_distance",
            "plot_type": "linegraph",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "section_name": "Barcode Extraction Edit Distance",
            "description": "Distribution of edit distances for corrected barcode matches, summed across all barcode components.",
            "pconfig": {
                "id": "carmack_extraction_edit_distance_plot",
                "title": "Barcode Extraction: Edit Distance Distribution",
                "xlab": "Edit distance",
                "ylab": "Reads",
            },
            "data": {prefix: dict(combined)},
        }


@dataclass
class BarcodeMethodCounters:
    """Running per-(barcode, method) tallies used by ExtractionStatsAccumulator."""

    reads_attempted: int = 0
    reads_success: int = 0
    reads_fail: int = 0
    reads_w_ambiguous_match: int = 0
    spacer_present: int = 0
    edit_distance_counter: Counter[int] = field(default_factory=Counter)


class ExtractionStatsAccumulator:
    """
    Incremental aggregator for ``ExtractionStats``.

    Consumes one ``ReadMatchResult`` at a time so callers can stream results to
    disk without holding the full list in memory. Call ``update`` per result
    and ``finalize`` once to obtain the equivalent ``ExtractionStats``.

    ``full_barcode_counts`` is exposed so callers can reuse it for
    ``bc_counts`` / ``bc_rank`` outputs without re-walking results.
    """

    def __init__(self, matchers: Mapping[MatchMethod, Mapping[str, MatcherBase]]) -> None:
        self.match_methods: list[MatchMethod] = list(matchers.keys())
        self.bc_names: list[str] = []
        self.chemistry_seen = False

        self.total_reads = 0
        self.perfect = 0
        self.corrok = 0
        self.fail = 0
        self.full_barcode_counts: Counter[str] = Counter()

        self.per_bc: dict[tuple[str, MatchMethod], BarcodeMethodCounters] = {}

    def update(self, result: ReadMatchResult) -> None:
        """Fold a single ``ReadMatchResult`` into the running aggregates."""
        if not self.chemistry_seen:
            self.bc_names = [
                bc.name
                for bc in result.chemistry.read_structure.get_components_by_type(
                    ReadComponentType.BARCODE
                )
            ]
            self.chemistry_seen = True

        self.total_reads += 1
        if result.is_perfect:
            self.perfect += 1
        elif result.success:
            self.corrok += 1
        else:
            self.fail += 1

        full_bc = result.full_barcode
        if full_bc is not None:
            self.full_barcode_counts[full_bc] += 1

        for bc_history in result.bc_results:
            attempts_by_method: dict[MatchMethod, list[BarcodeMatchAttempt]] = {}
            for attempt in bc_history.attempts:
                attempts_by_method.setdefault(attempt.method, []).append(attempt)

            for method, attempts in attempts_by_method.items():
                state = self.per_bc.setdefault(
                    (bc_history.bc_name, method), BarcodeMethodCounters()
                )
                state.reads_attempted += 1
                if any(a.match is not None for a in attempts):
                    state.reads_success += 1
                else:
                    state.reads_fail += 1
                if len(attempts) > 1:
                    state.reads_w_ambiguous_match += 1
                for a in attempts:
                    if a.edit_distance is not None:
                        state.edit_distance_counter[a.edit_distance] += 1
                    if a.spacer_upstream or a.spacer_downstream:
                        state.spacer_present += 1

    def finalize(self) -> ExtractionStats:
        """Materialise the accumulated state as an ``ExtractionStats``."""
        if self.total_reads == 0:
            raise ValueError("No results were accumulated")

        overall = OverallStats(
            total_reads=self.total_reads,
            perfect=self.perfect,
            corrok=self.corrok,
            fail=self.fail,
            top_10_barcodes=self.full_barcode_counts.most_common(10),
        )

        # Iterate (bc_name, method) in the same order from_results used:
        # outer loop over barcode components, inner over registered methods.
        per_barcode: list[PerBarcodeStats] = []
        for bc_name in self.bc_names:
            for method in self.match_methods:
                state = self.per_bc.get((bc_name, method))
                if state is None or state.reads_attempted == 0:
                    continue
                per_barcode.append(
                    PerBarcodeStats(
                        bc_name=bc_name,
                        method=method,
                        attempts=state.reads_attempted,
                        success=state.reads_success,
                        fail=state.reads_fail,
                        edit_distance_dist=(
                            state.edit_distance_counter if state.edit_distance_counter else None
                        ),
                        reads_w_ambiguous_match=state.reads_w_ambiguous_match,
                        spacer_present=state.spacer_present,
                    )
                )

        return ExtractionStats(overall=overall, per_barcode=per_barcode, bc_names=self.bc_names)
