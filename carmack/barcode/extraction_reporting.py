"""
Data structures and logic for reporting statistics on barcode extraction results.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import product

from carmack import __version__ as carmack_version
from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.matchers.matcher_base import MatcherBase


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

    @classmethod
    def from_results(
        cls, results: list[ReadMatchResult], matchers: dict[MatchMethod, Mapping[str, MatcherBase]]
    ) -> ExtractionStats:
        """
        Create an ExtractionStats object by aggregating statistics from a list of ReadMatchResult.

        Args:
            results: List of ReadMatchResult objects, one for each read processed in the barcode extraction run.
            matchers: Dictionary mapping MatchMethod to the corresponding MatcherBase used for that method.
                This is used to determine which matching methods were applied to each barcode component across the reads.

        Returns:
            ExtractionStats object containing aggregated statistics for the barcode extraction run.
        """
        if not isinstance(results, list):
            raise TypeError(f"Expected list of ReadMatchResult, got {type(results)}")
        if not all(isinstance(r, ReadMatchResult) for r in results):
            raise TypeError("All items in results must be of type ReadMatchResult")
        if len(results) == 0:
            raise ValueError("Results list is empty")

        # Record overall stats
        overall_stats = OverallStats(
            total_reads=len(results),
            perfect=sum(1 for r in results if r.is_perfect),
            corrok=sum(1 for r in results if r.success and not r.is_perfect),
            fail=sum(1 for r in results if not r.success),
            top_10_barcodes=Counter(
                r.full_barcode for r in results if r.full_barcode is not None
            ).most_common(10),
        )

        # Per barcode stats
        per_barcode_stats: list[PerBarcodeStats] = []
        bc_names = [
            bc.name for bc in results[0].chemistry.read_structure.components if bc.is_barcode
        ]
        match_methods = matchers.keys()
        for bc_name, match_method in product(bc_names, match_methods):
            # Get all attempts for this barcode component and method across all reads
            match_attempts: list[BarcodeMatchAttempt] = []
            reads_attempted: int = 0
            reads_w_ambiguous_match: int = 0

            reads_success: int = 0
            reads_fail: int = 0

            for r in results:
                r_attempts = r.get_attempts(bc_name, match_method)
                match_attempts.extend(r_attempts)

                if len(r_attempts) > 0:
                    reads_attempted += 1
                    if any(a.match is not None for a in r_attempts):
                        reads_success += 1
                    else:
                        reads_fail += 1
                if len(r_attempts) > 1:
                    reads_w_ambiguous_match += 1

            # Check if match_method was used for this barcode component in any read, regardless of success
            if len(match_attempts) == 0:
                continue
            edit_distances = [
                a.edit_distance for a in match_attempts if a.edit_distance is not None
            ]
            edit_distance_dist = Counter(edit_distances) if edit_distances else None

            stat = PerBarcodeStats(
                bc_name=bc_name,
                method=match_method,
                attempts=reads_attempted,  # Total reads attempted with this method
                success=reads_success,
                fail=reads_fail,
                edit_distance_dist=edit_distance_dist,
                reads_w_ambiguous_match=reads_w_ambiguous_match,
                spacer_present=sum(
                    1 for a in match_attempts if any([a.spacer_upstream, a.spacer_downstream])
                ),
            )
            per_barcode_stats.append(stat)

        return cls(overall=overall_stats, per_barcode=per_barcode_stats, bc_names=bc_names)

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
