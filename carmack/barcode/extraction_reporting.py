"""
Data structures and logic for reporting statistics on barcode extraction results.
"""

from __future__ import annotations

import math
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
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME, linegraph_xy_pairs


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

        The payload's ``data`` is pair-shaped rather than a mapping; see
        :func:`carmack.mqc_report.linegraph_xy_pairs` for why.

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
            "data": {prefix: linegraph_xy_pairs(combined)},
        }


BARCODE_RANK_MAX_POINTS = 300


def log_spaced_ranks(n: int, max_points: int = BARCODE_RANK_MAX_POINTS) -> list[int]:
    """
    Choose which ranks to plot from a curve of ``n`` barcodes, spaced logarithmically.

    A rank curve is read on log axes, so the points that carry its shape have to be
    spaced logarithmically too. Thinning uniformly would spend almost the whole budget
    on the flat tail and leave the knee, the part anyone actually reads, drawn by a
    handful of points.

    The result holds at most ``max_points`` ranks rather than exactly that many. At the
    low end consecutive log steps are far less than one rank apart, so several of them
    round to the same integer rank and collapse into one. The budget is an upper bound,
    and the length must not be read back as a count of anything.

    Ranks 1 and ``n`` are seeded literally instead of being left to the loop's float
    arithmetic. They carry the two numbers a reader takes off this chart, the top
    barcode's depth and the number of barcodes observed, and ``round(exp(log(n)))``
    landing a rank either side of ``n`` would misreport the second of them silently.
    The interior is clamped into ``[2, n - 1]`` for the same reason: the bound then
    holds by construction rather than by trusting the exponential to stay inside it.

    A budget below two points cannot carry both endpoints at once, so there is no curve
    to draw and it raises. Truncating to a two-element list instead would quietly
    violate the very bound the argument was asking for.

    Args:
        n: Number of barcodes on the curve, which is also the largest rank available.
        max_points: Upper bound on how many ranks are returned. Must be at least 2
            whenever ``n`` exceeds it.

    Returns:
        Strictly ascending ranks starting at 1 and ending at ``n``, or an empty list
        when ``n`` is not positive.

    Raises:
        ValueError: If ``n`` exceeds ``max_points`` and ``max_points`` is below 2.
    """
    if n <= 0:
        return []
    if n <= max_points:
        return list(range(1, n + 1))
    if max_points < 2:
        raise ValueError(f"max_points must be at least 2 to keep both endpoints, got {max_points}")

    steps = max_points - 1
    log_n = math.log(n)
    ranks = {1, n}
    for i in range(1, max_points - 1):
        ranks.add(min(n - 1, max(2, round(math.exp(log_n * i / steps)))))
    return sorted(ranks)


def to_mqc_barcode_rank(
    prefix: str,
    barcode_counts: Mapping[str, int],
    max_points: int = BARCODE_RANK_MAX_POINTS,
) -> dict[str, object] | None:
    """
    Build a MultiQC "linegraph" custom-content payload of the barcode rank curve.

    This is a module-level function rather than a method on ExtractionStats because the
    per-barcode counts it plots live on the accumulator and never reach the finalized
    stats object, which keeps only the top ten. Threading the whole counter through
    ``finalize()`` purely so this builder could sit beside its siblings would be the
    larger change, and would hang a per-barcode dict off a dataclass that exists to
    stay small.

    Three details of the payload are load-bearing against MultiQC's custom-content
    parser rather than matters of taste.

    ``data`` is a list of ``[x, y]`` pairs and not an ``{x: y}`` mapping. The
    custom-content linegraph path renders a mapping's keys as lexically sorted strings,
    which puts '10' between '1' and '2' -- ruinous for a curve running from rank 1 to
    rank one million. A list of pairs is a first-class input shape MultiQC builds the
    mapping from itself, and it keeps the ranks integers throughout.

    An empty counter returns ``None`` rather than a payload carrying an empty pair
    list. MultiQC indexes the first point unguarded, so an empty list raises IndexError
    and takes down the entire report, not merely this section. The writer's own
    emptiness check does not catch it either, because ``{prefix: []}`` is a dict
    holding an empty list and so is truthy. The suppression has to happen here.

    ``smooth_points`` is set explicitly and derived as ``max_points + 1``. MultiQC
    re-bins any series longer than that threshold onto uniform index spacing, which
    flattens exactly the log spacing this chart exists to produce, and setting
    ``smooth_points`` to null does not disable the re-binning. Deriving the threshold
    from the budget in force means a caller raising ``max_points`` cannot silently walk
    back into it.

    Args:
        prefix: Sample identifier used to key the payload's ``data`` section.
        barcode_counts: Read count per full barcode. Zero-count entries are barcodes
            the run never observed and are excluded from the curve.
        max_points: Upper bound on how many ranks the curve is plotted at.

    Returns:
        MultiQC custom-content payload nested under carmack's shared parent section,
        holding one log-spaced rank/depth pair per plotted point, or None if no barcode
        was observed at all.

    Raises:
        ValueError: If more barcodes were observed than ``max_points`` and ``max_points``
            is below 2, raised by the downsampler, which cannot keep both endpoints.
    """
    counts = sorted((count for count in barcode_counts.values() if count > 0), reverse=True)
    if not counts:
        return None

    data = [[rank, counts[rank - 1]] for rank in log_spaced_ranks(len(counts), max_points)]
    return {
        "id": "carmack_extraction_barcode_rank",
        "plot_type": "linegraph",
        "parent_id": CARMACK_PARENT_ID,
        "parent_name": CARMACK_PARENT_NAME,
        "section_name": "Barcode Extraction Barcode Rank",
        "description": (
            "Read depth of every observed full barcode against its abundance rank, on log "
            "axes. The knee of the curve separates barcodes carrying real cells from the "
            "ambient background tail."
        ),
        "pconfig": {
            "id": "carmack_extraction_barcode_rank_plot",
            "title": "Barcode Extraction: Barcode Rank",
            "xlab": "Barcode rank",
            "ylab": "Reads",
            "xlog": True,
            "ylog": True,
            "smooth_points": max_points + 1,
        },
        "data": {prefix: data},
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
