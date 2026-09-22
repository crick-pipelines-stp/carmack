"""Linear-dedup stats value object for the linear-dedup stage.

Holds the tallies that summarise a linear-dedup run, mirroring the UMI
module's ``UmiExtractionStats``.
"""

from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME


@dataclass(frozen=True)
class LinearDedupStats:
    """Reconciling tallies for a linear-dedup run's first pass.

    Attributes:
        total_pairs: Number of R1 records scanned.
        eligible_pairs: R1 records that were primary, paired, mapped (with a
            mapped mate), and carried the configured barcode tag -- these are
            the reads grouped into fragment keys and scored.
        skipped_unmapped: R1 records skipped because the R1 itself, or its
            mate, was unmapped.
        skipped_non_primary: R1 records skipped because they were secondary
            or supplementary alignments.
        skipped_unpaired: R1 records skipped because they were not paired.
        reads_missing_as: Eligible R1 records with no ``AS`` tag, scored as
            lowest priority.
        eligible_pairs_by_chromosome: Per-chromosome breakdown of
            ``eligible_pairs``.
        pairs_kept_by_chromosome: Per-chromosome breakdown of the winning
            fragment-key groups resolved from the eligible reads.

    By construction ``eligible_pairs + skipped_unmapped + skipped_non_primary
    + skipped_unpaired`` always equals ``total_pairs``. There is no dedicated
    "pairs kept" field: it is ``sum(pairs_kept_by_chromosome.values())``, and
    "pairs removed" (the duplication count) is ``eligible_pairs`` minus that
    sum -- both are derived wherever they are reported rather than stored a
    second time.
    """

    total_pairs: int
    eligible_pairs: int
    skipped_unmapped: int
    skipped_non_primary: int
    skipped_unpaired: int
    reads_missing_as: int
    eligible_pairs_by_chromosome: dict[str, int] = field(default_factory=dict)
    pairs_kept_by_chromosome: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    def pairs_kept(self) -> int:
        """Return the total number of winning fragment-key groups across all chromosomes."""
        return sum(self.pairs_kept_by_chromosome.values())

    def pairs_removed(self) -> int:
        """Return the number of eligible pairs removed as duplicates."""
        return self.eligible_pairs - self.pairs_kept()

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the run.

        Returns:
            A plain-text report carrying the run details and the pass-1
            counts by outcome, each shown alongside its percentage of the
            appropriate denominator: eligible/skipped counts are a fraction
            of ``total_pairs``, while pairs kept/removed and reads missing
            ``AS`` are a fraction of ``eligible_pairs``.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        pairs_kept = self.pairs_kept()
        pairs_removed = self.pairs_removed()

        report = f"# Carmack version: {carmack_version}\n"
        report += f"# Report generated at: {run_time}\n"
        report += "# Linear Dedup Stats\n"
        report += f"Total pairs: {self.total_pairs}\n"
        report += (
            f"Eligible: {self.eligible_pairs} "
            f"({self.fraction(self.eligible_pairs, self.total_pairs):.2%})\n"
        )
        report += (
            f"Skipped (unmapped): {self.skipped_unmapped} "
            f"({self.fraction(self.skipped_unmapped, self.total_pairs):.2%})\n"
        )
        report += (
            f"Skipped (non_primary): {self.skipped_non_primary} "
            f"({self.fraction(self.skipped_non_primary, self.total_pairs):.2%})\n"
        )
        report += (
            f"Skipped (unpaired): {self.skipped_unpaired} "
            f"({self.fraction(self.skipped_unpaired, self.total_pairs):.2%})\n"
        )
        report += (
            f"Pairs kept: {pairs_kept} "
            f"({self.fraction(pairs_kept, self.eligible_pairs):.2%})\n"
        )
        report += (
            f"Pairs removed (duplicates): {pairs_removed} "
            f"({self.fraction(pairs_removed, self.eligible_pairs):.2%})\n"
        )
        report += (
            f"Missing AS: {self.reads_missing_as} "
            f"({self.fraction(self.reads_missing_as, self.eligible_pairs):.2%})\n"
        )
        return report

    def to_mqc_general_stats(self, prefix: str) -> dict[str, object]:
        """Build a MultiQC "generalstats" custom-content payload summarising this run.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload with a single row of headline percentages
            (duplication rate, missing-AS rate) for this sample. ``namespace`` is
            what attributes those columns to Carmack: the custom-content parser
            branches on the generalstats plot type and returns before it reads
            ``parent_id``, so the parent keys that nest this stage's chart sections
            are inert here, and a namespace left unset falls back to the raw
            payload id.
        """
        pct_duplication = 100 * self.fraction(self.pairs_removed(), self.eligible_pairs)
        pct_missing_as = 100 * self.fraction(self.reads_missing_as, self.eligible_pairs)

        return {
            "id": "carmack_linear_dedup_general_stats",
            "plot_type": "generalstats",
            "namespace": CARMACK_PARENT_NAME,
            "pconfig": [
                {
                    "pct_duplication": {
                        "title": "% Duplication",
                        "description": "Percentage of eligible pairs removed as duplicates.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlOrRd",
                    }
                },
                {
                    "pct_missing_as": {
                        "title": "% Missing AS",
                        "description": "Percentage of eligible pairs with no AS (alignment score) tag.",
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
                    "pct_duplication": pct_duplication,
                    "pct_missing_as": pct_missing_as,
                }
            },
        }

    def to_mqc_breakdown(self, prefix: str) -> dict[str, object]:
        """Build a MultiQC "bargraph" custom-content payload of raw outcome counts.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload nested under carmack's shared parent section,
            with one bar per sample split into kept/removed/skipped pair counts.
        """
        return {
            "id": "carmack_linear_dedup_breakdown",
            "plot_type": "bargraph",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "section_name": "Linear Dedup Breakdown",
            "description": "Read pair counts broken down by linear-dedup outcome.",
            "pconfig": {
                "id": "carmack_linear_dedup_breakdown_plot",
                "title": "Linear Dedup: Outcomes",
                "ylab": "Pairs",
            },
            "data": {
                prefix: {
                    "kept": self.pairs_kept(),
                    "removed": self.pairs_removed(),
                    "skipped_unmapped": self.skipped_unmapped,
                    "skipped_non_primary": self.skipped_non_primary,
                    "skipped_unpaired": self.skipped_unpaired,
                }
            },
        }

    def to_mqc_chromosome_breakdown(self, prefix: str) -> dict[str, object] | None:
        """Build a MultiQC "bargraph" custom-content payload of duplicates removed per chromosome.

        Args:
            prefix: Sample identifier used to key the payload's ``data`` section.

        Returns:
            MultiQC custom-content payload with one bar per chromosome, each the
            number of eligible pairs on that chromosome minus the number kept
            there, or None when ``eligible_pairs_by_chromosome`` is empty (nothing
            to report).
        """
        if not self.eligible_pairs_by_chromosome:
            return None

        removed_by_chromosome = {
            chrom: eligible - self.pairs_kept_by_chromosome.get(chrom, 0)
            for chrom, eligible in self.eligible_pairs_by_chromosome.items()
        }

        return {
            "id": "carmack_linear_dedup_chromosome_breakdown",
            "plot_type": "bargraph",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "section_name": "Linear Dedup: Duplicates Removed per Chromosome",
            "description": "Duplicate pairs removed per chromosome.",
            "pconfig": {
                "id": "carmack_linear_dedup_chromosome_breakdown_plot",
                "title": "Linear Dedup: Duplicates Removed per Chromosome",
                "ylab": "Duplicate pairs removed",
            },
            "data": {prefix: removed_by_chromosome},
        }
