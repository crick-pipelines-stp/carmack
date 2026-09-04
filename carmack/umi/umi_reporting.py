"""UMI stats value objects and report rendering for the extract-umis stage.

Holds the tallies that summarise a UMI extraction/correction run and render the
plain-text ``umi_stats.txt`` report, mirroring the barcode module's
``extraction_reporting``. The extraction stage populates the read counts and the
length / anchor-run distributions; the correction stage populates
:class:`CorrectionStats`.
"""

from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version


@dataclass(frozen=True)
class CorrectionStats:
    """Reconciling tallies for a UMI correction run.

    ``assigned_reads + dropped_raw_n + dropped_off_length`` equals the number of
    accepted reads passed to correction.

    Attributes:
        assigned_reads: Reads that survived filtering and received a ``UB``.
        corrections_applied: Reads whose ``UB`` differs from their own normalised
            raw UMI, i.e. reads that clustering reassigned to another
            representative. Length-normalisation alone does not count.
        num_cell_barcodes: Cell-barcode groups that contributed at least one
            correctable read.
        dropped_raw_n: Reads dropped because their raw UMI already contained the
            padding sentinel.
        dropped_off_length: Reads dropped because their raw UMI length fell
            outside the ``[x - tol, x + tol]`` window.
        distinct_corrected_umis: Number of distinct ``(barcode, UB)`` molecules
            surviving correction.
        umi_collapses: Number of distinct normalised UMIs merged into another
            representative by clustering.
    """

    assigned_reads: int
    corrections_applied: int
    num_cell_barcodes: int
    dropped_raw_n: int
    dropped_off_length: int
    distinct_corrected_umis: int
    umi_collapses: int

    @property
    def mean_reads_per_umi(self) -> float:
        """Return assigned reads per distinct corrected UMI.

        A pre-alignment duplication proxy (grouping is by cell barcode only, with
        no genomic position). Returns ``0.0`` when nothing was corrected.
        """
        if not self.distinct_corrected_umis:
            return 0.0
        return self.assigned_reads / self.distinct_corrected_umis


@dataclass(frozen=True)
class UmiExtractionStats:
    """Reconciling tallies for a UMI extraction run.

    Attributes:
        total_reads: Number of input reads processed.
        accepted: Reads with a successfully extracted UMI.
        missing_left_anchor: Reads skipped because the header carried no left
            anchor position tag.
        no_polyg_anchor: Reads skipped because no homopolymer anchor run began
            within the allowed length window.
        length_counts: Mapping of extracted UMI length to the number of accepted
            reads with that length.
        homopolymer_base: The homopolymer base of the chemistry's right anchor
            (e.g. ``"G"``), or ``None`` when that anchor is not a homopolymer.
        homopolymer_run_counts: Mapping of the observed right-anchor homopolymer
            run length to the number of accepted reads with that run.
        correction: The correction summary when a correction pass ran, or
            ``None`` under ``raw`` extraction.

    By construction ``accepted + missing_left_anchor + no_polyg_anchor`` always
    equals ``total_reads``.
    """

    total_reads: int
    accepted: int
    missing_left_anchor: int
    no_polyg_anchor: int
    length_counts: dict[int, int]
    homopolymer_base: str | None = None
    homopolymer_run_counts: dict[int, int] = field(default_factory=dict)
    correction: CorrectionStats | None = None

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the run.

        Returns:
            A plain-text report carrying the run details, the accepted / rejected
            counts (by reason), the UMI length distribution, the right-anchor
            homopolymer run-length distribution, and the correction summary when
            a correction pass ran.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        report = f"# Carmack version: {carmack_version}\n"
        report += f"# Report generated at: {run_time}\n"
        report += "# UMI Extraction Stats\n"
        report += f"Total reads: {self.total_reads}\n"
        report += (
            f"Accepted: {self.accepted} ({self.fraction(self.accepted, self.total_reads):.2%})\n"
        )
        report += (
            f"Rejected (missing_left_anchor): {self.missing_left_anchor} "
            f"({self.fraction(self.missing_left_anchor, self.total_reads):.2%})\n"
        )
        report += (
            f"Rejected (no_polyg_anchor): {self.no_polyg_anchor} "
            f"({self.fraction(self.no_polyg_anchor, self.total_reads):.2%})\n"
        )
        report += "\n# UMI Length Distribution\n"
        for length in sorted(self.length_counts):
            count = self.length_counts[length]
            report += f"\t{length}\t{count} ({self.fraction(count, self.accepted):.2%})\n"
        if self.homopolymer_run_counts:
            report += self.anchor_run_section()
        if self.correction is not None:
            report += self.correction_report()
        return report

    def anchor_run_section(self) -> str:
        """Render the right-anchor homopolymer run-length distribution.

        The anchor base is chemistry-defined, so the section is labelled with the
        actual base rather than hard-coding poly-G.
        """
        base = self.homopolymer_base or "homopolymer"
        section = f"\n# Anchor {base}-run Length Distribution\n"
        for run_length in sorted(self.homopolymer_run_counts):
            count = self.homopolymer_run_counts[run_length]
            section += f"\t{run_length}\t{count} ({self.fraction(count, self.accepted):.2%})\n"
        return section

    def correction_report(self) -> str:
        """Render the UMI correction section of the report."""
        correction = self.correction
        section = "\n# UMI Correction Stats\n"
        section += f"Cell barcodes (groups): {correction.num_cell_barcodes}\n"
        section += (
            f"Reads assigned UB: {correction.assigned_reads} "
            f"({self.fraction(correction.assigned_reads, self.accepted):.2%} of accepted)\n"
        )
        section += (
            f"Corrections applied (UB != raw): {correction.corrections_applied} "
            f"({self.fraction(correction.corrections_applied, correction.assigned_reads):.2%} of assigned)\n"
        )
        section += (
            f"Dropped (raw_N): {correction.dropped_raw_n} "
            f"({self.fraction(correction.dropped_raw_n, self.accepted):.2%})\n"
        )
        section += (
            f"Dropped (off_length): {correction.dropped_off_length} "
            f"({self.fraction(correction.dropped_off_length, self.accepted):.2%})\n"
        )
        section += f"Distinct corrected UMIs: {correction.distinct_corrected_umis}\n"
        section += f"Mean reads per UMI: {correction.mean_reads_per_umi:.2f}\n"
        section += f"UMI collapses: {correction.umi_collapses}\n"
        return section
