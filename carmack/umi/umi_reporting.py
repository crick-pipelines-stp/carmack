"""UMI stats value objects and report rendering for the extract-umis stage.

Holds the tallies that summarise a UMI extraction run and render the plain-text
``umi_stats.txt`` report, mirroring the barcode module's
``extraction_reporting``.
"""

from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version


@dataclass(frozen=True)
class UmiExtractionStats:
    """Reconciling tallies for a UMI extraction run.

    Attributes:
        total_reads: Number of input reads processed.
        accepted: Reads with a UMI extracted.
        missing_left_anchor: Reads skipped because the header carried no position
            tag for the anchor the UMI is measured from.
        truncated: Reads skipped because the read ended before the UMI did.
        umi_length: The fixed UMI length taken from the chemistry. Reported
            rather than measured, since every accepted read yields exactly this
            many bases.
        homopolymer_base: The homopolymer base of the anchor 3' of the UMI (e.g.
            ``"G"``), or ``None`` when the UMI has no homopolymer neighbour.
        homopolymer_run_counts: Mapping of the anchor homopolymer run length
            observed at the first base after the UMI to the number of accepted
            reads with that run.

    By construction ``accepted + missing_left_anchor + truncated`` always equals
    ``total_reads``. Where ``homopolymer_base`` is not ``None`` a second
    invariant holds: every accepted read is measured, so
    ``sum(homopolymer_run_counts.values())`` equals ``accepted`` and ``accepted``
    is the section's denominator.
    """

    total_reads: int
    accepted: int
    missing_left_anchor: int
    truncated: int
    umi_length: int
    homopolymer_base: str | None = None
    homopolymer_run_counts: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the run.

        Returns:
            A plain-text report carrying the run details, the fixed UMI length,
            the accepted / rejected counts by reason, and the anchor homopolymer
            run-length distribution.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        report = f"# Carmack version: {carmack_version}\n"
        report += f"# Report generated at: {run_time}\n"
        report += "# UMI Extraction Stats\n"
        report += self.length_note()
        report += f"Total reads: {self.total_reads}\n"
        report += (
            f"Accepted: {self.accepted} ({self.fraction(self.accepted, self.total_reads):.2%})\n"
        )
        report += (
            f"Rejected (missing_left_anchor): {self.missing_left_anchor} "
            f"({self.fraction(self.missing_left_anchor, self.total_reads):.2%})\n"
        )
        report += (
            f"Rejected (truncated): {self.truncated} "
            f"({self.fraction(self.truncated, self.total_reads):.2%})\n"
        )
        if self.homopolymer_run_counts:
            report += self.anchor_run_section()
        return report

    def length_note(self) -> str:
        """Render the note explaining why no UMI length distribution is reported.

        The UMI is a fixed slice, so its length is asserted rather than measured
        and a distribution over it would be a single bin at 100%. Stating the
        constant here says the same thing honestly, and pre-empts a reader
        looking for the distribution an earlier version of this report carried.
        """
        return (
            f"# The UMI is the fixed {self.umi_length} bases following the left anchor, so its\n"
            "# length is not a measurement and no length distribution is reported.\n"
        )

    def anchor_run_section(self) -> str:
        """Render the anchor homopolymer run-length distribution.

        The anchor base is chemistry-defined, so the section is labelled with the
        actual base rather than hard-coding poly-G. Nothing in this stage gates
        on the run, so the section is purely a check that the layout is holding:
        the UMI boundary is asserted from the chemistry rather than found in the
        read, and mass in the zero bin is the only signal that the assertion is
        wrong for this library.
        """
        base = self.homopolymer_base or "homopolymer"
        section = f"\n# Anchor {base}-run Length Distribution\n"
        section += (
            "# Measured from the first base after the UMI, over accepted reads. A zero is a\n"
            "# read whose anchor run does not begin where the chemistry says it does.\n"
        )
        for run_length in sorted(self.homopolymer_run_counts):
            count = self.homopolymer_run_counts[run_length]
            section += f"\t{run_length}\t{count} ({self.fraction(count, self.accepted):.2%})\n"
        return section
