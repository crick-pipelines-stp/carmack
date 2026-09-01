"""Raw UMI extraction from annotated R1 FASTQ reads (SI-3 of the UMI epic).

Step 1 of UMI handling. For each annotated read the extractor locates the raw
UMI lying between its left anchor (the BC1 barcode, read from the header
``BC1_POS`` tag) and the downstream poly-G run, annotates the read with ``UMI``
and ``UMI_POS`` tags, and records the observed UMI length distribution.

Extraction is a single streaming pass with no correction: the raw UMI is neither
normalised nor padded and no ``UB`` tag is emitted. The ``raw`` flag threaded
through :meth:`UmiExtractor.extract_umis` documents that this is the terminal
step so that a later correction stage can gate on ``not raw``.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from carmack import __version__ as carmack_version
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.utils import get_prefix

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UmiExtractionStats:
    """Reconciling tallies for a UMI extraction run.

    Attributes:
        total_reads: Number of input reads processed.
        accepted: Reads with a successfully extracted UMI.
        missing_left_anchor: Reads skipped because the header carried no left
            anchor position tag.
        no_polyg_anchor: Reads skipped because no poly-G run began within the
            allowed length window.
        length_counts: Mapping of extracted UMI length to the number of accepted
            reads with that length.

    By construction ``accepted + missing_left_anchor + no_polyg_anchor`` always
    equals ``total_reads``.
    """

    total_reads: int
    accepted: int
    missing_left_anchor: int
    no_polyg_anchor: int
    length_counts: dict[int, int]

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the extraction run.

        Returns:
            A plain-text report carrying the run details, overall accepted /
            rejected counts (by reason) and the UMI length distribution.
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
        return report


class UmiExtractor:
    """Extracts raw UMIs from an annotated R1 FASTQ using its chemistry layout.

    The extractor is a thin wrapper mirroring the barcode extractor pattern: it
    resolves the chemistry, validates that a UMI can be anchored on both sides,
    and streams the reads once to write the annotated UMI FASTQ and stats report.
    """

    def __init__(self, fastq_file: str, chemistry_name: str) -> None:
        """Resolve the chemistry and derive the UMI extraction parameters.

        Args:
            fastq_file: Path to the annotated R1 FASTQ (produced by barcode
                extraction, carrying ``BC1_POS`` header tags).
            chemistry_name: Name of the chemistry describing the read layout.

        Raises:
            ValueError: If the chemistry is unknown, cannot anchor a UMI on its
                5' side, or has no downstream poly-G anchor.
        """
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_umi_extraction():
            raise ValueError(
                f"chemistry '{chemistry_name}' has no usable left anchor for UMI extraction"
            )

        left = self.chemistry.umi_left_anchor()
        umi = self.chemistry.umi_component()
        right = self.chemistry.umi_right_anchor()

        if right is None:
            raise ValueError(
                f"chemistry '{chemistry_name}' has no poly-G anchor for UMI extraction"
            )

        self.umi_name = umi.name
        self.umi_length = umi.length
        self.umi_length_tolerance = umi.length_tolerance
        self.polyg_base = right.homopolymer_base
        self.polyg_min_run = right.min_run
        self.left_key = position_key(left.name)

    def find_polyg_start(self, seq: str, umi_start: int) -> int | None:
        """Return the greedy earliest poly-G run start bounding the UMI.

        The UMI length must lie in ``[x - tol, x + tol]``, so the poly-G run may
        begin only at offsets ``[umi_start + (x - tol), umi_start + (x + tol)]``
        inclusive. Offsets are scanned from the smallest upward and the first
        position ``p`` where ``min_run`` copies of the poly-G base begin is
        returned. This naturally enforces the length window: runs beginning
        before the window make the UMI too short and are never scanned, while
        runs beginning after it make the UMI too long and are likewise skipped.

        Args:
            seq: The read sequence.
            umi_start: 0-based index at which the UMI begins (the BC1 span end).

        Returns:
            The poly-G run start index, or ``None`` when no qualifying run begins
            within the window.
        """
        run = self.polyg_base * self.polyg_min_run
        min_start = umi_start + (self.umi_length - self.umi_length_tolerance)
        max_start = umi_start + (self.umi_length + self.umi_length_tolerance)
        for polyg_start in range(min_start, max_start + 1):
            end = polyg_start + self.polyg_min_run
            if end <= len(seq) and seq[polyg_start:end] == run:
                return polyg_start
        return None

    def extract_umis(
        self,
        output_dir: str = ".",
        prefix: str | None = None,
        raw: bool = False,
    ) -> UmiExtractionStats:
        """Stream the reads, annotate extracted UMIs and write the output files.

        Args:
            output_dir: Directory for the generated files.
            prefix: Prefix for the generated files (default: derived from the
                input filename).
            raw: When ``True`` this is the terminal step and no correction is
                applied. Correction does not yet exist, so behaviour is identical
                either way; no ``UB`` tag is emitted in either case.

        Returns:
            The reconciling :class:`UmiExtractionStats` for the run.
        """
        log.info(f"Extracting UMIs from {self.fastq.filename} (raw={raw})...")

        prefix = prefix or get_prefix(self.fastq.filename)
        output_path = Path(output_dir)
        umi_fastq_path = output_path / f"{prefix}.r1_umi.fastq.gz"
        umi_stats_path = output_path / f"{prefix}.umi_stats.txt"

        total = 0
        accepted = 0
        missing_left_anchor = 0
        no_polyg_anchor = 0
        length_counts: Counter[int] = Counter()

        with GzipFile(str(umi_fastq_path)).open_write_stream() as umi_stream:
            for name, seq, qual, *_ in self.fastq.open_read_iterator(as_string=True):
                total += 1
                ann = ReadAnnotation.parse(name)

                pos = ann.get(self.left_key)
                if pos is None:
                    missing_left_anchor += 1
                    continue

                umi_start = parse_span(pos)[1]
                polyg_start = self.find_polyg_start(seq, umi_start)
                if polyg_start is None:
                    no_polyg_anchor += 1
                    continue

                raw_umi = seq[umi_start:polyg_start]
                ann.set(self.umi_name, raw_umi)
                ann.set(position_key(self.umi_name), format_span(umi_start, polyg_start))
                FastqFile.write_read(umi_stream, ann.render(), seq, qual)

                accepted += 1
                length_counts[len(raw_umi)] += 1

        stats = UmiExtractionStats(
            total_reads=total,
            accepted=accepted,
            missing_left_anchor=missing_left_anchor,
            no_polyg_anchor=no_polyg_anchor,
            length_counts=dict(length_counts),
        )

        with umi_stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        log.info(f"Extracted UMIs for {accepted}/{total} reads")
        return stats
