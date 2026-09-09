"""UMI extraction from annotated R1 FASTQ reads.

The UMI is defined positionally: the fixed number of bases the chemistry
declares, taken immediately after the component the UMI hangs off. That
component's span is read from the annotated header written by barcode
extraction, so the slice absorbs any upstream indel that moved it. Each read is
annotated with a ``UMI`` tag and a ``UMI_POS`` span, then re-emitted.

The span is written because a tag naming a sequence says nothing about where in
the read that sequence came from, and every other extracted component records
its own coordinates. It is not read back by any later stage: target assignment
derives the same coordinate from the chemistry, so the two stay independent and
this span is a record of what was cut, for a consumer that wants to trim or
re-inspect the read, not a channel between stages.

There is nothing to search for and nothing to correct. The boundary is not
inferred from the read, so no read is rejected for failing to present one, and
the extracted UMI is the sequence that occupies those positions whatever it
happens to be. A read is skipped only when its anchor was never recorded, or
when it ends before the UMI does.

Because the boundary is asserted rather than found, the report carries the
anchor homopolymer run length observed at the first base after the UMI. Nothing
gates on it, but it is the one place a wrong offset -- or a library that is not
this chemistry -- shows up, as mass in the zero bin.
"""

import logging
from collections import Counter
from itertools import chain
from pathlib import Path

from carmack.chemistry.annotation import format_span, parse_span
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponentType
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_reporting import UmiExtractionStats
from carmack.utils import get_prefix, homopolymer_run_length

log = logging.getLogger(__name__)


class UmiExtractor:
    """Extracts fixed-length UMIs from an annotated R1 FASTQ using its chemistry.

    The extractor is a thin wrapper mirroring the barcode extractor pattern: it
    resolves the chemistry, works out which recorded anchor the UMI is measured
    from, and streams the reads once to write the annotated UMI FASTQ and the
    stats report.
    """

    def __init__(self, fastq_file: str, chemistry_name: str) -> None:
        """Resolve the chemistry and derive the UMI extraction parameters.

        Args:
            fastq_file: Path to the annotated R1 FASTQ (produced by barcode
                extraction, carrying the anchor's position tag).
            chemistry_name: Name of the chemistry describing the read layout.

        Raises:
            ValueError: If the chemistry is unknown, declares no UMI component,
                or places no recorded anchor 5' of that UMI at a fixed distance.
        """
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_umi_extraction():
            raise ValueError(f"chemistry '{chemistry_name}' declares no UMI component")

        umi = self.chemistry.umi_component()
        located = self.chemistry.resolve_anchor_offset(umi)

        self.umi_name = umi.name
        self.umi_pos_key = umi.position_key
        self.umi_length = umi.length
        self.anchor_pos_key = located.position_key
        self.umi_offset = located.offset

        # Diagnostic only, so a chemistry whose UMI has no homopolymer 3' of it
        # extracts exactly as well and simply reports no anchor-run section. A
        # non-homopolymer neighbour carries no homopolymer_base, which is why the
        # base is read off the component rather than assumed present.
        right = self.chemistry.umi_right_anchor()
        self.anchor_base = right.homopolymer_base if right is not None else None

        self.barcode_names = [
            component.name
            for component in self.chemistry.read_structure.get_components_by_type(
                ReadComponentType.BARCODE
            )
        ]

    def validate_header(self, ann: ReadAnnotation) -> None:
        """Validate that an annotated read carries the tags this chemistry needs.

        Checks the first annotated read against the supplied chemistry: it must
        carry the anchor position tag the UMI is measured from, and a value tag
        for every barcode component. This is the stage's only agreement check
        between the chemistry it was given and the FASTQ it was pointed at.
        Everything after it succeeds unconditionally, so without this a
        three-barcode chemistry run against reads annotated with two would
        complete normally and emit UMIs cut from the wrong coordinate.

        Args:
            ann: The parsed header of the first annotated read.

        Raises:
            ValueError: When an expected tag is absent from the header.
        """
        required = [self.anchor_pos_key, *self.barcode_names]
        missing = [key for key in required if ann.get(key) is None]
        if missing:
            raise ValueError(
                f"Annotated read '{ann.read_id}' is missing expected tag(s) {missing} "
                f"for chemistry '{self.chemistry_name}'. The annotated FASTQ may have "
                "been produced with a different chemistry."
            )

    def extract_umis(self, output_dir: str = ".", prefix: str | None = None) -> UmiExtractionStats:
        """Stream the reads, annotate the extracted UMI and write the output files.

        Args:
            output_dir: Directory for the generated files.
            prefix: Prefix for the generated files (default: derived from the
                input filename).

        Returns:
            The reconciling :class:`UmiExtractionStats` for the run.

        Raises:
            ValueError: If the first read's header does not match the supplied
                chemistry.
        """
        log.info(f"Extracting UMIs from {self.fastq.filename}...")

        prefix = prefix or get_prefix(self.fastq.filename)
        output_path = Path(output_dir)
        umi_fastq_path = output_path / f"{prefix}.r1_umi.fastq.gz"
        umi_stats_path = output_path / f"{prefix}.umi_stats.txt"

        total = 0
        accepted = 0
        missing_left_anchor = 0
        truncated = 0
        run_counts: Counter[int] = Counter()

        # Validate the first read's header against the chemistry before writing
        # anything, so a chemistry / FASTQ mismatch fails fast with a clear error
        # rather than producing a full output file cut from a wrong coordinate.
        reads = self.fastq.open_read_iterator(as_string=True)
        first_read = next(reads, None)
        if first_read is not None:
            self.validate_header(ReadAnnotation.parse(first_read[0]))
            reads = chain([first_read], reads)

        with GzipFile(str(umi_fastq_path)).open_write_stream() as umi_stream:
            for name, seq, qual, *_ in reads:
                total += 1
                ann = ReadAnnotation.parse(name)

                pos = ann.get(self.anchor_pos_key)
                if pos is None:
                    missing_left_anchor += 1
                    continue

                umi_start = parse_span(pos)[1] + self.umi_offset
                umi_end = umi_start + self.umi_length
                if umi_end > len(seq):
                    # The read ends inside the UMI, so the fixed-length slice
                    # does not exist. Taking a short one would emit a UMI whose
                    # length silently disagrees with every other read's.
                    truncated += 1
                    continue

                ann.set(self.umi_name, seq[umi_start:umi_end])
                # Always the full fixed width: a read that could only yield a
                # short span was counted truncated above and never reaches here.
                ann.set(self.umi_pos_key, format_span(umi_start, umi_end))
                FastqFile.write_read(umi_stream, ann.render(), seq, qual)

                accepted += 1
                if self.anchor_base is not None:
                    # umi_end is also where the chemistry says the anchor run
                    # begins, and it is the same coordinate target assignment
                    # recomputes from the chemistry rather than reading back.
                    run_counts[homopolymer_run_length(seq, umi_end, self.anchor_base)] += 1

        stats = UmiExtractionStats(
            total_reads=total,
            accepted=accepted,
            missing_left_anchor=missing_left_anchor,
            truncated=truncated,
            umi_length=self.umi_length,
            homopolymer_base=self.anchor_base,
            homopolymer_run_counts=dict(run_counts),
        )

        with umi_stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        log.info(f"Extracted UMIs for {accepted}/{total} reads")
        return stats
