"""Writes one TGIDX=NONE read into the scRNA arm's three STARsolo-shaped output files.

A ``TGIDX=NONE`` read carries no target index, so nothing anchors an insert past a
homopolymer run: the read is scRNA cDNA rather than TGIDX-tagged genomic material. This
arm trims R1 down to that cDNA insert, passes R2 through untouched, and synthesizes a
third barcodes FASTQ by slicing the original untrimmed R1 at each barcode and UMI
component's own recorded ``*_POS`` span - the shape STARsolo expects for its own
barcode/UMI-plus-cDNA read pair.
"""

from carmack.chemistry.annotation import parse_span, position_key
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_component import ReadComponentType
from carmack.io.fastq_file import FastqFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.prepare_reads.insert_locator import insert_start


class ScrnaWriter:
    """Writes one TGIDX=NONE read into the scRNA arm's three STARsolo-shaped files."""

    def __init__(self, chemistry: ChemistryBase) -> None:
        """Resolve and cache the chemistry layout this writer needs for every read.

        Everything resolved here is fixed for the lifetime of the writer and is never
        re-derived per read, so a mistake made here would silently and consistently
        corrupt every read the writer ever processes.

        Args:
            chemistry: The already-resolved chemistry describing the read layout.

        Raises:
            ValueError: If the chemistry declares no UMI component. Every read this
                writer is ever handed in production already passed through
                extract-umis, which transitively guarantees a UMI component exists,
                but construction must not trust that transitively.
        """
        umi = chemistry.umi_component()
        if umi is None:
            raise ValueError(
                f"chemistry '{chemistry.name}' declares no UMI component, so this writer "
                "cannot locate the UMI span it needs to build the barcodes FASTQ"
            )

        self.umi = umi
        self.umi_position_key = position_key(self.umi.name)
        self.umi_right_anchor = chemistry.umi_right_anchor()
        self.barcode_position_keys = tuple(
            position_key(comp.name)
            for comp in chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)
        )

    def read_span(self, ann: ReadAnnotation, key: str) -> tuple[int, int]:
        """Read and parse one component's span off an annotation, guarding a missing tag.

        ``write_read`` has no run-wide first-read validation to lean on the way
        ``TargetAssigner.validate_header`` and ``UmiExtractor.validate_header`` do: it is
        handed one read at a time, so this defensive check runs on every call instead.

        Args:
            ann: The parsed annotation header of the read.
            key: The ``*_POS`` tag key to read the span from.

        Returns:
            The ``(start, end)`` span parsed from the tag's value.

        Raises:
            ValueError: If the annotation carries no value under ``key``.
        """
        value = ann.get(key)
        if value is None:
            raise ValueError(f"Read '{ann.read_id}' is missing the expected '{key}' tag")
        return parse_span(value)

    def write_read(
        self,
        ann: ReadAnnotation,
        r1_seq: str,
        r1_qual: str,
        r2_name: str,
        r2_seq: str,
        r2_qual: str,
        r1_stream,
        r2_stream,
        barcodes_stream,
    ) -> None:
        """Write one read into the scRNA arm's R1, R2, and barcodes FASTQ streams.

        Every output header is reduced to its bare read id: R1 and the barcodes record
        both take ``ann.read_id``, while R2's header is parsed independently out of
        ``r2_name`` rather than reused from ``ann``, since validating that R1 and R2
        actually pair up is a later driver's job, not this method's.

        Args:
            ann: The parsed annotation header of the R1 read.
            r1_seq: The full, untrimmed R1 sequence.
            r1_qual: The full, untrimmed R1 quality string.
            r2_name: The FASTQ header line for R2.
            r2_seq: The R2 sequence.
            r2_qual: The R2 quality string.
            r1_stream: Output stream for the trimmed R1 insert FASTQ.
            r2_stream: Output stream for the passthrough R2 FASTQ.
            barcodes_stream: Output stream for the synthesized barcodes FASTQ.
        """
        umi_start, umi_end = self.read_span(ann, self.umi_position_key)
        cut = insert_start(reference=umi_end, anchor=self.umi_right_anchor, seq=r1_seq)
        FastqFile.write_read(r1_stream, ann.read_id, r1_seq[cut:], r1_qual[cut:])

        r2_read_id = ReadAnnotation.parse(r2_name).read_id
        FastqFile.write_read(r2_stream, r2_read_id, r2_seq, r2_qual)

        # The UMI's own span was already parsed above for the R1 cut point, so it is
        # reused here rather than re-derived from self.umi_position_key, keeping the
        # UMI slice appended last, after every barcode segment, in read-structure order.
        spans = [self.read_span(ann, key) for key in self.barcode_position_keys]
        spans.append((umi_start, umi_end))
        barcodes_seq = "".join(r1_seq[start:end] for start, end in spans)
        barcodes_qual = "".join(r1_qual[start:end] for start, end in spans)
        FastqFile.write_read(barcodes_stream, ann.read_id, barcodes_seq, barcodes_qual)
