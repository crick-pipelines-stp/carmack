"""Writes one TGIDX=NONE read into the scRNA arm's three STARsolo-shaped output files.

A ``TGIDX=NONE`` read carries no target index, so nothing anchors an insert past a
homopolymer run: the read is scRNA cDNA rather than TGIDX-tagged genomic material. This
arm trims R1 down to that cDNA insert, passes R2 through untouched, and synthesizes a
third barcodes FASTQ carrying that read's barcode and UMI - the shape STARsolo expects
for its own barcode/UMI-plus-cDNA read pair.

The two halves of that synthesized record come from two different sources. Its sequence
is read out of the header annotation's own value tags and is never re-sliced out of the
raw read: barcode extraction already wrote the corrected whitelist entry into the same
annotation the position came from, the scTIP arm already emits exactly those corrected
values in its own header, and because this is a multiome experiment the two arms are
joined downstream by barcode - so a raw slice here would present one physical cell under
two different barcodes and split its data in half whenever a sequencing error had been
corrected. Its quality has no such source, because a header carries no quality
information, so it is sliced out of the read's own quality string at each component's
recorded start, for that component's own declared length, never out to the matched
span's end. Slicing by the declared length is what keeps the record at one fixed total
width even when a corrected indel left a recorded span a base narrower or wider than the
component really is. The rare cost, accepted deliberately rather than chased for more
precision, is that one quality position inside that fixed window can really belong to
the neighbouring component - the same approximation CellRanger's CB/CY tag convention
already makes.
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
        corrupt every read the writer ever processes. The chemistry itself is kept
        alongside the components unpacked from it, because building the barcodes
        record's sequence goes through ``chemistry.construct_full_barcode`` - the same
        helper the scTIP arm's header rendering uses, which is what makes both arms of
        one multiome experiment spell a full barcode identically.

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

        self.chemistry = chemistry
        self.umi = umi
        self.umi_position_key = position_key(self.umi.name)
        self.umi_right_anchor = chemistry.umi_right_anchor()
        self.barcode_components = tuple(
            chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)
        )

    def read_value(self, ann: ReadAnnotation, key: str) -> str:
        """Read one component's recorded value off an annotation, guarding a missing tag.

        Args:
            ann: The parsed annotation header of the read.
            key: The value tag key to read, which is the component's own name.

        Returns:
            The value recorded under ``key``.

        Raises:
            ValueError: If the annotation carries no value under ``key``.
        """
        value = ann.get(key)
        if value is None:
            raise ValueError(f"Read '{ann.read_id}' is missing the expected '{key}' tag")
        return value

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
        return parse_span(self.read_value(ann, key))

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

        The barcodes record's sequence is taken from the annotation's value tags alone -
        the corrected whitelist barcodes barcode extraction already wrote there, joined
        in read-structure order by ``construct_full_barcode``, then the UMI's own tag
        value - and never re-sliced out of ``r1_seq``. The scTIP arm emits exactly those
        corrected values, and the two arms of one multiome experiment are joined
        downstream by barcode, so re-deriving raw bases here would fragment one physical
        cell into two apparent cells whenever a sequencing error had been corrected. Its
        quality has to come from the read, because a header carries no quality
        information, and is taken from each component's recorded start for that
        component's own declared length rather than out to the matched span's end, which
        is what holds the record at one fixed total width across every read. The
        accepted limitation is that a corrected indel can leave one quality position in
        that fixed window really belonging to the neighbouring component; this is the
        same approximation CellRanger's CB/CY tag convention makes, and is not worth
        more precision.

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

        Raises:
            ValueError: If the annotation is missing any barcode or UMI component's
                value tag or ``*_POS`` tag.
        """
        umi_start, umi_end = self.read_span(ann, self.umi_position_key)
        cut = insert_start(reference=umi_end, anchor=self.umi_right_anchor, seq=r1_seq)
        FastqFile.write_read(r1_stream, ann.read_id, r1_seq[cut:], r1_qual[cut:])

        r2_read_id = ReadAnnotation.parse(r2_name).read_id
        FastqFile.write_read(r2_stream, r2_read_id, r2_seq, r2_qual)

        barcode_values: dict[str, str] = {}
        quality_parts: list[str] = []
        for comp in self.barcode_components:
            barcode_values[comp.name] = self.read_value(ann, comp.name)
            start = self.read_span(ann, comp.position_key)[0]
            quality_parts.append(r1_qual[start : start + comp.length])

        # The UMI's own span was already parsed above for the R1 cut point, so it is
        # reused here rather than re-derived from self.umi_position_key, keeping the
        # UMI appended last, after every barcode segment, in read-structure order.
        umi_value = self.read_value(ann, self.umi.name)
        quality_parts.append(r1_qual[umi_start : umi_start + self.umi.length])

        barcodes_seq = self.chemistry.construct_full_barcode(barcode_values) + umi_value
        FastqFile.write_read(barcodes_stream, ann.read_id, barcodes_seq, "".join(quality_parts))
