"""Tests for ScrnaWriter, which writes one TGIDX=NONE read into the scRNA arm's three
STARsolo-shaped output files: a trimmed R1 insert, a passthrough R2, and a synthesized
barcodes FASTQ carrying that read's barcode and UMI.

The barcodes record's two halves come from two different sources, and most of this file
exists to hold that split in place. Its sequence is read out of the header annotation's
own value tags - the corrected whitelist barcodes barcode extraction already wrote
there, concatenated in read-structure order by ``chemistry.construct_full_barcode``,
followed by the UMI's own tag value - and is never re-sliced out of the raw read. The
scTIP arm already emits exactly those corrected values in its ``CB=``/``UR=`` header,
and the two arms of one multiome experiment are joined downstream by barcode, so an arm
that re-derived raw, error-containing bases here would split one physical cell into two
apparent cells. The record's quality has no such source - a header carries no quality -
so it is sliced out of the read's own quality string at each component's recorded
start, for that component's own declared length, never out to the matched span's end.
Slicing by declared length is what keeps the record at one fixed total width (38bp for
``carmack_custom_seq_1_0``) even when a corrected indel left a recorded span a base
narrower or wider than the component really is. The rare consequence, accepted
deliberately rather than chased for more precision, is that one quality position inside
that fixed window can really belong to the neighbouring component.

``TestScrnaWriterInit`` covers the attributes ``__init__`` resolves and caches once at
construction time rather than re-deriving on every read: the chemistry itself, which
``write_read`` needs a handle on to call ``construct_full_barcode``; the UMI component;
the annotation key its position is recorded under; its right anchor (the anchor
``insert_start`` walks past to find where the trimmed R1 insert begins); and the
ordered tuple of barcode components, which fixes both the order barcode segments are
concatenated in and the declared length each one's quality is sliced by. Getting any of
these wrong at construction time would silently corrupt every read the writer ever
processes, since none of them are re-checked per read.

The UMI-less-chemistry test exists because ``ScrnaWriter`` is documented to fail
clearly, rather than derive nonsense from a ``None`` component, when constructed
directly against a chemistry that cannot supply a UMI. Every read this class is ever
handed in production already passed through extract-umis, which transitively
guarantees a UMI component exists, but the class must not trust that transitively.

The rest of the file covers ``write_read``. ``TestScrnaWriterWriteRead`` takes a read
whose every component matched cleanly and covers the R1 trim, the R2 passthrough, the
bare-read-id output headers, and barcode+UMI concatenation order.
``TestScrnaWriterEmitsCorrectedFixedWidthRecords`` drives the two cases a clean read
cannot reach: a recorded span whose raw bases disagree with the corrected value written
beside them, and a recorded span whose width disagrees with the component's declared
length. ``TestScrnaWriterBarcodeOrderIsStructural`` guards concatenation order against
ever hardcoding BC3/BC2/BC1, using a fabricated chemistry double.
``TestScrnaWriterMissingPositionTag`` and ``TestScrnaWriterMissingValueTag`` cover the
two per-read guards, one over the position tags quality is sliced by and one over the
value tags sequence is built from. ``TestScrnaWriterGoldenGzipRoundTrip`` is the only
test here that writes through a real gzip subprocess rather than an ``io.BytesIO()``.

The last two classes cover the split between deciding where R1 is cut and performing
the cut, which fall in two different processes. ``TestScrnaWriterInsertCut`` covers
``insert_cut``, the arithmetic side: the UMI's recorded span end handed to
``insert_start`` along with the chemistry's UMI right anchor, returning exactly what
that call returns for the caller to decide on. ``TestScrnaWriterCutIsGivenNotRecomputed``
covers the writing side: ``write_read`` slices R1 at the cut it is handed rather than at
one it derives for itself, which is pinned by handing it a cut that deliberately
disagrees with the one it would have derived - an assertion no implementation that
recomputes internally can pass. Those same tests pin that the barcodes record does not
move with the cut at all, because its quality windows are sliced in the original,
untrimmed read's coordinates rather than the trimmed insert's.
"""

import io
from functools import cached_property
from inspect import signature
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import ChemistryCarmackCustomSeq10
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.prepare_reads.insert_locator import insert_start
from carmack.prepare_reads.scrna_writer import ScrnaWriter

CHEMISTRY = "carmack_custom_seq_1_0"

# Both chemistries whose reads ever reach this writer. They declare the same UMI right
# anchor but seat the UMI at different read offsets, so a cut derived from the span the
# read itself records answers both, while one derived from a chemistry-fixed offset
# could only ever answer one. hydrop is absent deliberately: it declares no UMI at all,
# and ScrnaWriter refuses to be constructed against it.
SHIPPED_CHEMISTRIES = (CHEMISTRY, "carmack_custom_seq_1_0_primd")

# Barcode component names in the order the shipped chemistry's read structure
# declares them - the order barcode_components must reproduce, and the order the
# emitted barcodes record concatenates them in.
BARCODE_NAMES_IN_STRUCTURE_ORDER = ("BC3", "BC2", "BC1")

NO_UMI_CHEMISTRY = "custom_seq_without_umi_for_scrna_writer"

# The real carmack_custom_seq_1_0 chemistry's own declared component widths - its
# BC_CHUNK_LEN and UMI_LENGTH - restated here so the synthetic reads built below look like
# the real thing. These are not decorative: write_read slices quality by each component's
# declared length rather than out to its recorded span's end, so a synthetic read whose
# component widths disagreed with the chemistry's would be exercising a layout the writer is
# not contracted to handle. Tests that deliberately disagree with a declared width do so on
# one named component at a time, and say so.
BARCODE_LENGTH = 10
UMI_LENGTH = 8

# The one fixed total width every synthesized barcodes record has for this chemistry: BC3,
# BC2 and BC1 at BARCODE_LENGTH each, then the UMI at UMI_LENGTH. Written as the sum rather
# than as the literal 38 so a chemistry change moves it honestly; the tests that assert on it
# also check it against the chemistry's own declared lengths.
FIXED_BARCODES_RECORD_LENGTH = 3 * BARCODE_LENGTH + UMI_LENGTH


class ChemistryWithoutUmi(ChemistryCarmackCustomSeq10):
    """Shipped chemistry with its UMI component removed entirely.

    Keeps every other component (barcodes, primers, the poly-G anchor and the
    target index) exactly as ``carmack_custom_seq_1_0`` declares them, so
    construction still passes every other validation
    ``ChemistryBase.__post_init__`` runs; only the UMI is missing, which is
    the one thing ``ScrnaWriter.__init__`` must refuse to build against.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return NO_UMI_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return the shipped read structure with its UMI component dropped."""
        return ReadStructure(
            [comp for comp in super().read_structure if comp.type is not ReadComponentType.UMI]
        )


class TestScrnaWriterInit:
    """Tests for the attributes ScrnaWriter.__init__ resolves and caches once."""

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry ScrnaWriter is built against in production.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    def test_retains_the_chemistry_itself(self, chemistry: ChemistryBase) -> None:
        """Test that __init__ keeps the chemistry it was handed, not only facts derived from it.

        ``write_read`` builds the barcodes record's sequence by calling
        ``chemistry.construct_full_barcode`` - the same helper the scTIP arm's header
        rendering already uses - so the writer needs the chemistry object itself for the
        life of the run, not just the component metadata it unpacked at construction
        time. Reusing that one helper, rather than reimplementing concatenation here, is
        what guarantees both arms of a multiome experiment agree on how a full barcode
        is spelled.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.chemistry).is_same_as(chemistry)

    def test_caches_the_chemistrys_umi_component(self, chemistry: ChemistryBase) -> None:
        """Test that __init__ resolves chemistry.umi_component() once, onto self.umi.

        A later write_read call has no per-read way to re-derive this, so if
        this were ever wrong, cut placement for every read the writer
        processes would be wrong, and consistently wrong. The component carries
        the UMI's declared length as well as its name, and write_read needs all
        three facts - the name to read the corrected value tag under, the
        position key to read the recorded start from, and the declared length to
        slice that many quality characters - so caching the component itself is
        what makes a separate cached UMI length unnecessary.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.umi).is_equal_to(chemistry.umi_component())
        assert_that(writer.umi.name).is_equal_to("UMI")
        assert_that(writer.umi.type).is_equal_to(ReadComponentType.UMI)
        assert_that(writer.umi.length).is_equal_to(UMI_LENGTH)

    def test_caches_the_umi_position_key(self, chemistry: ChemistryBase) -> None:
        """Test that umi_position_key is the position_key of the resolved UMI's own name.

        write_read reads the UMI's span back out of the read's annotation
        header using this exact key, so it must be derived from the UMI
        component ScrnaWriter actually resolved, not a hardcoded "UMI_POS"
        literal that could silently drift if the chemistry ever renamed the
        component.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.umi_position_key).is_equal_to(position_key(writer.umi.name))
        assert_that(writer.umi_position_key).is_equal_to("UMI_POS")

    def test_caches_the_umi_right_anchor(self, chemistry: ChemistryBase) -> None:
        """Test that umi_right_anchor is resolved via chemistry.umi_right_anchor().

        For ``carmack_custom_seq_1_0`` this is the POLYG homopolymer
        immediately 3' of the UMI: the anchor insert_start walks past to find
        where the trimmed R1 insert begins.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.umi_right_anchor).is_equal_to(chemistry.umi_right_anchor())
        assert_that(writer.umi_right_anchor.type).is_equal_to(ReadComponentType.HOMOPOLYMER)
        assert_that(writer.umi_right_anchor.name).is_equal_to("POLYG")

    def test_caches_barcode_components_in_read_structure_order(
        self, chemistry: ChemistryBase
    ) -> None:
        """Test that barcode_components follows read_structure order, not a hardcoded list.

        Every per-read fact write_read needs about a barcode hangs off this one
        tuple, and each of the three comes from a different half of the record:
        the component's name is the tag the corrected sequence is read from, its
        position key is the tag the recorded start is read from, and its
        declared length is how many quality characters are taken from that
        start. Caching the components themselves rather than three parallel
        tuples keeps those three facts from ever drifting out of step with each
        other, and walking
        ``get_components_by_type(ReadComponentType.BARCODE)`` rather than a name
        list baked into the writer is what makes concatenation order follow
        whatever the chemistry declares.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.barcode_components).is_equal_to(
            tuple(chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE))
        )
        assert_that(tuple(comp.name for comp in writer.barcode_components)).is_equal_to(
            BARCODE_NAMES_IN_STRUCTURE_ORDER
        )
        assert_that(tuple(comp.position_key for comp in writer.barcode_components)).is_equal_to(
            tuple(position_key(name) for name in BARCODE_NAMES_IN_STRUCTURE_ORDER)
        )
        assert_that(tuple(comp.length for comp in writer.barcode_components)).is_equal_to(
            (BARCODE_LENGTH,) * len(BARCODE_NAMES_IN_STRUCTURE_ORDER)
        )

    def test_umiless_chemistry_raises_value_error(self) -> None:
        """Test that constructing against a chemistry with no UMI fails clearly.

        Every read ScrnaWriter is ever handed in production already passed
        through extract-umis, which transitively guarantees a UMI component
        exists, but the class must not trust that transitively: constructing
        it directly against an incompatible chemistry must fail here, at
        construction time, rather than derive nonsense from a None umi later.
        """
        chemistry = ChemistryWithoutUmi()
        assert_that(chemistry.umi_component()).is_none()

        with pytest.raises(ValueError):
            ScrnaWriter(chemistry)


def parse_fastq_record(stream: io.BytesIO) -> tuple[str, str, str]:
    """Parse the single four-line FASTQ record write_read wrote into a stream.

    write_read is contracted to write through FastqFile.write_read, which emits
    plain ``@name``/``seq``/``+``/``qual`` lines encoded as UTF-8 bytes, so a bare
    io.BytesIO() is a faithful, subprocess-free stream double for these tests.

    Args:
        stream: The stream a single write_read call wrote one record into.

    Returns:
        The ``(header, seq, qual)`` lines, with the header's leading ``@`` stripped.
    """
    header_line, seq_line, plus_line, qual_line = stream.getvalue().decode("UTF-8").splitlines()
    assert_that(plus_line).is_equal_to("+")
    return header_line[1:], seq_line, qual_line


def build_scrna_read(
    read_id: str,
    bc3_seq: str,
    bc2_seq: str,
    bc1_seq: str,
    umi_seq: str,
    polyg_run_length: int,
    insert_seq: str,
    extra_tag: tuple[str, str] | None = None,
) -> tuple[ReadAnnotation, str, str]:
    """Build a synthetic, fully-annotated R1 read for write_read tests.

    Lays out BC3, BC2, BC1, UMI, a poly-G run of the given length, and an insert, in
    that read-structure order, then annotates each barcode/UMI component exactly the way
    the pipeline's own extraction stages do: a ``<NAME>`` tag holding the component's
    sequence and a ``<NAME>_POS`` tag holding its span, written through position_key and
    format_span. Both halves matter, because write_read reads each half for a different
    purpose - the value tag is where the emitted sequence comes from, the position tag is
    where the emitted quality's start comes from - and a read annotated with only one of
    them is not a read the writer is ever handed in production.

    Every component here is laid out at exactly its declared width, and its value tag
    holds exactly the bases the read carries at its own span, so a read built by this
    helper is the clean case: nothing was corrected, and raw slice and corrected value
    would agree. Tests that need those two to disagree, in bases or in width, build
    their read through build_indel_corrected_scrna_read instead.

    Args:
        read_id: The read id to put on the annotation.
        bc3_seq: The BC3 component's sequence.
        bc2_seq: The BC2 component's sequence.
        bc1_seq: The BC1 component's sequence.
        umi_seq: The UMI component's sequence.
        polyg_run_length: How many ``G`` bases immediately follow the UMI, standing in
            for the real poly-G anchor's observed run.
        insert_seq: Sequence placed immediately after the poly-G run, standing in for
            the cDNA insert that follows it in a real read.
        extra_tag: An optional ``(key, value)`` tag to add to the annotation on top of
            the barcode/UMI position tags, so a test can confirm the annotation carries
            more than just the read id before checking that write_read's output headers
            still reduce to the bare read id.

    Returns:
        The annotation, the assembled R1 sequence, and a same-length quality string
        whose character at each position is derived from that position's own index, so
        a wrong slice or a wrong concatenation order shows up as a wrong quality string
        too, not only a wrong sequence.
    """
    bc3_start = 0
    bc3_end = bc3_start + len(bc3_seq)
    bc2_start = bc3_end
    bc2_end = bc2_start + len(bc2_seq)
    bc1_start = bc2_end
    bc1_end = bc1_start + len(bc1_seq)
    umi_start = bc1_end
    umi_end = umi_start + len(umi_seq)

    r1_seq = bc3_seq + bc2_seq + bc1_seq + umi_seq + ("G" * polyg_run_length) + insert_seq
    r1_qual = "".join(chr(33 + (position % 50)) for position in range(len(r1_seq)))

    annotation = ReadAnnotation(read_id=read_id)
    annotation.set("BC3", bc3_seq)
    annotation.set(position_key("BC3"), format_span(bc3_start, bc3_end))
    annotation.set("BC2", bc2_seq)
    annotation.set(position_key("BC2"), format_span(bc2_start, bc2_end))
    annotation.set("BC1", bc1_seq)
    annotation.set(position_key("BC1"), format_span(bc1_start, bc1_end))
    annotation.set("UMI", umi_seq)
    annotation.set(position_key("UMI"), format_span(umi_start, umi_end))
    if extra_tag is not None:
        annotation.set(*extra_tag)

    return annotation, r1_seq, r1_qual


def polyg_cut(annotation: ReadAnnotation, polyg_run_length: int) -> int:
    """Return the R1 cut point for a read ``build_scrna_read`` laid out.

    ``write_read`` is handed its cut rather than deriving one, so every call below has to
    supply it. For a read this file built, the cut is arithmetic the test already knows -
    the UMI's own recorded span end, plus however many poly-G bases the read was built to
    carry after it - so it is spelled out here rather than read back out of the writer,
    keeping each test's expected trim point independent of the code that produces the
    real one.

    Args:
        annotation: The annotation ``build_scrna_read`` returned for the read.
        polyg_run_length: The poly-G run length that read was built with.

    Returns:
        The 0-based coordinate that read's insert begins at.
    """
    return parse_span(annotation.get(position_key("UMI")))[1] + polyg_run_length


class TestScrnaWriterWriteRead:
    """Tests for write_read's R1 trim, R2 passthrough, and barcode+UMI concatenation.

    Every test here writes into three independent io.BytesIO() streams rather than
    real files: FastqFile.write_read only ever calls stream.write() with already-encoded
    bytes, so a BytesIO is a faithful, fast, subprocess-free double for the three
    already-open streams write_read is handed in production.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry write_read is exercised against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    @pytest.fixture
    def writer(self, chemistry: ChemistryBase) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(chemistry)

    def test_write_read_trims_r1_to_the_observed_polyg_run(
        self, writer: ScrnaWriter, chemistry: ChemistryBase
    ) -> None:
        """Test that the R1 output is cut at the UMI's own homopolymer-anchored insert start.

        The trim point arrives as an argument now, so what this test supplies is the cut
        a real dispatcher would have supplied for this read: the UMI span's end plus the
        real poly-G run length observed in this specific read, not any nominal length. It
        is spelled out here and checked against insert_start so the expected value never
        comes out of the code under test, and what write_read is then held to is slicing
        both r1_seq and r1_qual identically from that one point. Slicing seq and qual by
        different amounts would leave every downstream STARsolo alignment reading a
        corrupted or misaligned insert.
        """
        run_length = 5
        insert_seq = "TATAGCCT" + "CTCTTATACACATCTCCTC"
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read1",
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=run_length,
            insert_seq=insert_seq,
        )
        umi_end = parse_span(annotation.get(position_key("UMI")))[1]
        cut = polyg_cut(annotation, run_length)
        assert_that(cut).is_equal_to(umi_end + run_length)
        assert_that(cut).is_equal_to(
            insert_start(reference=umi_end, anchor=chemistry.umi_right_anchor(), seq=r1_seq)
        )

        r2_name = "read1"
        r2_seq = "ACGT" * 10
        r2_qual = "I" * len(r2_seq)
        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        header, seq, qual = parse_fastq_record(r1_stream)
        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_equal_to(r1_seq[cut:])
        assert_that(qual).is_equal_to(r1_qual[cut:])

    def test_write_read_passes_r2_through_unchanged(self, writer: ScrnaWriter) -> None:
        """Test that write_read never touches R2's sequence or quality, only its header.

        R1/R2 pairing validation is explicitly out of scope for write_read (a later
        driver owns it), so R2's header is parsed independently of ann; but its
        sequence and quality must reach the output byte-for-byte, since nothing about
        the scRNA arm's cDNA read needs, or is allowed, to trim R2.
        """
        polyg_run_length = 4
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read7",
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        cut = polyg_cut(annotation, polyg_run_length)
        r2_name = "read7"
        r2_seq = "GATTACAGATTACAGATTACA"
        r2_qual = "".join(chr(33 + (position % 40)) for position in range(len(r2_seq)))

        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        header, seq, qual = parse_fastq_record(r2_stream)
        assert_that(seq).is_equal_to(r2_seq)
        assert_that(qual).is_equal_to(r2_qual)
        assert_that(header).is_equal_to(ReadAnnotation.parse(r2_name).read_id)

    def test_write_read_concatenates_barcodes_and_umi_in_structure_order(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that the barcodes FASTQ concatenates BC3, BC2, BC1, then UMI, in that order.

        Each component here uses a distinct repeated base, so any wrong concatenation
        order - alphabetical, chemistry-declaration order reversed, or two barcodes
        swapped with each other - would show up as an unmistakably wrong sequence
        rather than a subtle one. This is the one behaviour that makes barcode order in
        the synthesized barcodes FASTQ follow the real chemistry's actual BC3, BC2, BC1
        read-structure order rather than some other ordering.
        """
        bc3_seq = "G" * BARCODE_LENGTH
        bc2_seq = "C" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH
        polyg_run_length = 3
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read9",
            bc3_seq=bc3_seq,
            bc2_seq=bc2_seq,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        cut = polyg_cut(annotation, polyg_run_length)

        bc3_end = len(bc3_seq)
        bc2_end = bc3_end + len(bc2_seq)
        bc1_end = bc2_end + len(bc1_seq)
        umi_end = bc1_end + len(umi_seq)
        expected_seq = bc3_seq + bc2_seq + bc1_seq + umi_seq
        expected_qual = (
            r1_qual[:bc3_end]
            + r1_qual[bc3_end:bc2_end]
            + r1_qual[bc2_end:bc1_end]
            + r1_qual[bc1_end:umi_end]
        )

        r2_name = "read9"
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)
        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        header, seq, qual = parse_fastq_record(barcodes_stream)
        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_equal_to(expected_seq)
        assert_that(qual).is_equal_to(expected_qual)

    def test_write_read_headers_carry_only_the_bare_read_id(self, writer: ScrnaWriter) -> None:
        """Test that every output header is reduced to the bare read id, tags stripped.

        A TGIDX=NONE read's own header carries the very tags (BC3_POS, UMI_POS,
        TGIDX=NONE, and so on) write_read reads its spans from, so if it wrote
        ann.render() or r2_name verbatim instead of the bare read id, STARsolo's own
        tools would see those tags as part of the read name. R1's and the barcodes
        record's bare id come from ann.read_id; R2's comes from parsing r2_name
        independently, since R1/R2 pairing validation is out of scope here. The
        annotation and r2_name are both built to carry tags beyond the read id, so this
        assertion is meaningful rather than trivially true.
        """
        read_id = "read5"
        polyg_run_length = 4
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id=read_id,
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
            extra_tag=("TGIDX", "NONE"),
        )
        cut = polyg_cut(annotation, polyg_run_length)
        assert_that(annotation.render()).is_not_equal_to(annotation.read_id)

        r2_name = f"{read_id} TGIDX=NONE"
        assert_that(ReadAnnotation.parse(r2_name).render()).is_not_equal_to(read_id)
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)

        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        r1_header = parse_fastq_record(r1_stream)[0]
        r2_header = parse_fastq_record(r2_stream)[0]
        barcodes_header = parse_fastq_record(barcodes_stream)[0]

        for header in (r1_header, r2_header, barcodes_header):
            assert_that(header).is_equal_to(read_id)
            assert_that(header).does_not_contain("=")


# A BC2 whitelist entry and one single-substitution misread of it. Barcode extraction
# writes the first into the read's BC2 tag while the read itself still carries the second,
# which is the whole point of correction: the pair is shaped after the real chemistry's own
# declared BC2 blind spot, where a single A->G - the dominant substitution direction on
# 2-colour Illumina chemistry - turns one valid cell barcode into another. Tests below
# assert these differ at exactly one base rather than trusting this comment.
CORRECTED_BC2 = "AGCTTGAGAG"
RAW_BC2_VARIANT = "AGCTTGACAG"


def build_indel_corrected_scrna_read(
    read_id: str,
    bc3_seq: str,
    bc2_observed_seq: str,
    bc2_corrected_seq: str,
    bc1_seq: str,
    umi_seq: str,
    polyg_run_length: int,
    insert_seq: str,
) -> tuple[ReadAnnotation, str, str]:
    """Build a synthetic R1 read whose BC2 tag disagrees with the bases BC2 was read from.

    ``build_scrna_read`` can only produce the clean case, where every component's value
    tag repeats the bases sitting at its own span. This helper produces the case
    correction actually creates: BC2's recorded span covers ``bc2_observed_seq``, the
    bases the sequencer produced, while BC2's value tag holds ``bc2_corrected_seq``, the
    whitelist entry the matcher resolved them to. The two may differ in bases, in width,
    or in both - ``KmerMatcher.extend_and_verify`` tolerates an indel by searching window
    lengths either side of the declared length and keeping whichever scores best, so a
    corrected span really can be a base narrower or wider than the barcode is. Every
    other component is left clean, so exactly one component is under test at a time.

    Args:
        read_id: The read id to put on the annotation.
        bc3_seq: The BC3 component's sequence, recorded and observed alike.
        bc2_observed_seq: The bases the read carries where BC2 was matched, and whose
            length therefore sets BC2's recorded span width.
        bc2_corrected_seq: The corrected whitelist value recorded in BC2's own tag.
        bc1_seq: The BC1 component's sequence, recorded and observed alike.
        umi_seq: The UMI component's sequence, recorded and observed alike.
        polyg_run_length: How many ``G`` bases immediately follow the UMI.
        insert_seq: Sequence placed immediately after the poly-G run.

    Returns:
        The annotation, the assembled R1 sequence, and a same-length quality string.
    """
    bc3_start = 0
    bc3_end = bc3_start + len(bc3_seq)
    bc2_start = bc3_end
    bc2_end = bc2_start + len(bc2_observed_seq)
    bc1_start = bc2_end
    bc1_end = bc1_start + len(bc1_seq)
    umi_start = bc1_end
    umi_end = umi_start + len(umi_seq)

    r1_seq = bc3_seq + bc2_observed_seq + bc1_seq + umi_seq + ("G" * polyg_run_length) + insert_seq
    r1_qual = "".join(chr(33 + (position % 50)) for position in range(len(r1_seq)))
    # Every quality character in this read has to be distinct, so that a test asserting one
    # specific read position's quality never reaches the emitted record is really asserting
    # that, rather than being satisfied by the same character arriving from somewhere else.
    # The generator above only starts repeating after 50 positions, so this holds while
    # callers keep their reads short; checking it here turns a silently weakened assertion
    # into a visible failure.
    assert_that(set(r1_qual)).is_length(len(r1_qual))

    annotation = ReadAnnotation(read_id=read_id)
    annotation.set("BC3", bc3_seq)
    annotation.set(position_key("BC3"), format_span(bc3_start, bc3_end))
    annotation.set("BC2", bc2_corrected_seq)
    annotation.set(position_key("BC2"), format_span(bc2_start, bc2_end))
    annotation.set("BC1", bc1_seq)
    annotation.set(position_key("BC1"), format_span(bc1_start, bc1_end))
    annotation.set("UMI", umi_seq)
    annotation.set(position_key("UMI"), format_span(umi_start, umi_end))

    return annotation, r1_seq, r1_qual


def write_barcodes_record(
    writer: ScrnaWriter, annotation: ReadAnnotation, r1_seq: str, r1_qual: str
) -> tuple[str, str, str]:
    """Drive one write_read call and hand back only the barcodes record it emitted.

    write_read writes all three output files in one call and cannot be asked for one of
    them in isolation, so all three streams are supplied here and the R1/R2 pair is
    discarded. R2 is a fixed stand-in rather than a per-test value because nothing in the
    barcodes record derives from it; the tests that do care about R1 and R2 live in
    ``TestScrnaWriterWriteRead`` and spell their own calls out. The R1 cut is a fixed
    stand-in for the same reason: the barcodes record's quality windows are sliced in the
    original read's coordinates and so do not move with the cut at all, which
    ``TestScrnaWriterCutIsGivenNotRecomputed`` pins directly rather than leaving to this
    helper's choice of value.

    Args:
        writer: The writer under test.
        annotation: The parsed annotation header of the R1 read.
        r1_seq: The full, untrimmed R1 sequence.
        r1_qual: The full, untrimmed R1 quality string.

    Returns:
        The barcodes record's ``(header, seq, qual)``.
    """
    r2_seq = "ACGT" * 5
    barcodes_stream = io.BytesIO()
    writer.write_read(
        annotation,
        r1_seq,
        r1_qual,
        0,
        annotation.read_id,
        r2_seq,
        "I" * len(r2_seq),
        io.BytesIO(),
        io.BytesIO(),
        barcodes_stream,
    )
    return parse_fastq_record(barcodes_stream)


class TestScrnaWriterEmitsCorrectedFixedWidthRecords:
    """Covers the two ways a real read's recorded span disagrees with what must be emitted.

    Every read in ``TestScrnaWriterWriteRead`` matched cleanly, so for those reads the
    raw bases under a span, the corrected value recorded beside it, and the component's
    declared width all coincide - which means those tests cannot tell the contract apart
    from re-slicing the read. Correction breaks that coincidence in two independent ways,
    and both are ordinary rather than exotic, so both are driven here directly.

    A substitution makes the bases disagree. Barcode extraction records the whitelist
    entry it resolved to in the component's own value tag, and the scTIP arm already
    emits that corrected value in its ``CB=`` header; because the two arms of one
    multiome experiment are joined downstream by barcode, an scRNA arm that re-derived
    the raw bases would present the same physical cell under two different barcodes and
    split its data in half.

    An indel makes the widths disagree. The matcher searches window lengths either side
    of the declared length and keeps the best-scoring one, so a corrected span can be a
    base narrower or wider than the component really is. Quality has no source but the
    read itself, so it must still be sliced out of ``r1_qual`` - but from the recorded
    start for the component's own declared length, never out to the matched end, or the
    emitted record's total width would drift read to read. The rare cost, accepted
    deliberately, is that one quality position in that fixed window can belong to the
    neighbouring component.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry write_read is exercised against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    @pytest.fixture
    def writer(self, chemistry: ChemistryBase) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(chemistry)

    def test_write_read_uses_the_corrected_annotation_value_not_the_raw_read_slice(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that emitted barcode bases come from the value tag, not from the read.

        This read's BC2 span covers a one-substitution misread while its BC2 tag holds
        the whitelist entry that misread was corrected to, so the two possible
        implementations - read the tag, or re-slice the read - produce visibly different
        records rather than the same one. The corrected base must appear at the
        substituted position, and the raw variant must not survive anywhere in the
        emitted record at all, which is what makes this arm agree with the scTIP arm's
        ``CB=`` header for the same physical cell instead of inventing a second one.

        Quality is asserted here too, unchanged: correcting the sequence must not move
        where quality is read from, since the header carries no quality to correct with.
        """
        bc3_seq = "G" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH
        differences = [
            index
            for index, (raw, corrected) in enumerate(zip(RAW_BC2_VARIANT, CORRECTED_BC2))
            if raw != corrected
        ]
        assert_that(differences).is_length(1)
        substitution = differences[0]

        annotation, r1_seq, r1_qual = build_indel_corrected_scrna_read(
            read_id="read-corrected-bc2",
            bc3_seq=bc3_seq,
            bc2_observed_seq=RAW_BC2_VARIANT,
            bc2_corrected_seq=CORRECTED_BC2,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=3,
            insert_seq="TATAGCCTA",
        )
        bc2_start, bc2_end = parse_span(annotation.get(position_key("BC2")))
        assert_that(r1_seq[bc2_start:bc2_end]).is_equal_to(RAW_BC2_VARIANT)

        header, seq, qual = write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_equal_to(bc3_seq + CORRECTED_BC2 + bc1_seq + umi_seq)
        assert_that(seq).is_length(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(seq[BARCODE_LENGTH + substitution]).is_equal_to(CORRECTED_BC2[substitution])
        assert_that(seq[BARCODE_LENGTH + substitution]).is_not_equal_to(
            RAW_BC2_VARIANT[substitution]
        )
        assert_that("\n".join((header, seq, qual))).does_not_contain(RAW_BC2_VARIANT)
        assert_that(qual).is_equal_to(r1_qual[:FIXED_BARCODES_RECORD_LENGTH])

    def test_write_read_emits_the_fixed_declared_width_for_an_indel_shortened_span(
        self, writer: ScrnaWriter, chemistry: ChemistryBase
    ) -> None:
        """Test that a 9bp recorded BC2 span still emits a full-width record.

        A deletion inside BC2 leaves the matcher's best window one base short, so the
        recorded span is 9bp for a component the chemistry declares as 10bp. The
        sequence half is unaffected - it comes from the corrected 10bp value tag - but
        the quality half would shrink the whole record to 37bp if it were sliced out to
        the matched span's end, and a record whose width drifts read to read is not the
        fixed-width barcode read STARsolo is being handed. Slicing the declared length
        from the recorded start keeps it at 38bp, at the accepted cost that BC2's last
        quality character here really belongs to BC1's first base; that overlap is
        asserted explicitly rather than left as a surprise.
        """
        bc3_seq = "G" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH
        bc2_observed_seq = CORRECTED_BC2[:-1]
        assert_that(bc2_observed_seq).is_length(BARCODE_LENGTH - 1)

        declared_total = (
            sum(
                comp.length
                for comp in chemistry.read_structure.get_components_by_type(
                    ReadComponentType.BARCODE
                )
            )
            + chemistry.umi_component().length
        )
        assert_that(declared_total).is_equal_to(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(FIXED_BARCODES_RECORD_LENGTH).is_equal_to(38)

        annotation, r1_seq, r1_qual = build_indel_corrected_scrna_read(
            read_id="read-indel-shortened-bc2",
            bc3_seq=bc3_seq,
            bc2_observed_seq=bc2_observed_seq,
            bc2_corrected_seq=CORRECTED_BC2,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=3,
            insert_seq="TATAGCCTAC",
        )
        bc2_start, bc2_end = parse_span(annotation.get(position_key("BC2")))
        assert_that(bc2_end - bc2_start).is_equal_to(BARCODE_LENGTH - 1)

        header, seq, qual = write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_length(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(qual).is_length(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(seq).is_equal_to(
            chemistry.construct_full_barcode(
                {"BC3": bc3_seq, "BC2": CORRECTED_BC2, "BC1": bc1_seq}
            )
            + umi_seq
        )

        bc2_quality = qual[BARCODE_LENGTH : 2 * BARCODE_LENGTH]
        assert_that(bc2_quality).is_equal_to(r1_qual[bc2_start : bc2_start + BARCODE_LENGTH])
        assert_that(bc2_quality).is_not_equal_to(r1_qual[bc2_start:bc2_end])
        assert_that(bc2_quality[-1]).is_equal_to(r1_qual[bc2_end])

    def test_write_read_emits_the_fixed_declared_width_for_an_indel_lengthened_span(
        self, writer: ScrnaWriter, chemistry: ChemistryBase
    ) -> None:
        """Test that an 11bp recorded BC2 span still emits a full-width record.

        The mirror of the shortened case: an insertion leaves the matcher's best window
        one base long, so slicing quality out to the matched span's end would stretch
        the record to 39bp. The declared length has to win in this direction too, which
        means the quality character at the recorded span's own last position is
        deliberately dropped rather than emitted - asserted here by its absence from the
        record, which is meaningful because this read's quality characters are all
        distinct.
        """
        bc3_seq = "G" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH
        bc2_observed_seq = CORRECTED_BC2 + "T"
        assert_that(bc2_observed_seq).is_length(BARCODE_LENGTH + 1)

        annotation, r1_seq, r1_qual = build_indel_corrected_scrna_read(
            read_id="read-indel-lengthened-bc2",
            bc3_seq=bc3_seq,
            bc2_observed_seq=bc2_observed_seq,
            bc2_corrected_seq=CORRECTED_BC2,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=3,
            insert_seq="TATAGCCT",
        )
        bc2_start, bc2_end = parse_span(annotation.get(position_key("BC2")))
        assert_that(bc2_end - bc2_start).is_equal_to(BARCODE_LENGTH + 1)

        header, seq, qual = write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_length(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(qual).is_length(FIXED_BARCODES_RECORD_LENGTH)
        assert_that(seq).is_equal_to(
            chemistry.construct_full_barcode(
                {"BC3": bc3_seq, "BC2": CORRECTED_BC2, "BC1": bc1_seq}
            )
            + umi_seq
        )

        bc2_quality = qual[BARCODE_LENGTH : 2 * BARCODE_LENGTH]
        assert_that(bc2_quality).is_equal_to(r1_qual[bc2_start : bc2_start + BARCODE_LENGTH])
        assert_that(bc2_quality).is_not_equal_to(r1_qual[bc2_start:bc2_end])
        assert_that(qual).does_not_contain(r1_qual[bc2_end - 1])

    def test_write_read_never_fabricates_bases_or_quality(self, writer: ScrnaWriter) -> None:
        """Test that every emitted base and quality character came from somewhere real.

        Padding the record out to a fixed width with invented bases or invented quality
        was considered and rejected: a fixed width is worth having, but not at the price
        of data nobody sequenced. The indel-shortened read is the case where a padding
        implementation would be tempted, so it is the one used here. Every emitted
        quality character is checked against the exact ``r1_qual`` index it should have
        come from, and every emitted base against the annotation value it should have
        come from, which together leave no room for a filler character to hide anywhere
        in the record.
        """
        bc3_seq = "G" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH

        annotation, r1_seq, r1_qual = build_indel_corrected_scrna_read(
            read_id="read-no-fabrication",
            bc3_seq=bc3_seq,
            bc2_observed_seq=CORRECTED_BC2[:-1],
            bc2_corrected_seq=CORRECTED_BC2,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=3,
            insert_seq="TATAGCCTAC",
        )

        source_indices: list[int] = []
        for name in BARCODE_NAMES_IN_STRUCTURE_ORDER:
            start = parse_span(annotation.get(position_key(name)))[0]
            source_indices.extend(range(start, start + BARCODE_LENGTH))
        umi_start = parse_span(annotation.get(position_key("UMI")))[0]
        source_indices.extend(range(umi_start, umi_start + UMI_LENGTH))

        header, seq, qual = write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(qual).is_length(len(source_indices))
        for offset, index in enumerate(source_indices):
            assert_that(qual[offset]).is_equal_to(r1_qual[index])
        assert_that(set(qual) - set(r1_qual)).is_empty()

        assert_that(seq).is_equal_to(
            annotation.get("BC3")
            + annotation.get("BC2")
            + annotation.get("BC1")
            + annotation.get("UMI")
        )
        assert_that(set(seq) - set("ACGT")).is_empty()
        assert_that(seq).is_length(FIXED_BARCODES_RECORD_LENGTH)


# Barcode component names for the fabricated chemistry double below, deliberately
# chosen so that neither a hardcoded BC3/BC2/BC1 assumption nor an accidental
# alphabetical-sort bug could reproduce this declared order: sorting these three
# names alphabetically yields ("ALPHA", "GAMMA", "ZETA"), not this tuple.
FABRICATED_BARCODE_NAMES_IN_STRUCTURE_ORDER = ("ZETA", "ALPHA", "GAMMA")


class FabricatedChemistry:
    """Minimal chemistry-shaped test double exposing only what ScrnaWriter calls.

    ``ScrnaWriter.__init__`` and ``write_read`` never touch whitelists, matching
    tolerances, or any other ``ChemistryBase`` validation machinery: the only
    surface they actually use is ``umi_component()``, ``umi_right_anchor()``,
    ``read_structure``, and ``construct_full_barcode()``. This double supplies
    exactly those four, so a test can drive ``ScrnaWriter`` against a read
    structure the real, registered chemistries would never produce, without
    needing whitelist files or ``ChemistryBase.__post_init__`` validation to
    pass first.

    ``construct_full_barcode`` here is a real implementation rather than a
    canned string, and derives its order the same way ``ChemistryBase``'s does -
    by walking this double's own read structure - precisely because the test it
    serves is about concatenation order. A double that returned a hardcoded
    order would decide the answer the test is asking for.
    """

    def __init__(self, read_structure: ReadStructure) -> None:
        self.read_structure = read_structure

    def umi_component(self) -> ReadComponent:
        """Return this chemistry's single UMI component.

        Returns:
            The ``ReadComponent`` of type ``UMI`` declared in ``read_structure``.
        """
        return self.read_structure.get_component_by_name("UMI")

    def umi_right_anchor(self) -> None:
        """Return no right anchor for the UMI.

        ``None`` is a legitimate value here: ``insert_start`` already treats it
        as "no anchor, cut point is the reference itself", so this fabricated
        chemistry does not need a homopolymer or primer component to be a valid
        double for ``ScrnaWriter``.
        """
        return None

    def construct_full_barcode(self, barcodes: dict[str, str]) -> str:
        """Concatenate the given barcode values in this chemistry's own declared order.

        Mirrors ``ChemistryBase.construct_full_barcode``, including its refusal
        to guess at a missing component, but derives the order from this
        double's own ``read_structure`` so the fabricated ZETA/ALPHA/GAMMA
        layout is honoured rather than overridden.

        Args:
            barcodes: Barcode component name to sequence value.

        Returns:
            The declared barcode components' values joined in read-structure order.

        Raises:
            ValueError: If ``barcodes`` omits a declared barcode component.
        """
        parts = []
        for comp in self.read_structure.get_components_by_type(ReadComponentType.BARCODE):
            value = barcodes.get(comp.name)
            if value is None:
                raise ValueError(
                    f"Missing barcode component '{comp.name}' for full barcode construction."
                )
            parts.append(value)
        return "".join(parts)


class TestScrnaWriterBarcodeOrderIsStructural:
    """Guards against the barcode-concatenation order ever hardcoding BC3/BC2/BC1.

    ``ScrnaWriter.__init__`` derives ``self.barcode_components`` generically, by
    walking
    ``chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)``,
    rather than naming ``BC3``/``BC2``/``BC1`` anywhere in the writer itself; and
    ``write_read`` collects each component's corrected value and quality slice by
    walking that tuple in order, handing the values to
    ``chemistry.construct_full_barcode``, which independently derives the same
    order from the same read structure.
    Every other test in this file exercises that mechanism only against the real
    ``carmack_custom_seq_1_0`` chemistry, whose barcodes happen to already be
    declared ``BC3``, ``BC2``, ``BC1`` in that order - so those tests alone
    cannot tell a genuinely structural implementation apart from one that
    silently hardcoded that specific name/order (or fell back to sorting names
    alphabetically, which for ``BC3``/``BC2``/``BC1`` would coincidentally
    still produce ``BC1``, ``BC2``, ``BC3`` - a different but equally wrong
    order that those tests also could not have caught).

    This test drives ``ScrnaWriter`` against a fabricated, unregistered
    chemistry-like double (see ``FabricatedChemistry`` above) whose barcode
    components are declared ``ZETA``, ``ALPHA``, ``GAMMA`` in that order: a
    sequence that matches neither ``BC3``/``BC2``/``BC1`` nor its own
    alphabetical sort (``ALPHA``, ``GAMMA``, ``ZETA``). The only way this test
    can pass is if barcode order in the synthesized barcodes FASTQ genuinely
    follows whatever order the read structure declares.

    This is a structural confirmation test, not a new-behaviour test: per the
    design, ``ScrnaWriter`` should already pass it unmodified.
    """

    def test_write_read_concatenates_fabricated_barcodes_in_declared_structure_order(
        self,
    ) -> None:
        """Test that barcode order in the barcodes FASTQ follows a reordered read structure.

        If ``write_read`` (or ``ScrnaWriter.__init__``) ever hardcoded
        ``BC3``+``BC2``+``BC1``, or fell back to sorting component names
        alphabetically, this fabricated chemistry's ``ZETA``/``ALPHA``/``GAMMA``
        order - which survives neither shortcut - would immediately produce the
        wrong concatenation, catching the regression this file's other tests
        structurally cannot.
        """
        zeta_seq = "T" * BARCODE_LENGTH
        alpha_seq = "G" * BARCODE_LENGTH
        gamma_seq = "C" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH

        components = [
            ReadComponent(name="ZETA", type=ReadComponentType.BARCODE, length=BARCODE_LENGTH),
            ReadComponent(name="ALPHA", type=ReadComponentType.BARCODE, length=BARCODE_LENGTH),
            ReadComponent(name="GAMMA", type=ReadComponentType.BARCODE, length=BARCODE_LENGTH),
            ReadComponent(name="UMI", type=ReadComponentType.UMI, length=UMI_LENGTH),
        ]
        chemistry = FabricatedChemistry(ReadStructure(components))
        writer = ScrnaWriter(chemistry)

        assert_that(tuple(comp.name for comp in writer.barcode_components)).is_equal_to(
            FABRICATED_BARCODE_NAMES_IN_STRUCTURE_ORDER
        )

        zeta_start = 0
        zeta_end = zeta_start + len(zeta_seq)
        alpha_start = zeta_end
        alpha_end = alpha_start + len(alpha_seq)
        gamma_start = alpha_end
        gamma_end = gamma_start + len(gamma_seq)
        umi_start = gamma_end
        umi_end = umi_start + len(umi_seq)

        r1_seq = zeta_seq + alpha_seq + gamma_seq + umi_seq
        r1_qual = "".join(chr(33 + (position % 50)) for position in range(len(r1_seq)))

        annotation = ReadAnnotation(read_id="read42")
        annotation.set("ZETA", zeta_seq)
        annotation.set(position_key("ZETA"), format_span(zeta_start, zeta_end))
        annotation.set("ALPHA", alpha_seq)
        annotation.set(position_key("ALPHA"), format_span(alpha_start, alpha_end))
        annotation.set("GAMMA", gamma_seq)
        annotation.set(position_key("GAMMA"), format_span(gamma_start, gamma_end))
        annotation.set("UMI", umi_seq)
        annotation.set(position_key("UMI"), format_span(umi_start, umi_end))

        # This fabricated chemistry declares no UMI right anchor at all, so the insert
        # starts at the UMI's own span end - the value insert_start returns when handed
        # no anchor, and the cut a dispatcher would hand write_read for such a read.
        cut = umi_end
        r2_name = "read42"
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)
        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        expected_seq = zeta_seq + alpha_seq + gamma_seq + umi_seq
        expected_qual = (
            r1_qual[zeta_start:zeta_end]
            + r1_qual[alpha_start:alpha_end]
            + r1_qual[gamma_start:gamma_end]
            + r1_qual[umi_start:umi_end]
        )

        header, seq, qual = parse_fastq_record(barcodes_stream)
        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_equal_to(expected_seq)
        assert_that(qual).is_equal_to(expected_qual)


class TestScrnaWriterMissingPositionTag:
    """Guards write_read itself against a header missing a required *_POS tag.

    TargetAssigner.validate_header and UmiExtractor.validate_header each validate only
    the first read of an entire run, once, at construction time, because their callers
    each own a whole read stream passed through one instance. ScrnaWriter has no such
    stream to validate against: write_read is called once per read, with no "first read
    of the run" to single out ahead of time, so the only place left to catch a missing
    *_POS tag is defensively, on every call, inside write_read itself. Without that
    guard, ann.get(key) returning None for a missing tag reaches
    parse_span(None).split(":"), which fails as a bare AttributeError that names
    neither the read nor the tag that was actually missing.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry write_read is exercised against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    @pytest.fixture
    def writer(self, chemistry: ChemistryBase) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(chemistry)

    def test_write_read_missing_barcode_position_tag_raises_value_error_naming_read_and_tag(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that a header missing BC1_POS raises a ValueError naming the read and tag.

        BC1_POS is the only tag dropped from an otherwise complete, valid annotation, so
        the only way this call can fail is the missing-tag guard itself, not some other
        malformed input. The raised message must contain both the read id and the exact
        missing key, so this test would fail if a future change swapped in a generic,
        unhelpful message instead of one naming what actually went wrong.
        """
        read_id = "read-missing-bc1-pos"
        bc3_seq = "G" * BARCODE_LENGTH
        bc2_seq = "C" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH
        insert_seq = "TATAGCCTCTCTTATACACATCTCCTC"

        bc3_start = 0
        bc3_end = bc3_start + len(bc3_seq)
        bc2_start = bc3_end
        bc2_end = bc2_start + len(bc2_seq)
        bc1_start = bc2_end
        bc1_end = bc1_start + len(bc1_seq)
        umi_start = bc1_end
        umi_end = umi_start + len(umi_seq)

        polyg_run_length = 4
        r1_seq = bc3_seq + bc2_seq + bc1_seq + umi_seq + ("G" * polyg_run_length) + insert_seq
        r1_qual = "".join(chr(33 + (position % 50)) for position in range(len(r1_seq)))
        cut = umi_end + polyg_run_length

        annotation = ReadAnnotation(read_id=read_id)
        annotation.set("BC3", bc3_seq)
        annotation.set(position_key("BC3"), format_span(bc3_start, bc3_end))
        annotation.set("BC2", bc2_seq)
        annotation.set(position_key("BC2"), format_span(bc2_start, bc2_end))
        # BC1's value tag is set, and only its position tag is left unset: the one
        # missing tag under test. Both halves of BC1 are read, for two different
        # purposes, so leaving both out would no longer single out the position guard.
        annotation.set("BC1", bc1_seq)
        annotation.set("UMI", umi_seq)
        annotation.set(position_key("UMI"), format_span(umi_start, umi_end))

        r2_name = read_id
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)
        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        with pytest.raises(ValueError) as exc_info:
            writer.write_read(
                annotation,
                r1_seq,
                r1_qual,
                cut,
                r2_name,
                r2_seq,
                r2_qual,
                r1_stream,
                r2_stream,
                barcodes_stream,
            )

        message = str(exc_info.value)
        assert_that(message).contains(read_id)
        assert_that(message).contains(position_key("BC1"))


class TestScrnaWriterMissingValueTag:
    """Guards write_read against a header carrying a component's position but not its value.

    The position tags and the value tags are written side by side by the same line of
    barcode extraction, so in practice a read has both or neither - but write_read now
    reads them for two different purposes, and a read that lost only its value tags
    fails in a far worse way than one that lost its position tags. A missing ``*_POS``
    reaches ``parse_span(None)`` and dies with an AttributeError; a missing value tag
    reaches string concatenation with ``None`` and dies with a TypeError naming neither
    the read nor the tag, or worse, reaches ``construct_full_barcode``, whose own
    ValueError names the component but has no idea which read it was looking at. Since
    write_read is called once per read with no first-read validation pass to lean on,
    the guard has to sit inside the call, the same way the position guard does.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry write_read is exercised against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    @pytest.fixture
    def writer(self, chemistry: ChemistryBase) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(chemistry)

    def test_write_read_missing_barcode_value_tag_raises_value_error_naming_read_and_tag(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that a header missing the BC1 value tag names the read and that tag.

        The annotation is complete apart from BC1's value tag - BC1_POS is still there,
        as are both halves of every other component - so the only thing that can fail
        this call is the missing-value guard. The message must name the value tag that
        was actually missing rather than the position tag beside it, since those two are
        one character apart in a log line and lead to entirely different conclusions
        about which stage went wrong.
        """
        read_id = "read-missing-bc1-value"
        bc3_seq = "G" * BARCODE_LENGTH
        bc2_seq = "C" * BARCODE_LENGTH
        bc1_seq = "T" * BARCODE_LENGTH
        umi_seq = "A" * UMI_LENGTH

        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id=read_id,
            bc3_seq=bc3_seq,
            bc2_seq=bc2_seq,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=4,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        del annotation.tags["BC1"]
        assert_that(annotation.get("BC1")).is_none()
        assert_that(annotation.get(position_key("BC1"))).is_not_none()

        with pytest.raises(ValueError) as exc_info:
            write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        message = str(exc_info.value)
        assert_that(message).contains(read_id)
        assert_that(message).contains("BC1")
        assert_that(message).does_not_contain(position_key("BC1"))

    def test_write_read_missing_umi_value_tag_raises_value_error_naming_read_and_tag(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that a header missing the UMI value tag fails the same way a barcode does.

        The UMI's corrected value is read from its own tag exactly as each barcode's is,
        and is the one component ``construct_full_barcode`` knows nothing about, so
        nothing downstream would catch its absence on the writer's behalf. Guarding it
        with the same check keeps a UMI-shaped failure from being reported as a bare
        TypeError from a string concatenation.
        """
        read_id = "read-missing-umi-value"

        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id=read_id,
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="A" * UMI_LENGTH,
            polyg_run_length=4,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        del annotation.tags["UMI"]
        assert_that(annotation.get("UMI")).is_none()
        assert_that(annotation.get(position_key("UMI"))).is_not_none()

        with pytest.raises(ValueError) as exc_info:
            write_barcodes_record(writer, annotation, r1_seq, r1_qual)

        message = str(exc_info.value)
        assert_that(message).contains(read_id)
        assert_that(message).contains("UMI")
        assert_that(message).does_not_contain(position_key("UMI"))


class TestScrnaWriterGoldenGzipRoundTrip:
    """The one test in this file that exercises the real gzip/pigz subprocess path.

    Every other test above writes into plain io.BytesIO() streams, which are faithful
    doubles for what write_read itself does - call stream.write() with already-encoded
    bytes - but never prove that a real GzipFile.open_write_stream() (a pigz or gzip
    child process piped to a file) actually receives, compresses, and flushes those
    bytes correctly, or that the resulting file decompresses back to the same content
    through FastqFile.open_read_iterator(). This test builds a small handful of
    synthetic scRNA reads, writes all of them into the same three real .fastq.gz
    streams the way a real run would, then reads each file back through FastqFile -
    which transparently decompresses - and checks the round trip reproduces every
    record exactly, in the order it was written.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryBase:
        """Provide the shipped chemistry write_read is exercised against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    @pytest.fixture
    def writer(self, chemistry: ChemistryBase) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(chemistry)

    def test_write_read_round_trips_several_reads_through_real_gzip_files(
        self, writer: ScrnaWriter, tmp_path: Path
    ) -> None:
        """Test that write_read's output survives a real gzip compress/decompress cycle.

        Three reads are built with distinct read ids, distinct barcode/UMI sequences,
        and distinct poly-G run lengths, then all three are written into the same three
        open GzipFile streams, matching how a real run accumulates many reads per
        output file rather than one file per read. Each read's expected trimmed,
        passthrough, and concatenated content is computed here from its own known
        component lengths and poly-G run length - the same explicit-slicing approach
        TestScrnaWriterWriteRead already uses - rather than by re-running write_read's
        own logic, so this test cannot pass merely because the code under test agrees
        with itself.
        """
        read_specs = (
            {
                "read_id": "golden-read-a",
                "bc3_seq": "A" * BARCODE_LENGTH,
                "bc2_seq": "C" * BARCODE_LENGTH,
                "bc1_seq": "G" * BARCODE_LENGTH,
                "umi_seq": "T" * UMI_LENGTH,
                "polyg_run_length": 3,
                "insert_seq": "ACGTACGTACGTACGTACGT",
                "r2_seq": "TTAACCGGTTAACCGGTTAA",
            },
            {
                "read_id": "golden-read-b",
                "bc3_seq": "T" * BARCODE_LENGTH,
                "bc2_seq": "G" * BARCODE_LENGTH,
                "bc1_seq": "A" * BARCODE_LENGTH,
                "umi_seq": "C" * UMI_LENGTH,
                "polyg_run_length": 6,
                "insert_seq": "TGCATGCATGCATGCATGCA",
                "r2_seq": "GGCCTTAAGGCCTTAAGGCC",
            },
            {
                "read_id": "golden-read-c",
                "bc3_seq": "ACACACACAC",
                "bc2_seq": "GTGTGTGTGT",
                "bc1_seq": "CACACACACA",
                "umi_seq": "GTGTGTGT",
                "polyg_run_length": 1,
                "insert_seq": "AAACCCGGGTTTACGTACGT",
                "r2_seq": "ACGTACGTACGTACGTACGT",
            },
        )

        built_reads = []
        expected_r1_records = []
        expected_r2_records = []
        expected_barcodes_records = []
        for spec in read_specs:
            r2_qual = "".join(chr(33 + (position % 40)) for position in range(len(spec["r2_seq"])))
            annotation, r1_seq, r1_qual = build_scrna_read(
                read_id=spec["read_id"],
                bc3_seq=spec["bc3_seq"],
                bc2_seq=spec["bc2_seq"],
                bc1_seq=spec["bc1_seq"],
                umi_seq=spec["umi_seq"],
                polyg_run_length=spec["polyg_run_length"],
                insert_seq=spec["insert_seq"],
            )

            bc3_end = len(spec["bc3_seq"])
            bc2_end = bc3_end + len(spec["bc2_seq"])
            bc1_end = bc2_end + len(spec["bc1_seq"])
            umi_end = bc1_end + len(spec["umi_seq"])
            cut = umi_end + spec["polyg_run_length"]
            built_reads.append((annotation, r1_seq, r1_qual, cut, spec["r2_seq"], r2_qual))

            expected_r1_records.append((spec["read_id"], r1_seq[cut:], r1_qual[cut:]))
            expected_r2_records.append((spec["read_id"], spec["r2_seq"], r2_qual))
            expected_barcodes_records.append(
                (
                    spec["read_id"],
                    spec["bc3_seq"] + spec["bc2_seq"] + spec["bc1_seq"] + spec["umi_seq"],
                    r1_qual[:bc3_end]
                    + r1_qual[bc3_end:bc2_end]
                    + r1_qual[bc2_end:bc1_end]
                    + r1_qual[bc1_end:umi_end],
                )
            )

        r1_path = tmp_path / "golden.r1.fastq.gz"
        r2_path = tmp_path / "golden.r2.fastq.gz"
        barcodes_path = tmp_path / "golden.barcodes.fastq.gz"

        with (
            GzipFile(str(r1_path)).open_write_stream() as r1_stream,
            GzipFile(str(r2_path)).open_write_stream() as r2_stream,
            GzipFile(str(barcodes_path)).open_write_stream() as barcodes_stream,
        ):
            for annotation, r1_seq, r1_qual, cut, r2_seq, r2_qual in built_reads:
                writer.write_read(
                    annotation,
                    r1_seq,
                    r1_qual,
                    cut,
                    annotation.read_id,
                    r2_seq,
                    r2_qual,
                    r1_stream,
                    r2_stream,
                    barcodes_stream,
                )

        r1_records = list(FastqFile(str(r1_path)).open_read_iterator(as_string=True))
        r2_records = list(FastqFile(str(r2_path)).open_read_iterator(as_string=True))
        barcodes_records = list(FastqFile(str(barcodes_path)).open_read_iterator(as_string=True))

        assert_that(r1_records).is_equal_to(expected_r1_records)
        assert_that(r2_records).is_equal_to(expected_r2_records)
        assert_that(barcodes_records).is_equal_to(expected_barcodes_records)


class TestScrnaWriterInsertCut:
    """Tests for insert_cut, the one place the scRNA arm's R1 trim point is derived.

    The trim point used to be worked out inside ``write_read``, which put it on the
    single thread that writes every unmatched read and put it after the read had already
    been dispatched. It is derived here, as a method of its own, so the dispatcher can
    ask for it while the read is still in a worker and carry the answer along with the
    read. Keeping the arithmetic on this class rather than rebuilding it beside the
    dispatcher is what stops one read getting two answers: the UMI's position key and the
    chemistry's UMI right anchor are already resolved and cached here, once, at
    construction, and ``write_read`` slices at whatever this returns.

    What the method must return is exactly ``insert_start``'s answer for the UMI's own
    recorded span end, with nothing laid over it. That includes a run which consumes the
    rest of the read, reported as reaching the read end rather than clamped back inside
    it: reporting it honestly is ``insert_start``'s own tested contract, and a clamp
    added here would quietly disagree with the arm that shares that function.
    """

    @pytest.fixture
    def writer(self) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(ChemistryFactory.get_chemistry(CHEMISTRY))

    @pytest.mark.parametrize("chemistry_name", SHIPPED_CHEMISTRIES)
    @pytest.mark.parametrize("polyg_run_length", [0, 1, 5])
    def test_insert_cut_is_insert_start_taken_off_the_umi_spans_end(
        self, chemistry_name: str, polyg_run_length: int
    ) -> None:
        """Test that insert_cut answers with insert_start's own answer for this read.

        Two independent expectations are asserted against, deliberately. The first is
        arithmetic this test already knows and the code under test does not participate
        in - the UMI's recorded span end plus the poly-G run the read was built to carry
        after it - which is what makes the assertion meaningful rather than circular. The
        second is ``insert_start`` itself, called the way the method is contracted to
        call it, which is what pins the two as the same computation rather than two that
        merely agree on the reads this file happens to build.

        A run length of zero is included because it is the one case where the anchor
        contributes nothing and the cut has to fall exactly on the span end, and both
        shipped chemistries are driven because they seat the UMI at different read
        offsets: a cut derived from the span the read records answers both, while one
        derived from an offset fixed by a chemistry could only ever answer one.
        """
        writer = ScrnaWriter(ChemistryFactory.get_chemistry(chemistry_name))
        annotation, r1_seq, _ = build_scrna_read(
            read_id="read-cut",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        umi_end = parse_span(annotation.get(position_key("UMI")))[1]

        cut = writer.insert_cut(annotation, r1_seq)

        assert_that(cut).is_equal_to(umi_end + polyg_run_length)
        assert_that(cut).is_equal_to(
            insert_start(reference=umi_end, anchor=writer.umi_right_anchor, seq=r1_seq)
        )

    def test_insert_cut_is_the_umi_span_end_when_the_chemistry_declares_no_right_anchor(
        self,
    ) -> None:
        """Test that a chemistry with no UMI right anchor cuts at the UMI span end itself.

        ``insert_start``'s no-anchor branch returns its reference unchanged, and this is
        the case that tells a method which really delegates to it apart from one that
        reimplemented the homopolymer scan: the read here still carries a poly-G run
        after its UMI, so an implementation that walked the run regardless of what the
        chemistry declares would return the run's end and fail here, while every read in
        the parametrized test above would still have passed.
        """
        components = [
            ReadComponent(name=name, type=ReadComponentType.BARCODE, length=BARCODE_LENGTH)
            for name in BARCODE_NAMES_IN_STRUCTURE_ORDER
        ]
        components.append(ReadComponent(name="UMI", type=ReadComponentType.UMI, length=UMI_LENGTH))
        writer = ScrnaWriter(FabricatedChemistry(ReadStructure(components)))
        assert_that(writer.umi_right_anchor).is_none()

        polyg_run_length = 5
        annotation, r1_seq, _ = build_scrna_read(
            read_id="read-no-anchor",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        umi_end = parse_span(annotation.get(position_key("UMI")))[1]

        cut = writer.insert_cut(annotation, r1_seq)

        assert_that(cut).is_equal_to(umi_end)
        assert_that(cut).is_not_equal_to(umi_end + polyg_run_length)
        assert_that(cut).is_equal_to(insert_start(reference=umi_end, anchor=None, seq=r1_seq))

    def test_insert_cut_reaches_the_read_end_when_the_anchor_run_terminates_the_read(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that a run consuming the rest of the read is reported, not clamped back.

        On a 2-colour instrument an unsequenced tail comes back as a run of the anchor
        base, so the scan can walk to the read's last base and the honest answer for
        where the insert starts is the read end. This method reports that answer
        unchanged; deciding what to do about a read that has no insert left is the
        caller's, taken where the outcome and the tallies are, and a clamp introduced
        here would take that decision away from it by pretending an insert remained.
        """
        polyg_run_length = 6
        annotation, r1_seq, _ = build_scrna_read(
            read_id="read-saturating-run",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="",
        )

        cut = writer.insert_cut(annotation, r1_seq)

        assert_that(cut).is_equal_to(len(r1_seq))
        assert_that(r1_seq[cut:]).is_equal_to("")

    def test_insert_cut_missing_umi_position_tag_raises_value_error_naming_read_and_tag(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that a header with no UMI_POS fails here, naming the read and the tag.

        This is the guard ``read_span`` already applies for every other span the writer
        reads, reached through a second caller. It matters more than it looks: the cut is
        now asked for while the read is still being dispatched, so this is where a read
        that never went through extract-umis is caught, and the message has to name both
        the read and the tag for that to be actionable rather than a bare parse failure
        somewhere inside span parsing.
        """
        read_id = "read-missing-umi-pos"
        annotation, r1_seq, _ = build_scrna_read(
            read_id=read_id,
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=4,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        del annotation.tags[position_key("UMI")]
        assert_that(annotation.get(position_key("UMI"))).is_none()

        with pytest.raises(ValueError) as exc_info:
            writer.insert_cut(annotation, r1_seq)

        message = str(exc_info.value)
        assert_that(message).contains(read_id)
        assert_that(message).contains(position_key("UMI"))


class TestScrnaWriterCutIsGivenNotRecomputed:
    """write_read slices R1 at the cut it is handed, and nothing else moves with it.

    Deciding where R1 is cut and performing the cut now happen in two different
    processes: the cut is computed while the read is being dispatched, in parallel, and
    arrives at this writer as a value. A writer that recomputed it would still produce
    the right output for every read whose recorded spans agree with the cut it was
    handed - which is every read in the rest of this file - so the only way to tell the
    two implementations apart is to hand over a cut that deliberately disagrees, and
    assert the output follows the argument. That is what these tests do, and it is why
    the cut they pass is one this writer would never have derived for the read.

    The other half is what must NOT follow the cut. ``r1_seq`` and ``r1_qual`` still
    arrive full and untrimmed, because the barcodes record's quality is sliced at each
    component's recorded start, in the original read's coordinates, and those
    coordinates only mean anything against the untrimmed string. So the same read written
    at two different cuts must produce two different R1 records and byte-identical
    barcodes records.
    """

    @pytest.fixture
    def writer(self) -> ScrnaWriter:
        """Provide a ScrnaWriter constructed against the shipped chemistry.

        Returns:
            A ``ScrnaWriter`` built from the ``carmack_custom_seq_1_0`` chemistry.
        """
        return ScrnaWriter(ChemistryFactory.get_chemistry(CHEMISTRY))

    @pytest.mark.parametrize(
        "cut_shift", [-2, 3], ids=["short of the run end", "past the run end"]
    )
    def test_write_read_slices_r1_at_the_cut_it_is_given(
        self, writer: ScrnaWriter, cut_shift: int
    ) -> None:
        """Test that the R1 record follows the given cut, not the one the read implies.

        The cut handed over here is deliberately wrong for this read - a couple of bases
        short of the poly-G run's end in one case, a few bases into the insert in the
        other - and is checked against the cut the writer itself would derive so that
        "deliberately wrong" is asserted rather than assumed. Both directions are driven
        because an implementation that quietly took the larger or the smaller of the two
        would still pass a test that only ever pushed the cut one way.

        The R1 record must be the read sliced at the argument, in sequence and in
        quality alike, and must NOT equal the read sliced at the derived cut; any
        implementation that recomputes the cut for itself fails on that second
        assertion.
        """
        polyg_run_length = 5
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read-given-cut",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        derived_cut = polyg_cut(annotation, polyg_run_length)
        assert_that(derived_cut).is_equal_to(writer.insert_cut(annotation, r1_seq))
        given_cut = derived_cut + cut_shift
        assert_that(given_cut).is_not_equal_to(derived_cut)

        r2_name = "read-given-cut"
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)
        r1_stream = io.BytesIO()
        r2_stream = io.BytesIO()
        barcodes_stream = io.BytesIO()

        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            given_cut,
            r2_name,
            r2_seq,
            r2_qual,
            r1_stream,
            r2_stream,
            barcodes_stream,
        )

        header, seq, qual = parse_fastq_record(r1_stream)
        assert_that(header).is_equal_to(annotation.read_id)
        assert_that(seq).is_equal_to(r1_seq[given_cut:])
        assert_that(qual).is_equal_to(r1_qual[given_cut:])
        assert_that(seq).is_not_equal_to(r1_seq[derived_cut:])

    def test_write_read_barcodes_and_r2_records_do_not_move_with_the_cut(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that only the R1 record changes when the same read is written at two cuts.

        The barcodes record's quality is sliced at each component's recorded start, and
        those starts are coordinates in the original, untrimmed read - which is exactly
        why the untrimmed sequence and quality are what this writer is handed, rather
        than a read already cut down to its insert. Writing one read twice, at two cuts
        far enough apart to straddle the barcode segments, is what shows the barcodes
        record does not silently follow the trim. R2 is asserted alongside it because it
        is passed through untouched and has no business moving either; the R1 records are
        asserted to differ so that the whole test cannot pass by writing nothing at all.
        """
        polyg_run_length = 4
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read-two-cuts",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        r2_name = "read-two-cuts"
        r2_seq = "ACGT" * 5
        r2_qual = "I" * len(r2_seq)

        records = []
        for cut in (0, polyg_cut(annotation, polyg_run_length)):
            r1_stream = io.BytesIO()
            r2_stream = io.BytesIO()
            barcodes_stream = io.BytesIO()
            writer.write_read(
                annotation,
                r1_seq,
                r1_qual,
                cut,
                r2_name,
                r2_seq,
                r2_qual,
                r1_stream,
                r2_stream,
                barcodes_stream,
            )
            records.append(
                (
                    parse_fastq_record(r1_stream),
                    parse_fastq_record(r2_stream),
                    parse_fastq_record(barcodes_stream),
                )
            )

        first, second = records
        assert_that(first[2]).is_equal_to(second[2])
        assert_that(first[1]).is_equal_to(second[1])
        assert_that(first[0]).is_not_equal_to(second[0])

    def test_write_read_barcode_and_umi_quality_windows_stay_in_original_coordinates(
        self, writer: ScrnaWriter
    ) -> None:
        """Test that each quality window is read from the untrimmed read's own coordinates.

        The previous test shows the barcodes record does not move with the cut; this one
        says where it is actually read from, so that a writer which happened to produce a
        stable but wrong window could not satisfy both. Every expected window here is
        computed from the spans recorded on the annotation, applied to the full ``r1_qual``
        the writer was handed, while the cut passed alongside sits well past all of them -
        the arrangement in which an implementation that sliced quality out of the trimmed
        insert would emit an unmistakably different record rather than a subtly shifted
        one.
        """
        polyg_run_length = 4
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read-original-coordinates",
            bc3_seq="A" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=polyg_run_length,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
        cut = polyg_cut(annotation, polyg_run_length)
        expected_qual = ""
        for name in BARCODE_NAMES_IN_STRUCTURE_ORDER:
            start = parse_span(annotation.get(position_key(name)))[0]
            expected_qual += r1_qual[start : start + BARCODE_LENGTH]
        umi_start = parse_span(annotation.get(position_key("UMI")))[0]
        expected_qual += r1_qual[umi_start : umi_start + UMI_LENGTH]
        assert_that(cut).is_greater_than(umi_start + UMI_LENGTH)

        r2_seq = "ACGT" * 5
        barcodes_stream = io.BytesIO()
        writer.write_read(
            annotation,
            r1_seq,
            r1_qual,
            cut,
            annotation.read_id,
            r2_seq,
            "I" * len(r2_seq),
            io.BytesIO(),
            io.BytesIO(),
            barcodes_stream,
        )

        qual = parse_fastq_record(barcodes_stream)[2]
        assert_that(qual).is_equal_to(expected_qual)
        assert_that(qual).is_length(FIXED_BARCODES_RECORD_LENGTH)

    def test_write_read_takes_the_cut_immediately_after_the_r1_sequence_and_quality(
        self,
    ) -> None:
        """Test that the cut sits with the rest of R1 in the parameter list, not apart from it.

        Every one of these arguments is passed positionally by the driver, so their order
        is the contract rather than a detail. The cut belongs directly after the sequence
        and quality it applies to: those three describe one read, and separating them -
        putting the cut after the R2 triple, or last with the streams - would let a
        caller mismatch a cut with a read it does not belong to and still be written
        without complaint.
        """
        parameters = list(signature(ScrnaWriter.write_read).parameters)

        assert_that(parameters).is_equal_to(
            [
                "self",
                "ann",
                "r1_seq",
                "r1_qual",
                "cut",
                "r2_name",
                "r2_seq",
                "r2_qual",
                "r1_stream",
                "r2_stream",
                "barcodes_stream",
            ]
        )
