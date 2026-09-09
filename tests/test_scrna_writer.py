"""Tests for ScrnaWriter, which writes one TGIDX=NONE read into the scRNA arm's three
STARsolo-shaped output files: a trimmed R1 insert, a passthrough R2, and a synthesized
barcodes FASTQ built by slicing the original untrimmed R1 at each barcode/UMI
component's own recorded ``*_POS`` span.

This file currently covers only ``ScrnaWriter.__init__``: the handful of attributes it
resolves and caches once at construction time, rather than re-deriving on every read -
the UMI component itself, the annotation key its position is recorded under, its right
anchor (the anchor ``insert_start`` walks past to find where the trimmed R1 insert
begins), and the ordered tuple of barcode components that fixes the order barcode
segments are concatenated into the synthesized barcodes FASTQ. Getting any of these
wrong at construction time would silently corrupt every read the writer ever
processes, since none of them are re-checked per read.

The UMI-less-chemistry test exists because ``ScrnaWriter`` is documented to fail
clearly, rather than derive nonsense from a ``None`` component, when constructed
directly against a chemistry that cannot supply a UMI. Every read this class is ever
handed in production already passed through extract-umis, which transitively
guarantees a UMI component exists, but the class must not trust that transitively.

Later additions to this file will cover ``write_read``'s R1 trim, R2 passthrough, and
barcode+UMI concatenation order (using ``io.BytesIO()`` streams), a reordering guard
against a read structure whose barcodes are not declared BC3/BC2/BC1, a missing
``*_POS``-tag failure case, and one golden gzip round-trip test using
``GzipFile.open_write_stream()``.
"""

import io
from functools import cached_property
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

# Barcode component names in the order the shipped chemistry's read structure
# declares them - the order barcode_position_keys must reproduce.
BARCODE_NAMES_IN_STRUCTURE_ORDER = ("BC3", "BC2", "BC1")

NO_UMI_CHEMISTRY = "custom_seq_without_umi_for_scrna_writer"


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

    def test_caches_the_chemistrys_umi_component(self, chemistry: ChemistryBase) -> None:
        """Test that __init__ resolves chemistry.umi_component() once, onto self.umi.

        A later write_read call has no per-read way to re-derive this, so if
        this were ever wrong, cut placement for every read the writer
        processes would be wrong, and consistently wrong.
        """
        writer = ScrnaWriter(chemistry)

        assert_that(writer.umi).is_equal_to(chemistry.umi_component())
        assert_that(writer.umi.name).is_equal_to("UMI")
        assert_that(writer.umi.type).is_equal_to(ReadComponentType.UMI)

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

    def test_caches_barcode_position_keys_in_read_structure_order(
        self, chemistry: ChemistryBase
    ) -> None:
        """Test that barcode_position_keys follows read_structure order, not a hardcoded list.

        The synthesized barcodes FASTQ concatenates barcode segments in this
        order, so this is the one mechanism that makes that order follow
        whatever the chemistry declares rather than a name list baked into
        the writer.
        """
        writer = ScrnaWriter(chemistry)

        expected = tuple(position_key(name) for name in BARCODE_NAMES_IN_STRUCTURE_ORDER)
        assert_that(writer.barcode_position_keys).is_equal_to(expected)
        assert_that(writer.barcode_position_keys).is_equal_to(
            tuple(
                position_key(comp.name)
                for comp in chemistry.read_structure.get_components_by_type(
                    ReadComponentType.BARCODE
                )
            )
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


# Fixed component lengths used to build synthetic reads below, matching the real
# carmack_custom_seq_1_0 chemistry's own BC_CHUNK_LEN and UMI_LENGTH so the synthetic
# reads look like the real thing, even though write_read itself never checks a
# component's length against the chemistry - only against the span already recorded on
# the read's own annotation header.
BARCODE_LENGTH = 10
UMI_LENGTH = 8


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
    that read-structure order, then records each barcode/UMI component's own span on a
    ReadAnnotation exactly the way the pipeline's own extraction stages do -
    position_key plus format_span - since that header is the only input write_read is
    contracted to trust for locating any of them.

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
    annotation.set(position_key("BC3"), format_span(bc3_start, bc3_end))
    annotation.set(position_key("BC2"), format_span(bc2_start, bc2_end))
    annotation.set(position_key("BC1"), format_span(bc1_start, bc1_end))
    annotation.set(position_key("UMI"), format_span(umi_start, umi_end))
    if extra_tag is not None:
        annotation.set(*extra_tag)

    return annotation, r1_seq, r1_qual


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

        write_read must derive the cut point the same way insert_start's own dedicated
        test suite already verifies - the UMI span's end plus the real poly-G run length
        observed in this specific read, not any nominal length - and then slice both
        r1_seq and r1_qual identically from that point. Getting the cut wrong, or
        slicing seq and qual by different amounts, would leave every downstream
        STARsolo alignment reading a corrupted or misaligned insert.
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
        umi_start, umi_end = parse_span(annotation.get(position_key("UMI")))
        cut = umi_end + run_length
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
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read7",
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=4,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )
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
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id="read9",
            bc3_seq=bc3_seq,
            bc2_seq=bc2_seq,
            bc1_seq=bc1_seq,
            umi_seq=umi_seq,
            polyg_run_length=3,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
        )

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
        annotation, r1_seq, r1_qual = build_scrna_read(
            read_id=read_id,
            bc3_seq="G" * BARCODE_LENGTH,
            bc2_seq="C" * BARCODE_LENGTH,
            bc1_seq="T" * BARCODE_LENGTH,
            umi_seq="ACTACTAT",
            polyg_run_length=4,
            insert_seq="TATAGCCTCTCTTATACACATCTCCTC",
            extra_tag=("TGIDX", "NONE"),
        )
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


# Barcode component names for the fabricated chemistry double below, deliberately
# chosen so that neither a hardcoded BC3/BC2/BC1 assumption nor an accidental
# alphabetical-sort bug could reproduce this declared order: sorting these three
# names alphabetically yields ("ALPHA", "GAMMA", "ZETA"), not this tuple.
FABRICATED_BARCODE_NAMES_IN_STRUCTURE_ORDER = ("ZETA", "ALPHA", "GAMMA")


class FabricatedChemistry:
    """Minimal chemistry-shaped test double exposing only what ScrnaWriter calls.

    ``ScrnaWriter.__init__`` and ``write_read`` never touch whitelists, matching
    tolerances, or any other ``ChemistryBase`` validation machinery: the only
    surface they actually use is ``umi_component()``, ``umi_right_anchor()``, and
    ``read_structure``. This double supplies exactly those three, so a test can
    drive ``ScrnaWriter`` against a read structure the real, registered
    chemistries would never produce, without needing whitelist files or
    ``ChemistryBase.__post_init__`` validation to pass first.
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


class TestScrnaWriterBarcodeOrderIsStructural:
    """Guards against the barcode-concatenation order ever hardcoding BC3/BC2/BC1.

    ``ScrnaWriter.__init__`` derives ``self.barcode_position_keys`` generically,
    by walking
    ``chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)``,
    rather than naming ``BC3``/``BC2``/``BC1`` anywhere in the writer itself; and
    ``write_read`` concatenates barcode slices by walking that tuple in order.
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

        assert_that(writer.barcode_position_keys).is_equal_to(
            tuple(position_key(name) for name in FABRICATED_BARCODE_NAMES_IN_STRUCTURE_ORDER)
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
        annotation.set(position_key("ZETA"), format_span(zeta_start, zeta_end))
        annotation.set(position_key("ALPHA"), format_span(alpha_start, alpha_end))
        annotation.set(position_key("GAMMA"), format_span(gamma_start, gamma_end))
        annotation.set(position_key("UMI"), format_span(umi_start, umi_end))

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

        r1_seq = bc3_seq + bc2_seq + bc1_seq + umi_seq + ("G" * 4) + insert_seq
        r1_qual = "".join(chr(33 + (position % 50)) for position in range(len(r1_seq)))

        annotation = ReadAnnotation(read_id=read_id)
        annotation.set(position_key("BC3"), format_span(bc3_start, bc3_end))
        annotation.set(position_key("BC2"), format_span(bc2_start, bc2_end))
        # BC1_POS is deliberately left unset: the one missing tag under test.
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
            built_reads.append((annotation, r1_seq, r1_qual, spec["r2_seq"], r2_qual))

            bc3_end = len(spec["bc3_seq"])
            bc2_end = bc3_end + len(spec["bc2_seq"])
            bc1_end = bc2_end + len(spec["bc1_seq"])
            umi_end = bc1_end + len(spec["umi_seq"])
            cut = umi_end + spec["polyg_run_length"]

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
            for annotation, r1_seq, r1_qual, r2_seq, r2_qual in built_reads:
                writer.write_read(
                    annotation,
                    r1_seq,
                    r1_qual,
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
