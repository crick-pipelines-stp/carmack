"""Tests for the scTIP per-target FASTQ header rendering and paired-read writer.

``prepare-reads`` fans matched reads out into one open (r1, r2) stream pair per target
index bucket. This module covers the two small, independently failing pieces that
fan-out is built from: ``render_sctip_header``, which builds the ``CB=``/``UR=`` header
scTIP downstream tooling expects by delegating entirely to
``ChemistryBase.construct_full_barcode`` for the barcode half and passing the UMI
argument straight through unmodified, and ``write_sctip_read``, which looks a target
index up in an already-open bucket map and writes one read's R1 and R2 records into it
via the same wire format ``FastqFile.write_read`` produces.

Both pieces are pure dispatch with no chemistry- or I/O-specific logic of their own:
header rendering never re-derives what counts as a valid barcode set, and the writer
never decides which reads belong in which bucket. So the two things worth pinning here
are that neither piece silently swallows the failure mode its delegate already defines
(a missing barcode component, an unopened bucket) and that the paired write is
byte-exact and does not leak across buckets when multiple buckets are open at once.
"""

import io
import re

import pytest
from assertpy import assert_that

# Importing the concrete chemistry module registers it with ChemistryFactory as a
# side effect of import, the same pattern test_insert_locator.py and
# test_target_assigner.py rely on to obtain a real chemistry instance below.
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (  # noqa: F401
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.prepare_reads.sctip_writer import (
    SctipBucketWriters,
    render_sctip_header,
    write_sctip_read,
)

CHEMISTRY = "carmack_custom_seq_1_0"

# A whitelist-shaped barcode dict for the shipped chemistry: BC1, BC2 and BC3 are
# concatenated in read-structure order (BC3 + BC2 + BC1) by construct_full_barcode.
BARCODES = {"BC1": "AAAAAAAAAA", "BC2": "CCCCCCCCCC", "BC3": "GGGGGGGGGG"}
UMI = "ACGTACGT"


def fastq_record(header: str, seq: str, qual: str) -> bytes:
    """Return the exact bytes FastqFile.write_read would put on the stream.

    Args:
        header: The already-rendered read name, without the leading '@'.
        seq: The read sequence.
        qual: The read quality string.

    Returns:
        The four-line, UTF-8 encoded FASTQ record.
    """
    return f"@{header}\n{seq}\n+\n{qual}\n".encode("utf-8")


@pytest.fixture
def chemistry() -> ChemistryBase:
    """Return the shipped chemistry instance header rendering is exercised against."""
    return ChemistryFactory.get_chemistry(CHEMISTRY)


class TestRenderSctipHeader:
    """The exact string render_sctip_header produces for a well-formed read."""

    @pytest.mark.parametrize(
        "read_id,bc1,bc2,bc3,umi",
        [
            ("read1", "AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG", "ACGTACGT"),
            ("SRR000001.42", "TTTTTTTTTT", "ACGTACGTAC", "GTACGTACGT", "TTTTGGGG"),
        ],
    )
    def test_render_sctip_header_builds_the_expected_string(
        self,
        chemistry: ChemistryBase,
        read_id: str,
        bc1: str,
        bc2: str,
        bc3: str,
        umi: str,
    ) -> None:
        """A normal read renders to '{read_id}|CB={BC3}{BC2}{BC1}|UR={UMI}' exactly."""
        barcodes = {"BC1": bc1, "BC2": bc2, "BC3": bc3}

        header = render_sctip_header(chemistry, read_id, barcodes, umi)

        assert_that(header).is_equal_to(f"{read_id}|CB={bc3}{bc2}{bc1}|UR={umi}")

    def test_render_sctip_header_has_no_embedded_whitespace(
        self, chemistry: ChemistryBase
    ) -> None:
        header = render_sctip_header(chemistry, "read1", BARCODES, UMI)

        assert_that(" " in header).is_false()
        assert_that(re.search(r"\s", header)).is_none()

    def test_render_sctip_header_cb_value_is_identical_to_construct_full_barcode(
        self, chemistry: ChemistryBase
    ) -> None:
        """CB=<value> must be exactly chemistry.construct_full_barcode's own output.

        Compared as an equal string extracted from the header, not merely re-derived
        from the same inputs, so a divergent concatenation order inside
        render_sctip_header itself would still be caught.
        """
        header = render_sctip_header(chemistry, "read1", BARCODES, UMI)
        expected_cb = chemistry.construct_full_barcode(BARCODES)

        cb_value = header.split("|CB=")[1].split("|UR=")[0]

        assert_that(cb_value).is_equal_to(expected_cb)
        assert_that(header).contains(f"|CB={expected_cb}|")

    @pytest.mark.parametrize(
        "umi",
        [
            "acgtACGT",  # mixed case: must not be normalised to one case
            "ACGTNNNN",  # ambiguity codes: passed through as-is
            "ACGTA",  # shorter than the chemistry's nominal UMI length: not padded/trimmed
        ],
    )
    def test_render_sctip_header_ur_value_is_the_umi_argument_verbatim(
        self, chemistry: ChemistryBase, umi: str
    ) -> None:
        """UR=<value> is exactly the umi argument: no correction, no case change, no trimming."""
        header = render_sctip_header(chemistry, "read1", BARCODES, umi)

        assert_that(header).ends_with(f"|UR={umi}")
        ur_value = header.split("|UR=")[1]
        assert_that(ur_value).is_equal_to(umi)


class TestRenderSctipHeaderPropagatesConstructFullBarcodeErrors:
    """render_sctip_header must not swallow construct_full_barcode's own validation."""

    @pytest.mark.parametrize("missing_component", ["BC1", "BC2", "BC3"])
    def test_missing_barcode_component_raises_value_error(
        self, chemistry: ChemistryBase, missing_component: str
    ) -> None:
        barcodes = dict(BARCODES)
        del barcodes[missing_component]

        with pytest.raises(ValueError):
            render_sctip_header(chemistry, "read1", barcodes, UMI)

    def test_empty_barcodes_dict_raises_value_error(self, chemistry: ChemistryBase) -> None:
        with pytest.raises(ValueError):
            render_sctip_header(chemistry, "read1", {}, UMI)


class TestWriteSctipReadBucketLookupFailure:
    """write_sctip_read must fail loudly, naming the tgidx, when no bucket is open for it."""

    def test_empty_writers_raises_key_error_naming_the_tgidx(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            write_sctip_read(
                writers={},
                tgidx="BADVALUE",
                header="read1|CB=AAAA|UR=TTTT",
                r1_seq="ACGT",
                r1_qual="IIII",
                r2_seq="TTTT",
                r2_qual="IIII",
            )

        assert_that(str(exc_info.value)).contains("BADVALUE")

    def test_tgidx_absent_from_populated_writers_raises_key_error_naming_the_tgidx(
        self,
    ) -> None:
        writers = {"AACCTTGG": SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO())}

        with pytest.raises(KeyError) as exc_info:
            write_sctip_read(
                writers=writers,
                tgidx="ZZZZZZZZ",
                header="read1|CB=AAAA|UR=TTTT",
                r1_seq="ACGT",
                r1_qual="IIII",
                r2_seq="TTTT",
                r2_qual="IIII",
            )

        assert_that(str(exc_info.value)).contains("ZZZZZZZZ")

    def test_unmatched_none_tgidx_against_real_whitelist_shaped_writers_raises_key_error(
        self,
    ) -> None:
        """A caller mistakenly routing an unmatched TGIDX=NONE read here must still fail.

        The driver builds `writers` only from chemistry.tgidx_whitelist() entries, so
        "NONE" is never a real key; it must raise the same KeyError as any other unknown
        tgidx rather than being treated as a special, silently-dropped case.
        """
        writers = {"AACCTTGG": SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO())}

        with pytest.raises(KeyError) as exc_info:
            write_sctip_read(
                writers=writers,
                tgidx="NONE",
                header="read1|CB=AAAA|UR=TTTT",
                r1_seq="ACGT",
                r1_qual="IIII",
                r2_seq="TTTT",
                r2_qual="IIII",
            )

        assert_that(str(exc_info.value)).contains("NONE")

    def test_failed_lookup_writes_nothing_to_the_open_bucket(self) -> None:
        """A failed lookup must not partially write into an unrelated open bucket."""
        writers = {"AACCTTGG": SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO())}

        with pytest.raises(KeyError):
            write_sctip_read(
                writers=writers,
                tgidx="NONE",
                header="read1|CB=AAAA|UR=TTTT",
                r1_seq="ACGT",
                r1_qual="IIII",
                r2_seq="TTTT",
                r2_qual="IIII",
            )

        assert_that(writers["AACCTTGG"].r1.getvalue()).is_equal_to(b"")
        assert_that(writers["AACCTTGG"].r2.getvalue()).is_equal_to(b"")


class TestWriteSctipReadGoldenMultiBucket:
    """Golden, byte-exact fixture proving cross-bucket write isolation."""

    BUCKET_A = "AACCTTGG"
    BUCKET_B = "GGTTCCAA"

    @pytest.fixture
    def writers(self) -> dict[str, SctipBucketWriters]:
        """Return two open buckets, each backed by fresh in-memory streams."""
        return {
            self.BUCKET_A: SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO()),
            self.BUCKET_B: SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO()),
        }

    def test_interleaved_writes_land_byte_exact_and_isolated_per_bucket(
        self, writers: dict[str, SctipBucketWriters]
    ) -> None:
        """Several reads addressed at two buckets, interleaved, land in write order.

        r1_seq/r1_qual and r2_seq/r2_qual are kept distinct per read below, so a
        byte-exact match against the expected records also proves r1 and r2 were not
        swapped: an accidental swap would fail this comparison exactly where it
        happens, not just leave the two streams merely "equal-looking".
        """
        reads = [
            (
                self.BUCKET_A,
                "readA1|CB=AAAA|UR=TTTT",
                "ACGTACGT",
                "IIIIIIII",
                "TTTTGGGG",
                "JJJJJJJJ",
            ),
            (
                self.BUCKET_B,
                "readB1|CB=CCCC|UR=GGGG",
                "GGGGCCCC",
                "KKKKKKKK",
                "AAAATTTT",
                "LLLLLLLL",
            ),
            (
                self.BUCKET_A,
                "readA2|CB=AAAA|UR=TTTT",
                "TTTTAAAA",
                "IIIIIIII",
                "CCCCGGGG",
                "JJJJJJJJ",
            ),
            (
                self.BUCKET_B,
                "readB2|CB=CCCC|UR=GGGG",
                "CCCCAAAA",
                "KKKKKKKK",
                "GGGGTTTT",
                "LLLLLLLL",
            ),
            (
                self.BUCKET_A,
                "readA3|CB=AAAA|UR=TTTT",
                "GGGGTTTT",
                "IIIIIIII",
                "AAAACCCC",
                "JJJJJJJJ",
            ),
        ]

        for tgidx, header, r1_seq, r1_qual, r2_seq, r2_qual in reads:
            result = write_sctip_read(writers, tgidx, header, r1_seq, r1_qual, r2_seq, r2_qual)
            assert_that(result).is_none()

        expected_r1 = {self.BUCKET_A: b"", self.BUCKET_B: b""}
        expected_r2 = {self.BUCKET_A: b"", self.BUCKET_B: b""}
        for tgidx, header, r1_seq, r1_qual, r2_seq, r2_qual in reads:
            expected_r1[tgidx] += fastq_record(header, r1_seq, r1_qual)
            expected_r2[tgidx] += fastq_record(header, r2_seq, r2_qual)

        assert_that(writers[self.BUCKET_A].r1.getvalue()).is_equal_to(expected_r1[self.BUCKET_A])
        assert_that(writers[self.BUCKET_A].r2.getvalue()).is_equal_to(expected_r2[self.BUCKET_A])
        assert_that(writers[self.BUCKET_B].r1.getvalue()).is_equal_to(expected_r1[self.BUCKET_B])
        assert_that(writers[self.BUCKET_B].r2.getvalue()).is_equal_to(expected_r2[self.BUCKET_B])

    def test_r1_and_r2_streams_never_receive_each_others_content(
        self, writers: dict[str, SctipBucketWriters]
    ) -> None:
        write_sctip_read(
            writers,
            self.BUCKET_A,
            "read1|CB=AAAA|UR=TTTT",
            r1_seq="R1ONLYSEQ",
            r1_qual="R1QUAL111",
            r2_seq="R2ONLYSEQ",
            r2_qual="R2QUAL222",
        )

        r1_bytes = writers[self.BUCKET_A].r1.getvalue()
        r2_bytes = writers[self.BUCKET_A].r2.getvalue()

        assert_that(r1_bytes).is_equal_to(
            fastq_record("read1|CB=AAAA|UR=TTTT", "R1ONLYSEQ", "R1QUAL111")
        )
        assert_that(r2_bytes).is_equal_to(
            fastq_record("read1|CB=AAAA|UR=TTTT", "R2ONLYSEQ", "R2QUAL222")
        )
        assert_that(r1_bytes).does_not_contain(b"R2ONLYSEQ")
        assert_that(r2_bytes).does_not_contain(b"R1ONLYSEQ")
