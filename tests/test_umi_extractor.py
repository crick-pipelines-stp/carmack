"""Tests for the extract-umis module.

The extractor reads an annotated R1 FASTQ and takes the UMI as a fixed-length
slice starting where its left anchor ends (BC1, read from the header ``BC1_POS``
tag), annotating the read with ``UMI`` and ``UMI_POS`` tags. Nothing is searched for, so most of
these tests are about what the slice contains whatever follows it, which reads
are skipped, and the reconciling stats; the rest cover the report and the CLI
wiring.
"""

import gzip
from pathlib import Path
from unittest import mock

import pytest
from assertpy import assert_that
from click.testing import CliRunner

import carmack.__main__
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_extractor import UmiExtractor
from carmack.umi.umi_reporting import UmiExtractionStats

CHEMISTRY = "carmack_custom_seq_1_0"
DUMMY_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"

# Layout used by all synthetic reads: BC1 occupies [BC1_START, UMI_START) and the
# UMI begins where BC1 ends. Only the BC1_POS *end* is consumed by the extractor.
BC1_START = 10
UMI_START = 20
TAIL = "TTTTTT"

# The chemistry's fixed UMI length, spelled out so a change to it is visible here.
UMI_LENGTH = 8

# Default barcode-component sequences carried by synthetic annotated reads. They
# mirror the BC1/BC2/BC3 tags written by barcode extraction, which the extractor
# checks the first read for.
BC1_SEQ = "AAAAAAAAAA"
BC2_SEQ = "CCCCCCCCCC"
BC3_SEQ = "GGGGGGGGGG"


def make_read_header(
    read_id: str,
    umi_start: int = UMI_START,
    bc1: str = BC1_SEQ,
    bc2: str = BC2_SEQ,
    bc3: str = BC3_SEQ,
) -> str:
    """Render a header carrying the BC1_POS anchor and the BC1/BC2/BC3 barcodes."""
    ann = ReadAnnotation(read_id=read_id)
    ann.set(position_key("BC1"), format_span(BC1_START, umi_start))
    ann.set("BC1", bc1)
    ann.set("BC2", bc2)
    ann.set("BC3", bc3)
    return ann.render()


def make_read(
    read_id: str,
    umi_seq: str,
    run_len: int,
    umi_start: int = UMI_START,
    bc1: str = BC1_SEQ,
    bc2: str = BC2_SEQ,
    bc3: str = BC3_SEQ,
) -> tuple[str, str, str]:
    """Build a synthetic ``(header, seq, qual)`` annotated read.

    The sequence is ``A * umi_start`` + ``umi_seq`` + ``G * run_len`` + tail, so
    the poly-G run begins exactly at ``umi_start + len(umi_seq)``. Passing a
    ``umi_seq`` shorter or longer than the chemistry's fixed length is how a read
    whose run starts early or late is built.
    """
    seq = "A" * umi_start + umi_seq + "G" * run_len + TAIL
    return make_read_header(read_id, umi_start, bc1, bc2, bc3), seq, "I" * len(seq)


def write_fastq(path, records: list[tuple[str, str, str]]) -> None:
    """Write ``(header, seq, qual)`` records to a gzipped FASTQ file."""
    with gzip.open(path, "wt") as handle:
        for header, seq, qual in records:
            handle.write(f"@{header}\n{seq}\n+\n{qual}\n")


def read_fastq(path) -> list[tuple[str, str, str, str]]:
    """Read a gzipped FASTQ file back into ``(header, seq, plus, qual)`` records."""
    with gzip.open(path, "rt") as handle:
        lines = handle.read().splitlines()
    return [tuple(lines[i : i + 4]) for i in range(0, len(lines), 4)]


def extracted_umis(path) -> dict[str, str]:
    """Return the ``read_id -> UMI`` mapping from an output FASTQ."""
    return {
        ReadAnnotation.parse(header[1:]).read_id: ReadAnnotation.parse(header[1:]).get("UMI")
        for header, _seq, _plus, _qual in read_fastq(path)
    }


@pytest.fixture
def build_extractor(tmp_path):
    """Return a factory that writes records to a FASTQ and builds an extractor."""

    def build(records: list[tuple[str, str, str]], name: str = "SK462.r1_annotated.fastq.gz"):
        fastq_path = tmp_path / name
        write_fastq(fastq_path, records)
        return UmiExtractor(str(fastq_path), CHEMISTRY)

    return build


class TestUmiExtractorConstruction:
    """Up-front validation and parameter resolution in the constructor."""

    def test_resolves_parameters_from_chemistry(self) -> None:
        extractor = UmiExtractor(DUMMY_FASTQ, CHEMISTRY)
        assert_that(extractor.umi_name).is_equal_to("UMI")
        assert_that(extractor.umi_length).is_equal_to(UMI_LENGTH)
        assert_that(extractor.anchor_pos_key).is_equal_to("BC1_POS")
        assert_that(extractor.umi_offset).is_equal_to(0)
        assert_that(extractor.anchor_base).is_equal_to("G")

    @pytest.mark.parametrize("attribute", ["umi_length_tolerance", "polyg_min_run", "left_key"])
    def test_removed_window_parameters_are_gone(self, attribute: str) -> None:
        """The length window and its min-run gate no longer exist on the extractor.

        Asserted rather than merely deleted, because a fixed-length slice that
        quietly regained a tolerance would still pass every other test here.
        """
        extractor = UmiExtractor(DUMMY_FASTQ, CHEMISTRY)
        assert_that(hasattr(extractor, attribute)).is_false()

    def test_chemistry_without_umi_raises(self) -> None:
        with pytest.raises(ValueError, match="declares no UMI component"):
            UmiExtractor(DUMMY_FASTQ, "hydrop")

    def test_invalid_chemistry_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            UmiExtractor(DUMMY_FASTQ, "does_not_exist")

    def test_non_homopolymer_neighbour_constructs_with_no_anchor_base(self) -> None:
        """A UMI with no homopolymer 3' of it extracts fine and reports no anchor base.

        The right neighbour is diagnostic only now, so its absence degrades the
        report's anchor-run section rather than failing the run. This replaces the
        constructor guard that used to reject such a chemistry outright.
        """
        chemistry = UmiExtractor(DUMMY_FASTQ, CHEMISTRY).chemistry
        with mock.patch.object(type(chemistry), "umi_right_anchor", return_value=None):
            extractor = UmiExtractor(DUMMY_FASTQ, CHEMISTRY)
        assert_that(extractor.anchor_base).is_none()


class TestExtractUmis:
    """End-to-end extraction over synthetic annotated FASTQ files."""

    def test_umi_is_the_fixed_window_after_the_left_anchor(
        self, build_extractor, tmp_path
    ) -> None:
        """Every accepted read yields exactly the chemistry's length, whatever follows.

        The three reads place the poly-G run one base early, exactly where the
        layout says, and one base late. Under the old two-sided window these came
        out at 7, 8 and 9 bases; the slice makes all three 8.
        """
        records = [
            make_read("early", "ACTACTA", 4),
            make_read("exact", "ACTACTAC", 4),
            make_read("late", "ACTACTACT", 4),
        ]
        stats = build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.accepted).is_equal_to(3)
        assert_that(extracted_umis(tmp_path / "out.r1_umi.fastq.gz")).is_equal_to(
            {"early": "ACTACTAG", "exact": "ACTACTAC", "late": "ACTACTAC"}
        )

    def test_read_with_no_anchor_run_is_still_accepted(self, build_extractor, tmp_path) -> None:
        """A read presenting no poly-G at all is accepted, not rejected.

        This is the behaviour change with the widest effect: these reads used to
        be dropped from the output entirely.
        """
        no_g = (make_read_header("nog"), "A" * 40, "I" * 40)
        stats = build_extractor([no_g]).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.accepted).is_equal_to(1)
        assert_that(extracted_umis(tmp_path / "out.r1_umi.fastq.gz")).is_equal_to(
            {"nog": "AAAAAAAA"}
        )
        assert_that(stats.homopolymer_run_counts).is_equal_to({0: 1})

    def test_umi_containing_n_is_extracted_unfiltered(self, build_extractor, tmp_path) -> None:
        """An ambiguous base in the UMI is carried through untouched.

        Correction used to drop these, because a sentinel base collided with its
        padding character. With no correction there is no sentinel and no reason
        to treat the read differently from any other.
        """
        records = [make_read("ambiguous", "ACTNCTAC", 4)]
        build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(extracted_umis(tmp_path / "out.r1_umi.fastq.gz")).is_equal_to(
            {"ambiguous": "ACTNCTAC"}
        )

    def test_read_too_short_for_the_umi_is_counted_truncated(
        self, build_extractor, tmp_path
    ) -> None:
        short = (make_read_header("short"), "A" * (UMI_START + UMI_LENGTH - 1), "I" * 27)
        stats = build_extractor([short]).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.truncated).is_equal_to(1)
        assert_that(stats.accepted).is_equal_to(0)
        assert_that(read_fastq(tmp_path / "out.r1_umi.fastq.gz")).is_empty()

    def test_read_ending_exactly_at_the_umi_end_is_accepted(
        self, build_extractor, tmp_path
    ) -> None:
        """The boundary is inclusive: a read that stops at the UMI's last base counts."""
        exact = (make_read_header("exact"), "A" * (UMI_START + UMI_LENGTH), "I" * 28)
        stats = build_extractor([exact]).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.accepted).is_equal_to(1)
        assert_that(stats.truncated).is_equal_to(0)

    def test_missing_left_anchor_excluded(self, build_extractor, tmp_path) -> None:
        good = make_read("good", "ACTACTAC", 4)
        header_no_bc1 = ReadAnnotation(read_id="nobc1").render()
        bad = (header_no_bc1, "A" * 40, "I" * 40)
        stats = build_extractor([good, bad]).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.missing_left_anchor).is_equal_to(1)
        assert_that(stats.accepted).is_equal_to(1)
        assert_that(list(extracted_umis(tmp_path / "out.r1_umi.fastq.gz"))).is_equal_to(["good"])

    def test_umi_pos_span_is_written_and_indexes_the_umi(self, build_extractor, tmp_path) -> None:
        """The span is the fixed slice, and slicing the read by it returns the UMI.

        Asserted against the read rather than against the expected numbers alone,
        so the tag is checked to be a usable coordinate and not merely present.
        """
        records = [make_read("a", "ACTACTAC", 4), make_read("b", "GGCATCAT", 5)]
        build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        for header, seq, _plus, _qual in read_fastq(tmp_path / "out.r1_umi.fastq.gz"):
            ann = ReadAnnotation.parse(header[1:])
            start, end = parse_span(ann.get(position_key("UMI")))
            assert_that((start, end)).is_equal_to((UMI_START, UMI_START + UMI_LENGTH))
            assert_that(seq[start:end]).is_equal_to(ann.get("UMI"))

    def test_umi_pos_follows_the_recorded_anchor_rather_than_a_nominal_start(
        self, build_extractor, tmp_path
    ) -> None:
        """An upstream indel moves BC1_POS, and the span moves with it.

        The whole point of measuring off the recorded anchor is that the span is
        a read coordinate, not the layout's nominal one.
        """
        shifted = 23
        records = [make_read("shifted", "ACTACTAC", 4, umi_start=shifted)]
        build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        header, seq, _plus, _qual = read_fastq(tmp_path / "out.r1_umi.fastq.gz")[0]
        ann = ReadAnnotation.parse(header[1:])
        assert_that(parse_span(ann.get(position_key("UMI")))).is_equal_to(
            (shifted, shifted + UMI_LENGTH)
        )
        assert_that(seq[shifted : shifted + UMI_LENGTH]).is_equal_to("ACTACTAC")

    def test_no_corrected_umi_tag_or_map_is_written(self, build_extractor, tmp_path) -> None:
        """Correction is gone: no UB tag, and no umi_map.tsv alongside the outputs."""
        build_extractor([make_read("a", "ACTACTAC", 4)]).extract_umis(
            output_dir=str(tmp_path), prefix="out"
        )

        text = gzip.decompress((tmp_path / "out.r1_umi.fastq.gz").read_bytes()).decode()
        assert_that(text).does_not_contain("UB")
        assert_that((tmp_path / "out.umi_map.tsv").exists()).is_false()
        assert_that(sorted(p.name for p in Path(tmp_path).glob("out.*"))).is_equal_to(
            ["out.r1_umi.fastq.gz", "out.umi_stats.txt"]
        )

    def test_stats_reconcile(self, build_extractor, tmp_path) -> None:
        records = [
            make_read("a", "ACTACTAC", 4),
            make_read("b", "ACTACTA", 4),
            (ReadAnnotation(read_id="c").render(), "A" * 40, "I" * 40),
            (make_read_header("d"), "A" * (UMI_START + 1), "I" * (UMI_START + 1)),
        ]
        stats = build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.total_reads).is_equal_to(4)
        assert_that(stats.accepted).is_equal_to(2)
        assert_that(stats.missing_left_anchor).is_equal_to(1)
        assert_that(stats.truncated).is_equal_to(1)
        assert_that(stats.accepted + stats.missing_left_anchor + stats.truncated).is_equal_to(
            stats.total_reads
        )

    def test_anchor_run_is_measured_at_the_fixed_offset(self, build_extractor, tmp_path) -> None:
        """The tally counts the run from the base after the UMI, not from its true start.

        The early read's run genuinely spans five bases, but one of them was
        consumed by the slice, so four are left to count. That difference is the
        point: the section describes the layout as the slice sees it.
        """
        records = [
            make_read("exact", "ACTACTAC", 5),
            make_read("early", "ACTACTA", 5),
        ]
        stats = build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.homopolymer_run_counts).is_equal_to({5: 1, 4: 1})

    def test_every_accepted_read_is_measured(self, build_extractor, tmp_path) -> None:
        """The run tally covers exactly the accepted reads, so accepted is its denominator."""
        records = [
            make_read("a", "ACTACTAC", 4),
            make_read("b", "ACTACTAC", 6),
            (make_read_header("nog"), "A" * 40, "I" * 40),
        ]
        stats = build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(sum(stats.homopolymer_run_counts.values())).is_equal_to(stats.accepted)

    def test_output_preserves_input_order_seq_and_qual(self, build_extractor, tmp_path) -> None:
        r_first = make_read("first", "ACTACTAC", 4)
        r_second = make_read("second", "ACTACTA", 4)
        build_extractor([r_first, r_second]).extract_umis(output_dir=str(tmp_path), prefix="out")

        out = read_fastq(tmp_path / "out.r1_umi.fastq.gz")
        ids = [ReadAnnotation.parse(h[1:]).read_id for h, *_ in out]
        assert_that(ids).is_equal_to(["first", "second"])
        # seq / qual are carried through untouched.
        assert_that(out[0][1]).is_equal_to(r_first[1])
        assert_that(out[0][3]).is_equal_to(r_first[2])

    def test_prefix_derived_from_input_filename(self, build_extractor, tmp_path) -> None:
        extractor = build_extractor(
            [make_read("a", "ACTACTAC", 4)], name="SK462.r1_annotated.fastq.gz"
        )
        extractor.extract_umis(output_dir=str(tmp_path))

        assert_that((tmp_path / "SK462.r1_umi.fastq.gz").exists()).is_true()
        assert_that((tmp_path / "SK462.umi_stats.txt").exists()).is_true()

    def test_stats_file_reports_counts(self, build_extractor, tmp_path) -> None:
        records = [
            make_read("a", "ACTACTAC", 4),
            (make_read_header("b"), "A" * (UMI_START + 1), "I" * (UMI_START + 1)),
        ]
        build_extractor(records).extract_umis(output_dir=str(tmp_path), prefix="out")

        report = (tmp_path / "out.umi_stats.txt").read_text()
        assert_that(report).contains("Total reads: 2")
        assert_that(report).contains("Accepted: 1")
        assert_that(report).contains("truncated): 1")
        assert_that(report).contains("missing_left_anchor): 0")
        assert_that(report).does_not_contain("no_polyg_anchor")


class TestUmiExtractionStatsReport:
    """The stats value object and its report rendering."""

    @staticmethod
    def build(**overrides) -> UmiExtractionStats:
        """Return stats with the given fields overridden on a sane default."""
        fields = {
            "total_reads": 4,
            "accepted": 2,
            "missing_left_anchor": 1,
            "truncated": 1,
            "umi_length": UMI_LENGTH,
        }
        return UmiExtractionStats(**{**fields, **overrides})

    def test_report_includes_run_details_and_counts(self) -> None:
        report = self.build().get_report()
        assert_that(report).contains("# Carmack version:")
        assert_that(report).contains("# UMI Extraction Stats")
        assert_that(report).contains("Total reads: 4")
        assert_that(report).contains("Accepted: 2")
        assert_that(report).contains("Rejected (missing_left_anchor): 1")
        assert_that(report).contains("Rejected (truncated): 1")

    def test_report_states_the_fixed_umi_length(self) -> None:
        """The constant replaces the distribution a measured length would have earned."""
        assert_that(self.build().get_report()).contains(f"fixed {UMI_LENGTH} bases")

    def test_report_omits_the_length_distribution_section(self) -> None:
        """A one-bin histogram over a constant is noise, so it is not rendered."""
        assert_that(self.build().get_report()).does_not_contain("UMI Length Distribution")

    def test_report_omits_the_correction_section(self) -> None:
        assert_that(self.build().get_report()).does_not_contain("Correction")

    def test_report_handles_zero_reads_without_error(self) -> None:
        report = self.build(
            total_reads=0, accepted=0, missing_left_anchor=0, truncated=0
        ).get_report()
        assert_that(report).contains("Total reads: 0")
        assert_that(report).contains("0.00%")

    def test_report_includes_anchor_run_distribution(self) -> None:
        report = self.build(
            total_reads=3,
            accepted=3,
            missing_left_anchor=0,
            truncated=0,
            homopolymer_base="G",
            homopolymer_run_counts={3: 1, 4: 2},
        ).get_report()
        assert_that(report).contains("# Anchor G-run Length Distribution")
        assert_that(report).contains("\t3\t1")
        assert_that(report).contains("\t4\t2")

    def test_report_omits_the_anchor_section_when_no_base_is_known(self) -> None:
        """A chemistry with no homopolymer 3' of its UMI simply has nothing to report."""
        assert_that(self.build().get_report()).does_not_contain("Anchor")


class TestExtractUmisCli:
    """CLI wiring for the extract-umis command."""

    def test_cli_invokes_extractor(self, tmp_path) -> None:
        runner = CliRunner()
        with mock.patch("carmack.__main__.UmiExtractor", autospec=True) as mock_extractor:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [
                    "extract-umis",
                    DUMMY_FASTQ,
                    "--chemistry",
                    CHEMISTRY,
                    "--output_dir",
                    str(tmp_path),
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_extractor.assert_called_once_with(DUMMY_FASTQ, CHEMISTRY)
        mock_extractor.return_value.extract_umis.assert_called_once_with(str(tmp_path), None)

    @pytest.mark.parametrize("option", ["--raw", "--temp-dir", "--shard-count"])
    def test_removed_options_are_rejected(self, tmp_path, option: str) -> None:
        """The correction options are gone, so passing one is an error rather than a no-op."""
        runner = CliRunner()
        result = runner.invoke(
            carmack.__main__.carmack_cli,
            ["extract-umis", DUMMY_FASTQ, "--chemistry", CHEMISTRY, option],
        )

        assert_that(result.exit_code).is_not_equal_to(0)
        assert_that(result.output).contains("No such option")

    def test_cli_listed_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])
        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains("extract-umis")


class TestExtractUmisHeaderValidation:
    """The first read's header is validated against the supplied chemistry.

    This is the stage's only agreement check between the chemistry it was given
    and the FASTQ it was pointed at. Everything after it succeeds
    unconditionally, so without it a mismatched chemistry would run to completion
    and emit UMIs cut from the wrong coordinate.
    """

    def make_missing_bc3(self, read_id: str) -> tuple[str, str, str]:
        """An annotated read carrying BC1/BC2 (+BC1_POS) but no BC3 tag."""
        ann = ReadAnnotation(read_id=read_id)
        ann.set(position_key("BC1"), format_span(BC1_START, UMI_START))
        ann.set("BC1", BC1_SEQ)
        ann.set("BC2", BC2_SEQ)
        seq = "A" * UMI_START + "ACTACTAC" + "GGGG" + TAIL
        return ann.render(), seq, "I" * len(seq)

    def test_missing_barcode_tag_raises_clear_error(self, build_extractor, tmp_path) -> None:
        records = [self.make_missing_bc3("r1"), self.make_missing_bc3("r2")]
        extractor = build_extractor(records)

        with pytest.raises(ValueError, match="BC3"):
            extractor.extract_umis(output_dir=str(tmp_path), prefix="out")
        # Validation runs before any output is written.
        assert_that((tmp_path / "out.r1_umi.fastq.gz").exists()).is_false()

    def test_missing_left_anchor_on_first_read_raises(self, build_extractor, tmp_path) -> None:
        ann = ReadAnnotation(read_id="nobc1")
        ann.set("BC1", BC1_SEQ)
        ann.set("BC2", BC2_SEQ)
        ann.set("BC3", BC3_SEQ)
        bad = (ann.render(), "A" * 40, "I" * 40)
        extractor = build_extractor([bad])

        with pytest.raises(ValueError, match="BC1_POS"):
            extractor.extract_umis(output_dir=str(tmp_path), prefix="out")
