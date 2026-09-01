"""Tests for the extract-umis module (SI-3 of the UMI epic).

The extractor reads an annotated R1 FASTQ, extracts the raw UMI lying between
its left anchor (BC1, read from the header ``BC1_POS`` tag) and the downstream
poly-G run, annotates the read with ``UMI`` / ``UMI_POS`` and reports the length
distribution. These tests build synthetic annotated reads and exercise the
greedy poly-G anchoring, the two-sided length window, the reconciling stats and
the CLI wiring.
"""

import gzip
from unittest import mock

import pytest
from assertpy import assert_that
from click.testing import CliRunner

import carmack.__main__
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_extractor import UmiExtractionStats, UmiExtractor

CHEMISTRY = "carmack_custom_seq_1_0"
DUMMY_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"

# Layout used by all synthetic reads: BC1 occupies [BC1_START, UMI_START) and the
# UMI begins where BC1 ends. Only the BC1_POS *end* is consumed by the extractor.
BC1_START = 10
UMI_START = 20
TAIL = "TTTTTT"


def make_read_header(read_id: str, umi_start: int = UMI_START) -> str:
    """Render a header carrying only the BC1_POS tag ending at ``umi_start``."""
    ann = ReadAnnotation(read_id=read_id)
    ann.set(position_key("BC1"), format_span(BC1_START, umi_start))
    return ann.render()


def make_read(
    read_id: str,
    umi_seq: str,
    run_len: int,
    umi_start: int = UMI_START,
) -> tuple[str, str, str]:
    """Build a synthetic ``(header, seq, qual)`` annotated read.

    The sequence is ``A * umi_start`` + ``umi_seq`` + ``G * run_len`` + tail, so
    the poly-G run begins exactly at ``umi_start + len(umi_seq)``.
    """
    seq = "A" * umi_start + umi_seq + "G" * run_len + TAIL
    return make_read_header(read_id, umi_start), seq, "I" * len(seq)


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


class FakeChemistryNoRightAnchor:
    """Chemistry stub that supports a left anchor but exposes no poly-G anchor."""

    def supports_umi_extraction(self) -> bool:
        return True

    def umi_left_anchor(self) -> ReadComponent:
        return ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10)

    def umi_component(self) -> ReadComponent:
        return ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=1)

    def umi_right_anchor(self) -> None:
        return None


class TestUmiExtractorConstruction:
    """Up-front validation and parameter resolution in the constructor."""

    def test_resolves_parameters_from_chemistry(self) -> None:
        extractor = UmiExtractor(DUMMY_FASTQ, CHEMISTRY)
        assert_that(extractor.umi_name).is_equal_to("UMI")
        assert_that(extractor.umi_length).is_equal_to(8)
        assert_that(extractor.umi_length_tolerance).is_equal_to(1)
        assert_that(extractor.polyg_base).is_equal_to("G")
        assert_that(extractor.polyg_min_run).is_equal_to(3)
        assert_that(extractor.left_key).is_equal_to("BC1_POS")

    def test_unsupported_chemistry_raises(self) -> None:
        with pytest.raises(ValueError, match="no usable left anchor"):
            UmiExtractor(DUMMY_FASTQ, "hydrop")

    def test_missing_right_anchor_raises(self) -> None:
        with mock.patch(
            "carmack.umi.umi_extractor.ChemistryFactory.get_chemistry",
            return_value=FakeChemistryNoRightAnchor(),
        ):
            with pytest.raises(ValueError, match="poly-G"):
                UmiExtractor(DUMMY_FASTQ, "fake")

    def test_invalid_chemistry_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            UmiExtractor(DUMMY_FASTQ, "does_not_exist")


class TestFindPolygStart:
    """Greedy earliest poly-G run detection within the two-sided window."""

    @pytest.fixture
    def extractor(self) -> UmiExtractor:
        return UmiExtractor(DUMMY_FASTQ, CHEMISTRY)

    def test_returns_run_start_inside_window(self, extractor: UmiExtractor) -> None:
        # UMI length 8 -> poly-G begins at umi_start + 8 (inside window 7..9).
        seq = "A" * UMI_START + "ACTACTAC" + "GGG" + TAIL
        assert_that(extractor.find_polyg_start(seq, UMI_START)).is_equal_to(UMI_START + 8)

    def test_picks_earliest_run_when_two_present(self, extractor: UmiExtractor) -> None:
        # Runs begin at both +7 and +9; the greedy scan returns the earliest.
        seq = "A" * UMI_START + "ACTACTA" + "GGG" + "A" + "GGG" + TAIL
        assert_that(extractor.find_polyg_start(seq, UMI_START)).is_equal_to(UMI_START + 7)

    def test_run_before_window_not_scanned(self, extractor: UmiExtractor) -> None:
        # A minimal run beginning at +6 leaves no 3-run starting within 7..9.
        seq = "A" * UMI_START + "ACTACT" + "GGG" + TAIL
        assert_that(extractor.find_polyg_start(seq, UMI_START)).is_none()

    def test_run_after_window_not_scanned(self, extractor: UmiExtractor) -> None:
        # UMI length 10 pushes the run to +10, beyond the window.
        seq = "A" * UMI_START + "ACTACTACTA" + "GGG" + TAIL
        assert_that(extractor.find_polyg_start(seq, UMI_START)).is_none()

    def test_slippage_same_run_start(self, extractor: UmiExtractor) -> None:
        short_run = "A" * UMI_START + "ACTACTAC" + "GGG" + TAIL
        long_run = "A" * UMI_START + "ACTACTAC" + "GGGGGG" + TAIL
        start_short = extractor.find_polyg_start(short_run, UMI_START)
        start_long = extractor.find_polyg_start(long_run, UMI_START)
        assert_that(start_short).is_equal_to(UMI_START + 8)
        assert_that(start_long).is_equal_to(start_short)

    def test_run_truncated_by_read_end_not_matched(self, extractor: UmiExtractor) -> None:
        # Only two G's remain at the read end: the min-run guard rejects it.
        seq = "A" * 7 + "GG"
        assert_that(extractor.find_polyg_start(seq, 0)).is_none()

    def test_no_polyg_in_read(self, extractor: UmiExtractor) -> None:
        seq = "A" * 40
        assert_that(extractor.find_polyg_start(seq, UMI_START)).is_none()


class TestExtractUmis:
    """End-to-end extraction over synthetic annotated FASTQ files."""

    @pytest.fixture
    def extractor_factory(self, tmp_path):
        def build(records: list[tuple[str, str, str]], name: str = "SK462.r1_annotated.fastq.gz"):
            fastq_path = tmp_path / name
            write_fastq(fastq_path, records)
            return UmiExtractor(str(fastq_path), CHEMISTRY)

        return build

    def test_accepts_lengths_7_8_9(self, extractor_factory, tmp_path) -> None:
        records = [
            make_read("r7", "ACTACTA", 4),
            make_read("r8", "ACTACTAC", 4),
            make_read("r9", "ACTACTACT", 4),
        ]
        extractor = extractor_factory(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.accepted).is_equal_to(3)
        assert_that(stats.length_counts).is_equal_to({7: 1, 8: 1, 9: 1})

        out = read_fastq(tmp_path / "out.r1_umi.fastq.gz")
        assert_that(out).is_length(3)
        umis = {
            ReadAnnotation.parse(header[1:]).read_id: ReadAnnotation.parse(header[1:]).get("UMI")
            for header, _seq, _plus, _qual in out
        }
        assert_that(umis).is_equal_to({"r7": "ACTACTA", "r8": "ACTACTAC", "r9": "ACTACTACT"})

    def test_rejects_lengths_6_and_10(self, extractor_factory, tmp_path) -> None:
        records = [
            make_read("short6", "ACTACT", 3),
            make_read("long10", "ACTACTACTA", 3),
        ]
        extractor = extractor_factory(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.accepted).is_equal_to(0)
        assert_that(stats.no_polyg_anchor).is_equal_to(2)
        assert_that(read_fastq(tmp_path / "out.r1_umi.fastq.gz")).is_empty()

    def test_slippage_yields_identical_umi_and_position(self, extractor_factory, tmp_path) -> None:
        records = [
            make_read("run3", "ACTACTAC", 3),
            make_read("run6", "ACTACTAC", 6),
        ]
        extractor = extractor_factory(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        out = read_fastq(tmp_path / "out.r1_umi.fastq.gz")
        anns = [ReadAnnotation.parse(header[1:]) for header, *_ in out]
        assert_that({ann.get("UMI") for ann in anns}).is_equal_to({"ACTACTAC"})
        assert_that({ann.get("UMI_POS") for ann in anns}).is_equal_to({format_span(20, 28)})

    def test_missing_left_anchor_excluded(self, extractor_factory, tmp_path) -> None:
        good = make_read("good", "ACTACTAC", 4)
        header_no_bc1 = ReadAnnotation(read_id="nobc1").render()
        bad = (header_no_bc1, "A" * 40, "I" * 40)
        extractor = extractor_factory([good, bad])
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.missing_left_anchor).is_equal_to(1)
        assert_that(stats.accepted).is_equal_to(1)
        out = read_fastq(tmp_path / "out.r1_umi.fastq.gz")
        assert_that([ReadAnnotation.parse(h[1:]).read_id for h, *_ in out]).is_equal_to(["good"])

    def test_no_polyg_anchor_excluded(self, extractor_factory, tmp_path) -> None:
        good = make_read("good", "ACTACTAC", 4)
        no_g = (make_read_header("nog"), "A" * 40, "I" * 40)
        extractor = extractor_factory([good, no_g])
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.no_polyg_anchor).is_equal_to(1)
        assert_that(stats.accepted).is_equal_to(1)

    def test_stats_reconcile(self, extractor_factory, tmp_path) -> None:
        records = [
            make_read("a", "ACTACTAC", 4),
            make_read("b", "ACTACTA", 4),
            make_read("c", "ACTACT", 3),
            (ReadAnnotation(read_id="d").render(), "A" * 40, "I" * 40),
            (make_read_header("e"), "A" * 40, "I" * 40),
        ]
        extractor = extractor_factory(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        assert_that(stats.total_reads).is_equal_to(5)
        assert_that(
            stats.accepted + stats.missing_left_anchor + stats.no_polyg_anchor
        ).is_equal_to(stats.total_reads)

    def test_umi_pos_round_trips(self, extractor_factory, tmp_path) -> None:
        records = [make_read("r8", "ACTACTAC", 4), make_read("r9", "ACTACTACT", 5)]
        extractor = extractor_factory(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        for header, seq, _plus, _qual in read_fastq(tmp_path / "out.r1_umi.fastq.gz"):
            ann = ReadAnnotation.parse(header[1:])
            start, end = parse_span(ann.get("UMI_POS"))
            assert_that(seq[start:end]).is_equal_to(ann.get("UMI"))

    def test_output_preserves_input_order_seq_and_qual(self, extractor_factory, tmp_path) -> None:
        r_first = make_read("first", "ACTACTAC", 4)
        r_second = make_read("second", "ACTACTA", 4)
        extractor = extractor_factory([r_first, r_second])
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        out = read_fastq(tmp_path / "out.r1_umi.fastq.gz")
        ids = [ReadAnnotation.parse(h[1:]).read_id for h, *_ in out]
        assert_that(ids).is_equal_to(["first", "second"])
        # seq / qual are carried through untouched.
        assert_that(out[0][1]).is_equal_to(r_first[1])
        assert_that(out[0][3]).is_equal_to(r_first[2])

    @pytest.mark.parametrize("raw", [True, False])
    def test_raw_emits_no_ub_tag(self, extractor_factory, tmp_path, raw: bool) -> None:
        records = [make_read("a", "ACTACTAC", 4)]
        extractor = extractor_factory(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=raw)

        text = (tmp_path / "out.r1_umi.fastq.gz").read_bytes()
        assert_that(gzip.decompress(text).decode()).does_not_contain("UB")

    def test_prefix_derived_from_input_filename(self, extractor_factory, tmp_path) -> None:
        records = [make_read("a", "ACTACTAC", 4)]
        extractor = extractor_factory(records, name="SK462.r1_annotated.fastq.gz")
        extractor.extract_umis(output_dir=str(tmp_path))

        assert_that((tmp_path / "SK462.r1_umi.fastq.gz").exists()).is_true()
        assert_that((tmp_path / "SK462.umi_stats.txt").exists()).is_true()

    def test_stats_file_reports_counts(self, extractor_factory, tmp_path) -> None:
        records = [
            make_read("a", "ACTACTAC", 4),
            make_read("b", "ACTACT", 3),
        ]
        extractor = extractor_factory(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out")

        report = (tmp_path / "out.umi_stats.txt").read_text()
        assert_that(report).contains("Total reads: 2")
        assert_that(report).contains("Accepted: 1")
        assert_that(report).contains("no_polyg_anchor): 1")
        assert_that(report).contains("missing_left_anchor): 0")


class TestUmiExtractionStatsReport:
    """The stats value object and its report rendering."""

    def test_report_includes_run_details_and_distribution(self) -> None:
        stats = UmiExtractionStats(
            total_reads=4,
            accepted=2,
            missing_left_anchor=1,
            no_polyg_anchor=1,
            length_counts={8: 1, 9: 1},
        )
        report = stats.get_report()
        assert_that(report).contains("# Carmack version:")
        assert_that(report).contains("# UMI Extraction Stats")
        assert_that(report).contains("Total reads: 4")
        assert_that(report).contains("Accepted: 2")
        assert_that(report).contains("# UMI Length Distribution")
        assert_that(report).contains("\t8\t1")
        assert_that(report).contains("\t9\t1")

    def test_report_handles_zero_reads_without_error(self) -> None:
        stats = UmiExtractionStats(
            total_reads=0,
            accepted=0,
            missing_left_anchor=0,
            no_polyg_anchor=0,
            length_counts={},
        )
        report = stats.get_report()
        assert_that(report).contains("Total reads: 0")
        assert_that(report).contains("0.00%")


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
                    "--raw",
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_extractor.assert_called_once_with(DUMMY_FASTQ, CHEMISTRY)
        mock_extractor.return_value.extract_umis.assert_called_once_with(
            str(tmp_path), None, raw=True
        )

    def test_cli_listed_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])
        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains("extract-umis")
