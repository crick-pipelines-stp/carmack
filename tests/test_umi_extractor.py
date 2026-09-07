"""Tests for the extract-umis module.

The extractor reads an annotated R1 FASTQ, extracts the raw UMI lying between
its left anchor (BC1, read from the header ``BC1_POS`` tag) and the downstream
poly-G run, annotates the read with ``UMI`` / ``UMI_POS`` and reports the length
distribution. Most of these tests build synthetic annotated reads and exercise
the greedy poly-G anchoring, the two-sided length window, the reconciling stats
and the CLI wiring; the closing class drives real barcode extraction twice over
to pin the map's row order down across repeat runs.
"""

import gzip
from collections import defaultdict
from pathlib import Path
from unittest import mock

import pytest
from assertpy import assert_that
from click.testing import CliRunner

import carmack.__main__
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_corrector import CorrectedUmi, UmiCorrector, UmiRecord
from carmack.umi.umi_extractor import UmiExtractor
from carmack.umi.umi_reporting import CorrectionStats, UmiExtractionStats
from carmack.umi.umi_shards import UmiShardStore, shard_index
from tests.test_barcode_extractor import run_extraction_in_process_group

CHEMISTRY = "carmack_custom_seq_1_0"
DUMMY_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"

# Layout used by all synthetic reads: BC1 occupies [BC1_START, UMI_START) and the
# UMI begins where BC1 ends. Only the BC1_POS *end* is consumed by the extractor.
BC1_START = 10
UMI_START = 20
TAIL = "TTTTTT"

# Default barcode-component sequences carried by synthetic annotated reads. They
# mirror the BC1/BC2/BC3 tags written by barcode extraction so the correction
# stage can rebuild the full cell barcode.
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
    """Render a header carrying the BC1_POS anchor and the BC1/BC2/BC3 barcodes.

    The extractor consumes only the BC1_POS end to locate the UMI; the barcode
    tags let the correction stage reconstruct the full cell barcode.
    """
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
    the poly-G run begins exactly at ``umi_start + len(umi_seq)``.
    """
    seq = "A" * umi_start + umi_seq + "G" * run_len + TAIL
    return make_read_header(read_id, umi_start, bc1, bc2, bc3), seq, "I" * len(seq)


@pytest.fixture
def build_extractor(tmp_path):
    """Return a factory that writes records to a FASTQ and builds an extractor."""

    def build(records: list[tuple[str, str, str]], name: str = "SK462.r1_annotated.fastq.gz"):
        fastq_path = tmp_path / name
        write_fastq(fastq_path, records)
        return UmiExtractor(str(fastq_path), CHEMISTRY)

    return build


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

    def test_report_includes_anchor_run_distribution(self) -> None:
        stats = UmiExtractionStats(
            total_reads=3,
            accepted=3,
            missing_left_anchor=0,
            no_polyg_anchor=0,
            length_counts={8: 3},
            homopolymer_base="G",
            homopolymer_run_counts={3: 1, 4: 2},
        )
        report = stats.get_report()
        assert_that(report).contains("# Anchor G-run Length Distribution")
        assert_that(report).contains("\t3\t1")
        assert_that(report).contains("\t4\t2")

    def test_correction_report_includes_new_metrics(self) -> None:
        stats = UmiExtractionStats(
            total_reads=4,
            accepted=4,
            missing_left_anchor=0,
            no_polyg_anchor=0,
            length_counts={8: 4},
            correction=CorrectionStats(
                assigned_reads=4,
                corrections_applied=1,
                num_cell_barcodes=1,
                dropped_raw_n=0,
                dropped_off_length=0,
                distinct_corrected_umis=1,
                umi_collapses=3,
            ),
        )
        report = stats.get_report()
        assert_that(report).contains("# UMI Correction Stats")
        assert_that(report).contains("Cell barcodes (groups): 1")
        assert_that(report).contains("Reads assigned UB: 4")
        assert_that(report).contains("Corrections applied (UB != raw): 1")
        assert_that(report).contains("Mean reads per UMI: 4.00")


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
            str(tmp_path), None, raw=True, temp_dir=None, shard_count=256
        )

    def test_cli_listed_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])
        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains("extract-umis")


class TestExtractUmisCorrection:
    """The correction stage: the umi_map.tsv contract and correction stats."""

    def read_map(self, path) -> list[list[str]]:
        """Read the tab-separated, header-less umi_map.tsv into split rows."""
        return [line.split("\t") for line in path.read_text().splitlines()]

    def test_writes_umi_map_with_corrected_reads(self, build_extractor, tmp_path) -> None:
        records = [
            make_read("p1", "ACTACTAC", 4),
            make_read("p2", "ACTACTAC", 4),
            make_read("p3", "ACTACTAC", 4),
            make_read("v1", "ACTACTTC", 4),
        ]
        extractor = build_extractor(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=False)

        rows = self.read_map(tmp_path / "out.umi_map.tsv")
        assert_that(rows).is_length(4)
        for row in rows:
            assert_that(row).is_length(4)  # read_id, barcode, UR, UB

        full_barcode = extractor.chemistry.construct_full_barcode(
            {"BC1": BC1_SEQ, "BC2": BC2_SEQ, "BC3": BC3_SEQ}
        )
        by_id = {row[0]: row for row in rows}
        assert_that(by_id["v1"][1]).is_equal_to(full_barcode)
        assert_that(by_id["v1"][2]).is_equal_to("ACTACTTC")  # UR = faithful raw
        assert_that(by_id["v1"][3]).is_equal_to("ACTACTAC")  # UB = highest-count rep
        assert_that({row[3] for row in rows}).is_equal_to({"ACTACTAC"})
        assert_that(stats.correction.assigned_reads).is_equal_to(4)
        # Only v1 (ACTACTTC) was reassigned to the ACTACTAC representative.
        assert_that(stats.correction.corrections_applied).is_equal_to(1)
        # The anchor run-length distribution is recorded (all reads use a 4-G run).
        assert_that(stats.homopolymer_base).is_equal_to("G")
        assert_that(stats.homopolymer_run_counts).is_equal_to({4: 4})

    def test_raw_true_writes_no_map_and_no_ub(self, build_extractor, tmp_path) -> None:
        records = [make_read("a", "ACTACTAC", 4)]
        extractor = build_extractor(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=True)

        assert_that((tmp_path / "out.umi_map.tsv").exists()).is_false()
        assert_that(stats.correction).is_none()
        fastq = gzip.decompress((tmp_path / "out.r1_umi.fastq.gz").read_bytes()).decode()
        assert_that(fastq).does_not_contain("UB")

    def test_grouping_isolates_barcodes_end_to_end(self, build_extractor, tmp_path) -> None:
        records = [
            make_read("g1", "ACTACTAC", 4, bc1="AAAAAAAAAA"),
            make_read("g2", "ACTACTAC", 4, bc1="TTTTTTTTTT"),
        ]
        extractor = build_extractor(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=False)

        rows = {row[0]: row for row in self.read_map(tmp_path / "out.umi_map.tsv")}
        assert_that(rows["g1"][1]).is_not_equal_to(rows["g2"][1])
        assert_that(stats.correction.distinct_corrected_umis).is_equal_to(2)

    def test_raw_n_read_excluded_from_map(self, build_extractor, tmp_path) -> None:
        records = [
            make_read("clean", "ACTACTAC", 4),
            make_read("ncontam", "ACTNCTAC", 4),
        ]
        extractor = build_extractor(records)
        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=False)

        ids = [row[0] for row in self.read_map(tmp_path / "out.umi_map.tsv")]
        assert_that(ids).contains("clean")
        assert_that(ids).does_not_contain("ncontam")
        assert_that(stats.correction.dropped_raw_n).is_equal_to(1)

    def test_correction_stats_in_report(self, build_extractor, tmp_path) -> None:
        records = [make_read("a", "ACTACTAC", 4), make_read("b", "ACTACTAC", 4)]
        extractor = build_extractor(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=False)

        report = (tmp_path / "out.umi_stats.txt").read_text()
        assert_that(report).contains("# UMI Correction Stats")
        assert_that(report).contains("Cell barcodes (groups): 1")
        assert_that(report).contains("Reads assigned UB: 2")
        # Two identical UMIs: both are the representative, so nothing is reassigned.
        assert_that(report).contains("Corrections applied (UB != raw): 0")

    def test_raw_report_omits_correction_section(self, build_extractor, tmp_path) -> None:
        records = [make_read("a", "ACTACTAC", 4)]
        extractor = build_extractor(records)
        extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=True)

        report = (tmp_path / "out.umi_stats.txt").read_text()
        assert_that(report).does_not_contain("# UMI Correction Stats")


class TestExtractUmisHeaderValidation:
    """The first read's header is validated against the supplied chemistry."""

    def make_missing_bc3(self, read_id: str) -> tuple[str, str, str]:
        """An annotated read carrying BC1/BC2 (+BC1_POS) but no BC3 tag."""
        ann = ReadAnnotation(read_id=read_id)
        ann.set(position_key("BC1"), format_span(BC1_START, UMI_START))
        ann.set("BC1", BC1_SEQ)
        ann.set("BC2", BC2_SEQ)
        seq = "A" * UMI_START + "ACTACTAC" + "GGGG" + TAIL
        return ann.render(), seq, "I" * len(seq)

    @pytest.mark.parametrize("raw", [False, True])
    def test_missing_barcode_tag_raises_clear_error(
        self, build_extractor, tmp_path, raw: bool
    ) -> None:
        records = [self.make_missing_bc3("r1"), self.make_missing_bc3("r2")]
        extractor = build_extractor(records)

        with pytest.raises(ValueError, match="BC3"):
            extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=raw)
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
            extractor.extract_umis(output_dir=str(tmp_path), prefix="out", raw=True)


class TestCorrectionStatsCombine:
    """Summing per-shard correction stats back into one run-level total."""

    # Twelve cell barcodes whose crc32 digests populate all four shards, so the
    # sharded run genuinely spans several shards rather than degenerating to one.
    BARCODES = [f"{'ACGT' * 2}{index:02d}{'C' * 10}" for index in range(12)]

    def sharded_records(self) -> list[UmiRecord]:
        """Build a record set spanning many barcodes, collapses and both drop reasons.

        Every barcode carries a three-read parent UMI, a one-read neighbour that
        clustering collapses into it and a distant UMI that stays separate; some
        barcodes additionally carry a short UMI that normalisation pads, a raw
        UMI holding the padding sentinel and an off-length UMI.

        Returns:
            The assembled records in extraction order.
        """
        records: list[UmiRecord] = []
        for index, barcode in enumerate(self.BARCODES):
            umis = ["AAAAAAAA"] * 3 + ["AAAAAAAT"] + ["CCCCCCCC"] * 2
            if index % 2 == 0:
                umis.append("GGGGGGG")
            if index % 3 == 0:
                umis.append("AAAANAAA")
            if index % 4 == 0:
                umis.append("TTT")
            for position, raw_umi in enumerate(umis):
                records.append(
                    UmiRecord(
                        read_id=f"r{index:02d}_{position:02d}", barcode=barcode, raw_umi=raw_umi
                    )
                )
        return records

    def test_combine_sums_every_field(self) -> None:
        parts = [
            CorrectionStats(1, 2, 3, 4, 5, 6, 7),
            CorrectionStats(10, 20, 30, 40, 50, 60, 70),
            CorrectionStats(100, 0, 5, 0, 1, 2, 0),
        ]

        combined = CorrectionStats.combine(parts)

        assert_that(combined).is_equal_to(CorrectionStats(111, 22, 38, 44, 56, 68, 77))

    def test_combine_of_nothing_is_all_zero_stats(self) -> None:
        combined = CorrectionStats.combine([])

        assert_that(combined).is_equal_to(CorrectionStats(0, 0, 0, 0, 0, 0, 0))

    def test_combine_of_a_single_part_returns_that_part(self) -> None:
        part = CorrectionStats(
            assigned_reads=9,
            corrections_applied=2,
            num_cell_barcodes=3,
            dropped_raw_n=1,
            dropped_off_length=4,
            distinct_corrected_umis=5,
            umi_collapses=6,
        )

        assert_that(CorrectionStats.combine([part])).is_equal_to(part)

    def test_sharded_stats_combine_to_the_monolithic_stats(self) -> None:
        records = self.sharded_records()
        monolithic_map, monolithic = UmiCorrector(x=8, tol=1).correct(records)

        shards: dict[int, list[UmiRecord]] = defaultdict(list)
        for record in records:
            shards[shard_index(record.barcode, 4)].append(record)
        parts = []
        sharded_map: dict[str, CorrectedUmi] = {}
        for index in sorted(shards):
            mapping, stats = UmiCorrector(x=8, tol=1).correct(shards[index])
            sharded_map.update(mapping)
            parts.append(stats)
        combined = CorrectionStats.combine(parts)

        # Several shards must be populated or the equality proves nothing.
        assert_that(len(parts)).is_greater_than(1)
        assert_that(combined).is_equal_to(monolithic)
        assert_that(combined.num_cell_barcodes).is_equal_to(monolithic.num_cell_barcodes)
        assert_that(combined.distinct_corrected_umis).is_equal_to(
            monolithic.distinct_corrected_umis
        )
        assert_that(combined.umi_collapses).is_equal_to(monolithic.umi_collapses)
        assert_that(sharded_map).is_equal_to(monolithic_map)


# Synthetic cell-barcode panel used by the sharding tests. Seven barcodes each
# carrying four reads put twenty-eight reads through the run, so the ordinals run
# well past ten and the barcodes land in several distinct shards.
SHARDED_BARCODE_COUNT = 7
SHARDED_UMIS = ["ACTACTAC", "ACTACTAC", "ACTACTAC", "ACTACTTC"]


def make_cell_barcode(index: int) -> str:
    """Return a distinct ten-base BC1 sequence for one synthetic cell barcode.

    Args:
        index: Position of the barcode in the synthetic panel (0-15).

    Returns:
        A ten-base sequence differing from that of every other index.
    """
    bases = "ACGT"
    return f"AACCGGTT{bases[index // 4]}{bases[index % 4]}"


def make_many_barcode_records() -> list[tuple[str, str, str]]:
    """Build annotated reads spanning many cell barcodes and many ordinals.

    Every barcode carries a three-read parent UMI plus one single-read neighbour
    that directional clustering collapses into it, so the run exercises grouping,
    collapsing and more than ten reads at once.

    Returns:
        The ``(header, seq, qual)`` records in input order.
    """
    records: list[tuple[str, str, str]] = []
    for index in range(SHARDED_BARCODE_COUNT):
        for umi_seq in SHARDED_UMIS:
            read_id = f"read{len(records) + 1:02d}"
            records.append(make_read(read_id, umi_seq, 4, bc1=make_cell_barcode(index)))
    return records


def input_read_ids(records: list[tuple[str, str, str]]) -> list[str]:
    """Return the read ids of annotated records in input order.

    Args:
        records: The ``(header, seq, qual)`` records handed to the extractor.

    Returns:
        The parsed read ids, in the order the reads appear in the FASTQ.
    """
    return [ReadAnnotation.parse(header).read_id for header, _seq, _qual in records]


class TestExtractUmisSharding:
    """Sharded correction must be invisible in the extractor's output."""

    def read_map(self, path) -> list[list[str]]:
        """Read the tab-separated, header-less umi_map.tsv into split rows."""
        return [line.split("\t") for line in path.read_text().splitlines()]

    def full_barcodes(self, extractor: UmiExtractor) -> set[str]:
        """Return the full cell barcodes the synthetic panel resolves to.

        Args:
            extractor: The extractor whose chemistry assembles the barcode.

        Returns:
            The distinct full barcodes carried by the panel's reads.
        """
        return {
            extractor.chemistry.construct_full_barcode(
                {"BC1": make_cell_barcode(index), "BC2": BC2_SEQ, "BC3": BC3_SEQ}
            )
            for index in range(SHARDED_BARCODE_COUNT)
        }

    def test_umi_map_is_identical_across_shard_counts(self, build_extractor, tmp_path) -> None:
        records = make_many_barcode_records()
        extractor = build_extractor(records)

        outputs: dict[int, bytes] = {}
        for shard_count in (1, 3, 7, 256):
            output_dir = tmp_path / f"shards{shard_count}"
            output_dir.mkdir()
            extractor.extract_umis(
                output_dir=str(output_dir), prefix="out", shard_count=shard_count
            )
            outputs[shard_count] = (output_dir / "out.umi_map.tsv").read_bytes()

        # The panel must straddle several shards or the equality proves nothing.
        barcodes = self.full_barcodes(extractor)
        for shard_count in (3, 7, 256):
            spread = {shard_index(barcode, shard_count) for barcode in barcodes}
            assert_that(len(spread)).is_greater_than(1)

        # shard_count=1 reproduces the old single-pass behaviour, so it is the reference.
        reference = outputs[1]
        assert_that(reference).is_not_empty()
        assert_that(outputs[3]).is_equal_to(reference)
        assert_that(outputs[7]).is_equal_to(reference)
        assert_that(outputs[256]).is_equal_to(reference)

    @pytest.mark.parametrize("shard_count", [3, 7, 256])
    def test_correction_stats_match_the_single_shard_run(
        self, build_extractor, tmp_path, shard_count: int
    ) -> None:
        records = make_many_barcode_records()
        extractor = build_extractor(records)
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        sharded_dir = tmp_path / "sharded"
        sharded_dir.mkdir()

        reference = extractor.extract_umis(output_dir=str(single_dir), prefix="out", shard_count=1)
        sharded = extractor.extract_umis(
            output_dir=str(sharded_dir), prefix="out", shard_count=shard_count
        )

        assert_that(sharded.correction).is_equal_to(reference.correction)
        # The three counts of distinct things are the ones sharding could double count.
        assert_that(sharded.correction.num_cell_barcodes).is_equal_to(
            reference.correction.num_cell_barcodes
        )
        assert_that(sharded.correction.distinct_corrected_umis).is_equal_to(
            reference.correction.distinct_corrected_umis
        )
        assert_that(sharded.correction.umi_collapses).is_equal_to(
            reference.correction.umi_collapses
        )
        # The fixture must actually group and collapse, or the equality is vacuous.
        assert_that(reference.correction.num_cell_barcodes).is_equal_to(SHARDED_BARCODE_COUNT)
        assert_that(reference.correction.umi_collapses).is_greater_than(0)

    @pytest.mark.parametrize("shard_count", [1, 3, 7, 256])
    def test_read_order_is_preserved_past_the_tenth_read(
        self, build_extractor, tmp_path, shard_count: int
    ) -> None:
        records = make_many_barcode_records()
        extractor = build_extractor(records)

        extractor.extract_umis(output_dir=str(tmp_path), prefix="out", shard_count=shard_count)

        read_ids = [row[0] for row in self.read_map(tmp_path / "out.umi_map.tsv")]
        # Ordinals compared as text would sort 1, 10, 11, 2, ..., so the run must
        # be long enough for that scramble to show up in the read order.
        assert_that(len(read_ids)).is_greater_than(10)
        assert_that(read_ids).is_equal_to(input_read_ids(records))

    def test_dropped_read_omitted_while_neighbours_keep_their_order(
        self, build_extractor, tmp_path
    ) -> None:
        records = make_many_barcode_records()
        records.insert(14, make_read("sentinel", "ACTNCTAC", 4, bc1=make_cell_barcode(2)))
        extractor = build_extractor(records)

        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", shard_count=7)

        read_ids = [row[0] for row in self.read_map(tmp_path / "out.umi_map.tsv")]
        expected = [read_id for read_id in input_read_ids(records) if read_id != "sentinel"]
        assert_that(read_ids).does_not_contain("sentinel")
        assert_that(read_ids).is_equal_to(expected)
        assert_that(stats.correction.dropped_raw_n).is_equal_to(1)

    def test_every_read_rejected_writes_an_empty_map(self, build_extractor, tmp_path) -> None:
        records = [(make_read_header(f"nog{index}"), "A" * 40, "I" * 40) for index in range(3)]
        extractor = build_extractor(records)

        stats = extractor.extract_umis(output_dir=str(tmp_path), prefix="out", shard_count=4)

        umi_map_path = tmp_path / "out.umi_map.tsv"
        assert_that(umi_map_path.exists()).is_true()
        assert_that(umi_map_path.read_bytes()).is_empty()
        assert_that(stats.correction).is_equal_to(CorrectionStats(0, 0, 0, 0, 0, 0, 0))

    @pytest.mark.parametrize("shard_count", [0, -1])
    def test_shard_count_below_one_raises(
        self, build_extractor, tmp_path, shard_count: int
    ) -> None:
        extractor = build_extractor([make_read("a", "ACTACTAC", 4)])

        with pytest.raises(ValueError, match="shard_count"):
            extractor.extract_umis(output_dir=str(tmp_path), prefix="out", shard_count=shard_count)


def make_spilled_records(extractor: UmiExtractor, replicates: int) -> list[UmiRecord]:
    """Build the records extraction would spill for the synthetic barcode panel.

    Each record is shaped the way ``extract_umis`` shapes it: a full cell barcode
    assembled by the chemistry from the panel's BC1/BC2/BC3 tags, and one record
    per accepted read. Every UMI is in-window and sentinel-free, so correction
    assigns all of them and the merged map carries one row per record.

    Args:
        extractor: The extractor whose chemistry assembles the full barcode.
        replicates: Number of times the panel's reads are repeated, which sets
            how much text each shard's write handle has to hold.

    Returns:
        The records in extraction order, so their ordinals ascend from zero.
    """
    barcodes = [
        extractor.chemistry.construct_full_barcode(
            {"BC1": make_cell_barcode(index), "BC2": BC2_SEQ, "BC3": BC3_SEQ}
        )
        for index in range(SHARDED_BARCODE_COUNT)
    ]
    records: list[UmiRecord] = []
    for _ in range(replicates):
        for barcode in barcodes:
            for umi_seq in SHARDED_UMIS:
                records.append(
                    UmiRecord(read_id=f"read{len(records):05d}", barcode=barcode, raw_umi=umi_seq)
                )
    return records


class TestCorrectShards:
    """Correction driven straight against a shard store the caller populated."""

    # Enough replicates of the panel that every shard holds several times the
    # text one write handle buffers, so a shard read that missed the buffered
    # tail would drop rows from the middle of the map and not only its end.
    REPLICATES = 40

    def test_correct_shards_maps_every_record_without_a_caller_side_close(
        self, build_extractor, tmp_path
    ) -> None:
        extractor = build_extractor([make_read("a", "ACTACTAC", 4)])
        records = make_spilled_records(extractor, self.REPLICATES)
        umi_map_path = tmp_path / "out.umi_map.tsv"

        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            for ordinal, record in enumerate(records):
                store.write(ordinal, record)
            stats = extractor.correct_shards(store, umi_map_path)

        # The panel has to straddle several shards, or one readable shard could
        # carry the whole map and a lost shard would go unnoticed.
        spread = {shard_index(record.barcode, 4) for record in records}
        assert_that(len(spread)).is_greater_than(1)
        read_ids = [line.split("\t")[0] for line in umi_map_path.read_text().splitlines()]
        assert_that(read_ids).is_length(len(records))
        assert_that(read_ids).is_equal_to([record.read_id for record in records])
        assert_that(stats.assigned_reads).is_equal_to(len(records))


class TestExtractUmisTempDir:
    """Where the spill tree is created, and that it never outlives the run."""

    @pytest.fixture
    def spill_dir(self, tmp_path):
        """Return a pre-created, empty directory to hold the spill tree."""
        path = tmp_path / "spill"
        path.mkdir()
        return path

    def test_raw_run_does_no_shard_work(self, build_extractor, tmp_path, spill_dir) -> None:
        extractor = build_extractor([make_read("a", "ACTACTAC", 4), make_read("b", "ACTACTAC", 4)])

        extractor.extract_umis(
            output_dir=str(tmp_path), prefix="out", raw=True, temp_dir=str(spill_dir)
        )

        assert_that(list(spill_dir.iterdir())).is_empty()
        assert_that((tmp_path / "out.umi_map.tsv").exists()).is_false()

    def test_successful_run_leaves_no_spill_tree(
        self, build_extractor, tmp_path, spill_dir
    ) -> None:
        extractor = build_extractor(make_many_barcode_records())

        extractor.extract_umis(
            output_dir=str(tmp_path), prefix="out", temp_dir=str(spill_dir), shard_count=4
        )

        assert_that(list(spill_dir.iterdir())).is_empty()
        assert_that((tmp_path / "out.umi_map.tsv").exists()).is_true()

    def test_failed_correction_still_removes_the_spill_tree(
        self, build_extractor, tmp_path, spill_dir
    ) -> None:
        extractor = build_extractor(make_many_barcode_records())

        with mock.patch(
            "carmack.umi.umi_extractor.UmiCorrector.correct",
            side_effect=RuntimeError("correction exploded"),
        ):
            with pytest.raises(RuntimeError, match="correction exploded"):
                extractor.extract_umis(
                    output_dir=str(tmp_path), prefix="out", temp_dir=str(spill_dir), shard_count=4
                )

        assert_that(list(spill_dir.iterdir())).is_empty()

    def test_spill_tree_is_created_under_the_supplied_temp_dir(
        self, build_extractor, tmp_path, spill_dir
    ) -> None:
        extractor = build_extractor(make_many_barcode_records())
        original_correct = UmiCorrector.correct
        observed: list[list[str]] = []

        def spying_correct(corrector, records):
            """Record the supplied temp dir's contents at the moment correction runs."""
            observed.append([entry.name for entry in sorted(Path(spill_dir).iterdir())])
            return original_correct(corrector, records)

        with mock.patch.object(UmiCorrector, "correct", spying_correct):
            extractor.extract_umis(
                output_dir=str(tmp_path), prefix="out", temp_dir=str(spill_dir), shard_count=4
            )

        assert_that(observed).is_not_empty()
        assert_that(observed[0]).is_not_empty()
        for name in observed[0]:
            assert_that(name).starts_with("carmack-umi-")
        # The tree the run created is gone once the run is over.
        assert_that(list(spill_dir.iterdir())).is_empty()


class TestExtractUmisCliShardOptions:
    """CLI wiring for the temp-dir and shard-count options."""

    def test_cli_passes_temp_dir_and_shard_count(self, tmp_path) -> None:
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
                    "--temp-dir",
                    str(tmp_path),
                    "--shard-count",
                    "8",
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_extractor.return_value.extract_umis.assert_called_once_with(
            str(tmp_path), None, raw=False, temp_dir=str(tmp_path), shard_count=8
        )

    def test_cli_help_lists_the_sharding_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(carmack.__main__.carmack_cli, ["extract-umis", "--help"])

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains("--temp-dir")
        assert_that(result.output).contains("--shard-count")

    def test_cli_rejects_a_shard_count_below_one(self, tmp_path) -> None:
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
                    "--shard-count",
                    "0",
                ],
            )

        assert_that(result.exit_code).is_not_equal_to(0)
        assert_that(result.output).contains("not in the range")
        mock_extractor.return_value.extract_umis.assert_not_called()


UMI_MAP_DETERMINISM_INPUT = (
    Path(__file__).parent / "data" / "golden" / "custom_seq_1_0_small_R1.fastq.gz"
)
UMI_MAP_DETERMINISM_PREFIX = "determinism"
UMI_MAP_DETERMINISM_WORKERS = 16
UMI_MAP_DETERMINISM_RUNS = 2
# Barcode extraction sizes its batches as min(max(10, ceil(200 / 16)), 2500) = 13, so the
# 200-read input becomes 16 batches spread over 16 workers. That the batches genuinely run
# concurrently is what gives a completion-order fold the chance to reorder the annotated
# FASTQ, and so the map, in the first place.
UMI_MAP_DETERMINISM_BATCH_SIZE = 13
# Four shards keep the spilled records spread over several shard files while costing four
# open file descriptors rather than the production default's 256, which is worth avoiding
# for a 200-read run sharing a session with the rest of the suite.
UMI_MAP_DETERMINISM_SHARD_COUNT = 4


class TestUmiMapRepeatRunDeterminism:
    """The umi_map.tsv is byte-stable over repeat runs at a high worker count.

    Every accepted read is stamped with its ordinal in annotated-FASTQ order and
    the shard merge sorts the map on that ordinal, so the map's row order is
    exactly the annotated FASTQ's read order. While barcode extraction folded its
    worker results as they completed, that read order followed however the pool
    happened to schedule its batches rather than the input, so two runs of the
    same input at the same worker count could write the same rows in a different
    order. Folding in submission order is what removes that variation, and this
    is what keeps it removed.

    The two runs are compared against each other rather than against a stored
    golden, so the assertion holds whatever the goldens happen to contain.

    ``fast=True`` drops the alignment matcher tier, which is the tier that makes
    barcode extraction slow. Row order is what is asserted here, not matcher
    sensitivity, so dropping that tier costs no coverage.

    Barcode extraction is driven through a deadline-guarded process group: it is
    the stage that forks a worker pool while gzip writer subprocesses are open, so
    a teardown regression there hangs the session rather than failing it. UMI
    extraction opens no pool, so it runs in-process.
    """

    @pytest.fixture(scope="class")
    def umi_maps(self, tmp_path_factory: pytest.TempPathFactory) -> list[bytes]:
        """Run barcode then UMI extraction twice over, returning each run's map bytes.

        Each run gets its own output directory and its own spill directory under
        the session's temporary tree, so neither run can observe the other's
        files and the spill never reaches a possibly RAM-backed system default.

        Args:
            tmp_path_factory: Factory supplying the class-scoped working tree.

        Returns:
            The raw bytes of each run's ``umi_map.tsv``, in run order.
        """
        work_dir = tmp_path_factory.mktemp("umi_map_determinism")
        maps: list[bytes] = []

        for run in range(UMI_MAP_DETERMINISM_RUNS):
            run_dir = work_dir / f"run{run}"
            run_dir.mkdir()
            spill_dir = run_dir / "spill"
            spill_dir.mkdir()

            run_extraction_in_process_group(
                str(UMI_MAP_DETERMINISM_INPUT),
                run_dir,
                UMI_MAP_DETERMINISM_PREFIX,
                chemistry_name=CHEMISTRY,
                n_workers=UMI_MAP_DETERMINISM_WORKERS,
                fast=True,
            )

            annotated = run_dir / f"{UMI_MAP_DETERMINISM_PREFIX}.r1_annotated.fastq.gz"
            extractor = UmiExtractor(str(annotated), CHEMISTRY)
            extractor.extract_umis(
                output_dir=str(run_dir),
                prefix=UMI_MAP_DETERMINISM_PREFIX,
                temp_dir=str(spill_dir),
                shard_count=UMI_MAP_DETERMINISM_SHARD_COUNT,
            )
            maps.append((run_dir / f"{UMI_MAP_DETERMINISM_PREFIX}.umi_map.tsv").read_bytes())

        return maps

    def test_umi_map_is_byte_identical_across_repeat_runs(self, umi_maps: list[bytes]) -> None:
        first, second = umi_maps

        # Two empty maps compare equal for the wrong reason, and rows drawn from a single
        # batch could not have been reordered at all, so both are ruled out up front.
        assert_that(first).is_not_empty()
        assert_that(len(first.decode().splitlines())).is_greater_than(
            UMI_MAP_DETERMINISM_BATCH_SIZE
        )

        assert_that(second).described_as(
            f"umi_map.tsv over two independent {UMI_MAP_DETERMINISM_WORKERS}-worker runs"
        ).is_equal_to(first)
