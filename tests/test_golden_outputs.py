"""
Golden output regression baseline for the barcode, UMI and target assignment pipelines.

Each test runs a real extraction over a committed FASTQ input and compares every generated
output file against a blessed copy under ``tests/data/golden/expected/``. The point is not
to assert that the outputs are biologically correct, but to make any change in them
visible: a matcher change that shifts a single corrected barcode fails these tests and has
to be explained or re-blessed deliberately.

Tiers
-----
Two always-run cases cover the small inputs (200 reads of ``carmack_custom_seq_1_0`` and 50
reads of ``hydrop``) and cost roughly ten seconds in total. Two full-scale cases cover the
2000-read inputs and carry the ``only_run_with_direct_target`` marker, so the repo-root
conftest skips them unless ``-k`` selects them. The full-scale HyDrop case alone takes about
two minutes twenty and still dominates the suite, accounting for some 85% of the full-scale
tier: that library barely matches the HyDrop chemistry, so nearly every read falls through
all three matcher tiers.

Each case covers every stage its chemistry supports. The ``carmack_custom_seq_1_0`` cases
run barcode extraction, UMI extraction and target assignment; the HyDrop cases stop after
barcode extraction, because that chemistry declares neither a UMI component nor a target
index.

Determinism
-----------
``BarcodeExtractor.extract_barcodes`` drains its bounded in-flight window in submission
order, so ``bc_all``, ``bc_valid`` and ``r1_annotated`` are written in input order and
``bc_counts.csv`` orders ties by a ``Counter`` populated in that same order. The outputs are
byte-stable at any worker count. Every fixture here runs at ``n_workers=4`` precisely so
that the goldens cover a multi-batch, multi-worker configuration rather than the degenerate
single-batch one a single worker would produce.

``TargetAssigner.assign_targets`` drains that same bounded in-flight window in submission
order, so ``r1_tgidx`` is written in input order too, and ``tgidx_stats.txt`` folds the
per-batch tallies in that order, which leaves the report unaffected by how the reads were
batched at all. It runs at the same ``n_workers=4``, but its batch size comes from the
fixture rather than from the stage default: at the default every golden input would fit in
one batch, and the in-order fold this baseline is meant to cover would never be reached.
``GOLDEN_ASSIGN_BATCH_SIZE`` is chosen to put each of them over several batches instead.

``fast=True`` is deliberately not used: it drops the AlignmentMatcher, which is exactly the
tier this baseline exists to protect.

Comparison rules
----------------
Gzipped text outputs are compared decompressed, and their goldens are stored decompressed
too. Gzip bytes depend on the compressing tool's version and settings, so they are not a
stable contract; the text inside them is. Plain-text outputs are compared verbatim, except
the stats reports, which are normalised through ``strip_report_run_details`` to drop the
Carmack version line and the generation timestamp.

The barcode rank plot is checked only for being a non-empty PNG. Matplotlib's raster output
varies with library version, backend and available fonts, so byte-comparing it would fail
on an unrelated environment change rather than on a pipeline regression.

Regeneration
------------
The default path asserts. To re-bless the goldens after an intended change::

    CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py -q -p no:sugar
    CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py -q -p no:sugar -k golden

The first command re-blesses the always-run tier only. The second adds ``-k`` so the
full-scale tier is selected as well, and takes about two and three quarter minutes for the
whole 34-test tier. Always run pytest from the repo root, or a stale non-editable
``carmack`` in site-packages shadows the repo source and silently produces different files.
Review the resulting diff before committing it.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.assign_targets.target_assigner import TargetAssigner
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.umi.umi_extractor import UmiExtractor
from carmack.utils import get_prefix
from tests.utils import (
    assert_is_png,
    assert_matches_golden,
    read_gzip_text,
    strip_report_run_details,
)

GOLDEN_INPUT_DIR = Path(__file__).parent / "data" / "golden"
GOLDEN_EXPECTED_DIR = GOLDEN_INPUT_DIR / "expected"

CUSTOM_SEQ_CHEMISTRY = "carmack_custom_seq_1_0"
HYDROP_CHEMISTRY = "hydrop"

# Four workers put every golden input over more than one batch: the 200-read input splits
# 4x50, the 50-read input splits 13/13/13/11 and both 2000-read inputs split 4x500. That
# multi-batch shape is the property worth protecting, because a single worker would leave
# every fixture on one batch and so could not detect a reordering regression at all.
GOLDEN_WORKERS = 4

# Batch size target assignment runs at, small enough to split every input it is given. The
# stage consumes the UMI-annotated R1 rather than the raw input, and that file is shorter:
# the 200-read input leaves 190 reads and the 2000-read input leaves 1890. At 50 reads a
# batch the first splits 50/50/50/40 and the second into 38 batches, where the stage default
# of 2500 would leave both on a single batch and cover the in-order fold not at all.
GOLDEN_ASSIGN_BATCH_SIZE = 50


@dataclass(frozen=True)
class GoldenRun:
    """
    Locations of one completed extraction run and its matching golden files.

    Attributes:
        output_dir: Directory the extraction wrote its outputs into.
        prefix: Output file prefix shared by every file the run produced.
    """

    output_dir: Path
    prefix: str

    def produced(self, suffix: str) -> Path:
        """
        Resolve a file the extraction produced.

        Args:
            suffix: Output suffix after the prefix, such as ``"bc_counts.csv"``.

        Returns:
            Path to the produced file.
        """
        return self.output_dir / f"{self.prefix}.{suffix}"

    def golden(self, suffix: str) -> Path:
        """
        Resolve the golden file a produced file is compared against.

        Args:
            suffix: Golden suffix after the prefix, such as ``"bc_counts.csv"``.

        Returns:
            Path to the golden file under the expected-output directory.
        """
        return GOLDEN_EXPECTED_DIR / f"{self.prefix}.{suffix}"


def execute_golden_run(
    tmp_path_factory: pytest.TempPathFactory,
    input_name: str,
    chemistry_name: str,
    extract_umis: bool,
    assign_targets: bool,
) -> GoldenRun:
    """
    Run barcode extraction, and optionally UMI extraction and target assignment, over one
    golden input.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.
        input_name: File name of the input FASTQ under the golden input directory.
        chemistry_name: Registered chemistry name to extract with.
        extract_umis: Whether to chain UMI extraction onto the annotated R1 output.
        assign_targets: Whether to chain target assignment onto the UMI-annotated R1 output.

    Returns:
        The completed run, locating its outputs and goldens.
    """
    input_fastq = GOLDEN_INPUT_DIR / input_name
    prefix = get_prefix(input_fastq)
    output_dir = tmp_path_factory.mktemp(prefix)

    extractor = BarcodeExtractor(str(input_fastq), chemistry_name, n_workers=GOLDEN_WORKERS)
    extractor.extract_barcodes(str(output_dir), prefix)

    if extract_umis:
        annotated_fastq = output_dir / f"{prefix}.r1_annotated.fastq.gz"
        UmiExtractor(str(annotated_fastq), chemistry_name).extract_umis(str(output_dir), prefix)

    if assign_targets:
        umi_fastq = output_dir / f"{prefix}.r1_umi.fastq.gz"
        assigner = TargetAssigner(
            str(umi_fastq),
            chemistry_name,
            n_workers=GOLDEN_WORKERS,
            batch_size=GOLDEN_ASSIGN_BATCH_SIZE,
        )
        assigner.assign_targets(str(output_dir), prefix)

    return GoldenRun(output_dir=output_dir, prefix=prefix)


def assert_gzip_output_matches_golden(
    run: GoldenRun, produced_suffix: str, golden_suffix: str
) -> None:
    """
    Assert that a gzipped output matches its decompressed golden file.

    Args:
        run: The completed extraction run under test.
        produced_suffix: Suffix of the gzipped file the run produced.
        golden_suffix: Suffix of the decompressed golden file to compare against.
    """
    produced = run.produced(produced_suffix)
    assert_that(produced.is_file()).described_as(str(produced)).is_true()
    assert_matches_golden(read_gzip_text(produced), run.golden(golden_suffix))


def assert_text_output_matches_golden(run: GoldenRun, suffix: str) -> None:
    """
    Assert that a plain-text output matches its golden file verbatim.

    Args:
        run: The completed extraction run under test.
        suffix: Suffix shared by the produced file and its golden file.
    """
    produced = run.produced(suffix)
    assert_that(produced.is_file()).described_as(str(produced)).is_true()
    assert_matches_golden(produced.read_text(), run.golden(suffix))


def assert_report_output_matches_golden(run: GoldenRun, suffix: str) -> None:
    """
    Assert that a stats report matches its golden file once run details are stripped.

    Reports carry a Carmack version line and a generation timestamp that change on every
    run, so the produced report is normalised before comparison. Goldens are blessed from
    that same normalised text and so are already stripped; that is asserted here rather
    than assumed, which keeps the golden side normalised too without rewriting the file.

    Args:
        run: The completed extraction run under test.
        suffix: Suffix shared by the produced report and its golden file.
    """
    produced = run.produced(suffix)
    assert_that(produced.is_file()).described_as(str(produced)).is_true()

    golden = run.golden(suffix)

    if golden.is_file():
        golden_text = golden.read_text()
        assert_that(strip_report_run_details(golden_text)).described_as(
            f"golden {golden} must be stored with the volatile run details already stripped"
        ).is_equal_to(golden_text)

    assert_matches_golden(strip_report_run_details(produced.read_text()), golden)


@pytest.fixture(scope="module")
def custom_seq_small_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes, UMIs and target indices from the 200-read carmack_custom_seq_1_0 input.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.

    Returns:
        The completed run, shared by every test in the always-run custom-seq class.
    """
    return execute_golden_run(
        tmp_path_factory,
        input_name="custom_seq_1_0_small_R1.fastq.gz",
        chemistry_name=CUSTOM_SEQ_CHEMISTRY,
        extract_umis=True,
        assign_targets=True,
    )


@pytest.fixture(scope="module")
def hydrop_small_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes from the 50-read HyDrop input.

    HyDrop defines no UMI component, so this run has no UMI stage.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.

    Returns:
        The completed run, shared by every test in the always-run HyDrop class.
    """
    return execute_golden_run(
        tmp_path_factory,
        input_name="hydrop_small_R1.fastq.gz",
        chemistry_name=HYDROP_CHEMISTRY,
        extract_umis=False,
        assign_targets=False,
    )


@pytest.fixture(scope="module")
def custom_seq_full_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes, UMIs and target indices from the 2000-read carmack_custom_seq_1_0 input.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.

    Returns:
        The completed run, shared by every test in the full-scale custom-seq class.
    """
    return execute_golden_run(
        tmp_path_factory,
        input_name="custom_seq_1_0_R1.fastq.gz",
        chemistry_name=CUSTOM_SEQ_CHEMISTRY,
        extract_umis=True,
        assign_targets=True,
    )


@pytest.fixture(scope="module")
def hydrop_full_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes from the 2000-read HyDrop input.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.

    Returns:
        The completed run, shared by every test in the full-scale HyDrop class.
    """
    return execute_golden_run(
        tmp_path_factory,
        input_name="hydrop_R1.fastq.gz",
        chemistry_name=HYDROP_CHEMISTRY,
        extract_umis=False,
        assign_targets=False,
    )


class BarcodeGoldenOutputChecks:
    """
    Per-file golden checks for the six barcode extraction outputs.

    A test class opts in by inheriting this and overriding the `golden_run` fixture with the
    extraction run it covers. This class is not collected itself: it has no Test prefix.
    """

    def test_bc_all_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that every annotated read name written to bc_all matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "bc_all.txt.gz", "bc_all.txt")

    def test_bc_valid_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the successfully matched read names in bc_valid match the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "bc_valid.txt.gz", "bc_valid.txt")

    def test_r1_annotated_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the annotated R1 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(
            golden_run, "r1_annotated.fastq.gz", "r1_annotated.fastq"
        )

    def test_bc_counts_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the ranked full-barcode counts match the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_text_output_matches_golden(golden_run, "bc_counts.csv")

    def test_bc_stats_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the barcode stats report matches the golden file once run details are stripped.

        Args:
            golden_run: The extraction run under test.
        """
        assert_report_output_matches_golden(golden_run, "bc_stats.txt")

    def test_bc_rank_plot_is_png(self, golden_run: GoldenRun) -> None:
        """
        Test that the barcode rank plot is written as a non-empty PNG.

        The image is not byte-compared: matplotlib's raster output varies with library
        version, backend and available fonts, so a byte comparison would fail on an
        unrelated environment change rather than on a pipeline regression.

        Args:
            golden_run: The extraction run under test.
        """
        assert_is_png(golden_run.produced("bc_rank.png"))


class UmiGoldenOutputChecks:
    """
    Per-file golden checks for the two UMI extraction outputs.

    Only chemistries defining a UMI component reach this stage, so HyDrop test classes do
    not inherit it. This class is not collected itself: it has no Test prefix.
    """

    def test_r1_umi_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the UMI-annotated R1 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "r1_umi.fastq.gz", "r1_umi.fastq")

    def test_umi_stats_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the UMI stats report matches the golden file once run details are stripped.

        Args:
            golden_run: The extraction run under test.
        """
        assert_report_output_matches_golden(golden_run, "umi_stats.txt")


class TargetGoldenOutputChecks:
    """
    Per-file golden checks for the two target assignment outputs.

    Only chemistries declaring a target index reach this stage, so HyDrop test classes do
    not inherit it. This class is not collected itself: it has no Test prefix.
    """

    def test_r1_tgidx_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the target-annotated R1 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "r1_tgidx.fastq.gz", "r1_tgidx.fastq")

    def test_tgidx_stats_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the target stats report matches the golden file once run details are stripped.

        Args:
            golden_run: The extraction run under test.
        """
        assert_report_output_matches_golden(golden_run, "tgidx_stats.txt")


class TestCustomSeqSmallGoldenOutputs(
    BarcodeGoldenOutputChecks, UmiGoldenOutputChecks, TargetGoldenOutputChecks
):
    """Golden outputs for 200 reads of carmack_custom_seq_1_0, barcodes, UMIs and targets."""

    @pytest.fixture
    def golden_run(self, custom_seq_small_run: GoldenRun) -> GoldenRun:
        """
        Bind the inherited checks to the small carmack_custom_seq_1_0 run.

        Args:
            custom_seq_small_run: The module-scoped extraction run.

        Returns:
            The run the inherited checks assert against.
        """
        return custom_seq_small_run


class TestHydropSmallGoldenOutputs(BarcodeGoldenOutputChecks):
    """Golden barcode outputs for 50 reads of the HyDrop library."""

    @pytest.fixture
    def golden_run(self, hydrop_small_run: GoldenRun) -> GoldenRun:
        """
        Bind the inherited checks to the small HyDrop run.

        Args:
            hydrop_small_run: The module-scoped extraction run.

        Returns:
            The run the inherited checks assert against.
        """
        return hydrop_small_run


@pytest.mark.only_run_with_direct_target
class TestCustomSeqFullScaleGoldenOutputs(
    BarcodeGoldenOutputChecks, UmiGoldenOutputChecks, TargetGoldenOutputChecks
):
    """Golden outputs for 2000 reads of carmack_custom_seq_1_0, selected with -k only."""

    @pytest.fixture
    def golden_run(self, custom_seq_full_run: GoldenRun) -> GoldenRun:
        """
        Bind the inherited checks to the full-scale carmack_custom_seq_1_0 run.

        Args:
            custom_seq_full_run: The module-scoped extraction run.

        Returns:
            The run the inherited checks assert against.
        """
        return custom_seq_full_run


@pytest.mark.only_run_with_direct_target
class TestHydropFullScaleGoldenOutputs(BarcodeGoldenOutputChecks):
    """Golden barcode outputs for 2000 reads of the HyDrop library, selected with -k only."""

    @pytest.fixture
    def golden_run(self, hydrop_full_run: GoldenRun) -> GoldenRun:
        """
        Bind the inherited checks to the full-scale HyDrop run.

        Args:
            hydrop_full_run: The module-scoped extraction run.

        Returns:
            The run the inherited checks assert against.
        """
        return hydrop_full_run
