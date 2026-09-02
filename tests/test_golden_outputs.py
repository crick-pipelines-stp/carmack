"""
Golden output regression baseline for the barcode and UMI extraction pipelines.

Each test runs a real extraction over a committed FASTQ input and compares every generated
output file against a blessed copy under ``tests/data/golden/expected/``. The point is not
to assert that the outputs are biologically correct, but to make any change in them
visible: a matcher change that shifts a single corrected barcode fails these tests and has
to be explained or re-blessed deliberately.

Tiers
-----
Two always-run cases cover the small inputs (200 reads of ``carmack_custom_seq_1_0`` and 50
reads of ``hydrop``) and cost roughly half a minute in total. Two full-scale cases cover the
2000-read inputs and carry the ``only_run_with_direct_target`` marker, so the repo-root
conftest skips them unless ``-k`` selects them. The full-scale HyDrop case alone takes about
ten minutes: that library barely matches the HyDrop chemistry, so nearly every read falls
through all three matcher tiers.

Determinism
-----------
``BarcodeExtractor.extract_barcodes`` writes ``bc_all``, ``bc_valid`` and ``r1_annotated``
in future-completion order rather than input order, and ``bc_counts.csv`` orders ties by
``Counter`` insertion order. Its output is therefore byte-stable only when there is exactly
one batch. Every fixture here uses ``n_workers=1``, and all four inputs sit below
``MAX_READS_PER_BATCH``, so ``calc_batch_size()`` yields a single batch and completion order
is input order. Raising the worker count would make these tests flaky, not faster.

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
full-scale tier is selected as well, and takes ten to twelve minutes. Always run pytest from
the repo root, or a stale non-editable ``carmack`` in site-packages shadows the repo source
and silently produces different files. Review the resulting diff before committing it.
"""

import functools
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import pytest
from assertpy import assert_that

import carmack.barcode.barcode_extractor as barcode_extractor_module
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

# Only one worker keeps the extraction to a single batch, which is what makes the outputs
# byte-stable. See the module docstring: results are written in completion order, so more
# than one batch reorders bc_all, bc_valid, r1_annotated and the bc_counts tie order.
GOLDEN_WORKERS = 1


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
) -> GoldenRun:
    """
    Run barcode extraction, and optionally UMI extraction, over one golden input.

    The multiprocessing context is forced to ``spawn`` for the duration of the barcode
    extraction. This is a workaround for a production bug, not a test convenience:

    ``BarcodeExtractor.extract_barcodes`` opens its ``ProcessPoolExecutor`` first and the
    three ``gzip`` ``SubprocessStream`` writers second, inside a single ``with`` statement.
    Under the default ``fork`` context, ``executor.submit`` forks workers that inherit each
    ``gzip`` child's stdin write end. The ``with`` unwinds in reverse order, so the gzip
    streams are closed while those workers are still alive holding duplicate write ends,
    ``gzip`` never sees EOF, and ``SubprocessStream.close`` blocks forever in
    ``self.proc.wait()``. It reproduces every time, with as few as 25 reads, on both
    chemistries. Spawned workers are fresh interpreters that inherit no descriptors, so the
    pipes close cleanly.

    Remove this patch once ``extract_barcodes`` shuts the process pool down before closing
    the gzip output streams.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.
        input_name: File name of the input FASTQ under the golden input directory.
        chemistry_name: Registered chemistry name to extract with.
        extract_umis: Whether to chain UMI extraction onto the annotated R1 output.

    Returns:
        The completed run, locating its outputs and goldens.
    """
    input_fastq = GOLDEN_INPUT_DIR / input_name
    prefix = get_prefix(input_fastq)
    output_dir = tmp_path_factory.mktemp(prefix)

    spawning_executor = functools.partial(
        barcode_extractor_module.ProcessPoolExecutor,
        mp_context=multiprocessing.get_context("spawn"),
    )

    # A module-scoped fixture cannot use the function-scoped monkeypatch fixture, so the
    # patch is applied through an explicit context manager instead.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(barcode_extractor_module, "ProcessPoolExecutor", spawning_executor)
        extractor = BarcodeExtractor(str(input_fastq), chemistry_name, n_workers=GOLDEN_WORKERS)
        extractor.extract_barcodes(str(output_dir), prefix)

    if extract_umis:
        annotated_fastq = output_dir / f"{prefix}.r1_annotated.fastq.gz"
        UmiExtractor(str(annotated_fastq), chemistry_name).extract_umis(str(output_dir), prefix)

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
    Extract barcodes and UMIs from the 200-read carmack_custom_seq_1_0 input.

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
    )


@pytest.fixture(scope="module")
def custom_seq_full_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes and UMIs from the 2000-read carmack_custom_seq_1_0 input.

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
    Per-file golden checks for the three UMI extraction outputs.

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

    def test_umi_map_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the raw-to-corrected UMI map matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_text_output_matches_golden(golden_run, "umi_map.tsv")


class TestCustomSeqSmallGoldenOutputs(BarcodeGoldenOutputChecks, UmiGoldenOutputChecks):
    """Golden outputs for 200 reads of carmack_custom_seq_1_0, barcodes and UMIs."""

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
class TestCustomSeqFullScaleGoldenOutputs(BarcodeGoldenOutputChecks, UmiGoldenOutputChecks):
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
