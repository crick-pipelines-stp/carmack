"""
Golden output regression baseline for the barcode, UMI and target assignment pipelines.

Each test runs a real extraction over a committed FASTQ input and compares every generated
output file against a blessed copy under ``tests/data/golden/expected/``. The point is not
to assert that the outputs are biologically correct, but to make any change in them
visible: a matcher change that shifts a single corrected barcode fails these tests and has
to be explained or re-blessed deliberately.

Tiers
-----
Three always-run cases cover the small inputs (200 reads of ``carmack_custom_seq_1_0``, 50
reads of ``hydrop``, and those same 200 reads padded out to the
``carmack_custom_seq_1_0_primd`` layout) and cost roughly fifteen seconds in total. Two
full-scale cases cover the 2000-read inputs and carry the ``only_run_with_direct_target``
marker, so the repo-root conftest skips them unless ``-k`` selects them. The full-scale
HyDrop case alone takes about two minutes twenty and still dominates the suite, accounting
for some 85% of the full-scale tier: that library barely matches the HyDrop chemistry, so
nearly every read falls through all three matcher tiers.

Each case covers every stage its chemistry supports. The ``carmack_custom_seq_1_0`` cases --
and the primd case, which shares that chemistry's components beyond its leading PRIMER_D --
run barcode extraction, UMI extraction, target assignment and prepare-reads; the HyDrop
cases stop after barcode extraction, because that chemistry declares neither a UMI
component nor a target index.

No real R2 was ever collected alongside either committed ``carmack_custom_seq_1_0`` R1
input, so the ``carmack_custom_seq_1_0`` cases pair a synthesized R2 golden fixture (see
``tests/data_generators/golden.py`` for its provenance) with the target-annotated R1
target assignment produced, and hand both to ``ReadPreparer`` as they stand. The fixture
goes in whole: barcode and UMI extraction have already dropped reads from that R1, so the
R2 reads with no R1 half left are exactly what a real run carries, and pairing them is
``ReadPreparer``'s job rather than the harness's. HyDrop supports neither UMI extraction
nor target assignment, so it never reaches prepare-reads and has no R2 fixture at all.

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

``ReadPreparer.prepare_reads`` drains its own bounded in-flight window in submission order
too, so the unmatched arm's three files and each matched bucket's R1/R2 are written in the
target-annotated R1's own order, and ``prepare_stats.txt`` folds per-batch tallies in that
same order. It runs at the same ``n_workers=4``, with its own batch size chosen the same
way ``GOLDEN_ASSIGN_BATCH_SIZE`` is: ``GOLDEN_PREPARE_BATCH_SIZE`` puts each fixture over
several batches instead of the stage default's single one.

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

MultiQC payload contract
------------------------
Alongside the per-file comparisons, every run is held to a contract that applies to any
``*_mqc.json`` it emits, whatever stage wrote it. MultiQC reads a custom-content file's
*top-level* ``data`` and never walks nested payloads, so a file bundling several payloads
under keys of a stage's own choosing is discarded whole -- and discarded with a warning
rather than an error, which is how a run goes on looking entirely healthy while the report
it produced quietly loses most of what was measured. Asserting the handful of payloads we
happen to know about would close that handful; asserting the contract over every file a
real run emits closes the class, for the stages carmack has today and for the ones it
grows.

Only the top level of ``data`` is asserted non-empty, never the per-sample level. Target
assignment deliberately never suppresses its target distribution, so a run in which nothing
matched legitimately emits ``data`` of ``{prefix: {}}``: empty per sample, truthy at the
top, and rendered by MultiQC exactly as intended. Asserting per-sample non-emptiness would
fail that run, which is the one whose distribution a reader most needs to see.

The expected file set is asserted per chemistry as well, so a payload that silently stops
being written is caught alongside one that is written malformed.

carmack_custom_seq_1_0_primd
----------------------------
The contract covers all three registered chemistries, and the third has no golden input to
run over. ``carmack_custom_seq_1_0_primd`` is ``carmack_custom_seq_1_0`` with a leading
PRIMER_D, which shifts every component after it, so running it over the committed
custom_seq input as it stands matches nothing at all and suppresses the conditional
payloads the contract most needs to see. Its fixture therefore builds an input at run time
by padding that same committed R1 out to the primd layout, which gives a run of the same
strength as its custom_seq counterpart. The pad length is read off the two read structures
rather than restated from PRIMER_D's length, because read geometry in this repo comes from
the ``ReadStructure`` and never from a per-chemistry constant. The committed R2 is paired
in unmodified: R2 carries no barcode structure, so it has nothing to shift.

That run is not blessed against anything. ``TestPrimdGoldenOutputs`` inherits the payload
contract alone and deliberately none of the golden-comparison mix-ins, because its input is
derived at run time rather than committed and there is nothing a golden could meaningfully
be taken from.

Regeneration
------------
The default path asserts. To re-bless the goldens after an intended change::

    CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py -q -p no:sugar
    CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py -q -p no:sugar -k golden

The first command re-blesses the always-run tier only. The second adds ``-k`` so the
full-scale tier is selected as well, and takes about two and three quarter minutes for the
whole 81-test tier. Always run pytest from the repo root, or a stale non-editable
``carmack`` in site-packages shadows the repo source and silently produces different files.
Review the resulting diff before committing it.

Regeneration touches the golden comparisons alone. The payload contract checks are
assertions about a shape MultiQC requires rather than comparisons against a blessed copy,
so they have nothing to re-bless and go on asserting under that environment variable too. A
failing contract check during a regeneration run is a real defect, not a stale golden.
"""

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from assertpy import assert_that

from carmack.assign_targets.target_assigner import TargetAssigner
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.mqc_report import CARMACK_PARENT_ID
from carmack.prepare_reads.read_preparer import ReadPreparer
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
PRIMD_CHEMISTRY = "carmack_custom_seq_1_0_primd"

# The shipped custom_seq target whitelist's one entry, read off the chemistry itself
# rather than restated as a literal, so a whitelist change is caught here too.
CUSTOM_SEQ_TGIDX_VALUE = ChemistryFactory.get_chemistry(CUSTOM_SEQ_CHEMISTRY).tgidx_whitelist()[0]

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

# Batch size prepare-reads runs at, small enough to split every input it is given. The
# stage consumes the target-annotated R1, which target assignment never filters, so it
# carries the same read count as the UMI-annotated file above: 190 reads for the small
# input, 1890 for the full one. At 50 reads a batch the small input splits 50/50/50/40 and
# the full one splits into 38 batches, where the stage default of 2500 would leave both on
# a single batch and cover neither the in-order fold nor the R2 sidecar's alignment across
# batch boundaries at all.
GOLDEN_PREPARE_BATCH_SIZE = 50

# Every MultiQC custom-content file a stage writes carries this suffix, and the contract
# checks walk whatever the glob finds rather than a list of names, so a stage added later is
# covered the day it first writes a payload.
MQC_PAYLOAD_GLOB = "*_mqc.json"

# Every payload id is namespaced under carmack's MultiQC parent id, and each payload's file
# is named from its id with that namespace removed. Both are read off the production
# constant rather than restated here, so the contract cannot drift from the namespace it is
# asserting.
MQC_ID_PREFIX = f"{CARMACK_PARENT_ID}_"

# The payload files each stage emits, one file per payload. Listed per stage so that a
# chemistry's expected set is assembled from the stages it actually runs. Five of these are
# conditional -- a stage suppresses an edit-distance or anchor-run distribution it never
# observed, and the barcode rank curve when no read yielded a full barcode, because an empty
# chart claiming a measurement that was never taken is worse than an absent section -- but
# every input this module runs exercises all five, so they belong in the expected set rather
# than being treated as optional here.
EXTRACTION_MQC_STEMS = (
    "extraction_general_stats",
    "extraction_breakdown",
    "extraction_barcode_rank",
    "extraction_edit_distance",
)
UMI_MQC_STEMS = (
    "umi_general_stats",
    "umi_breakdown",
    "umi_anchor_run",
)
TGIDX_MQC_STEMS = (
    "tgidx_general_stats",
    "tgidx_breakdown",
    "tgidx_target_distribution",
    "tgidx_edit_distance",
    "tgidx_anchor_run",
)
PREPARE_MQC_STEMS = (
    "prepare_general_stats",
    "prepare_target_distribution",
)

# A chemistry running every stage emits all four stages' payloads; one that stops after
# barcode extraction emits only the first stage's.
FULL_PIPELINE_MQC_STEMS = (
    *EXTRACTION_MQC_STEMS,
    *UMI_MQC_STEMS,
    *TGIDX_MQC_STEMS,
    *PREPARE_MQC_STEMS,
)
BARCODE_ONLY_MQC_STEMS = EXTRACTION_MQC_STEMS

# The first component carmack_custom_seq_1_0 and its PRIMER_D variant share, and so the
# anchor the primd pad length is measured against. It is the first fixed-length component of
# both layouts, so its start always resolves to a concrete position.
PRIMD_SHARED_COMPONENT = "BC3"

# Bases and quality characters the primd pad is built from. PRIMER_D's sequence is unknown
# and its component deliberately carries none, so nothing in the pipeline ever matches
# against the pad; it only has to occupy the right number of bases at a quality no stage
# would discard, which is the same quality character the committed input uses for its good
# bases.
PRIMD_PAD_BASE = "A"
PRIMD_PAD_QUALITY = "I"

# FASTQ record layout, used to pad the sequence and quality lines while leaving the header
# and separator lines alone.
FASTQ_RECORD_LINES = 4
FASTQ_SEQUENCE_LINE = 1
FASTQ_QUALITY_LINE = 3


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
    input_name: str | Path,
    chemistry_name: str,
    extract_umis: bool,
    assign_targets: bool,
    r2_input_name: str | None = None,
) -> GoldenRun:
    """
    Run barcode extraction, and optionally UMI extraction, target assignment and
    prepare-reads, over one golden input.

    Args:
        tmp_path_factory: Session-scoped factory supplying the run's output directory.
        input_name: File name of the input FASTQ under the golden input directory, or an
            absolute path to an input built at run time. Only the primd case needs the
            latter, because it has no committed fixture of its own and derives one instead.
        chemistry_name: Registered chemistry name to extract with.
        extract_umis: Whether to chain UMI extraction onto the annotated R1 output.
        assign_targets: Whether to chain target assignment onto the UMI-annotated R1 output.
        r2_input_name: File name of a committed R2 golden fixture under the golden input
            directory, or ``None``. When given -- and only once both ``extract_umis`` and
            ``assign_targets`` are true -- prepare-reads is chained onto the target-annotated
            R1 output, paired with this R2 fixture exactly as committed: whole, unfiltered
            and still carrying every read the earlier stages dropped from R1, which is the
            shape a real pipeline hands ``ReadPreparer`` too.

    Returns:
        The completed run, locating its outputs and goldens.
    """
    input_fastq = Path(input_name)
    if not input_fastq.is_absolute():
        input_fastq = GOLDEN_INPUT_DIR / input_fastq
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

        if r2_input_name is not None:
            tgidx_fastq = output_dir / f"{prefix}.r1_tgidx.fastq.gz"
            full_r2_fastq = GOLDEN_INPUT_DIR / r2_input_name

            preparer = ReadPreparer(
                str(tgidx_fastq),
                str(full_r2_fastq),
                chemistry_name,
                n_workers=GOLDEN_WORKERS,
                batch_size=GOLDEN_PREPARE_BATCH_SIZE,
            )
            preparer.prepare_reads(str(output_dir), prefix)

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


def load_mqc_payloads(run: GoldenRun) -> dict[Path, Any]:
    """
    Read every MultiQC payload file a run emitted, keyed by the path it was read from.

    The glob is asserted non-empty here rather than in each caller. A run that emitted no
    payload at all would otherwise satisfy every check that walks these files by walking
    nothing, so the contract would pass most confidently on the run that has the most to
    answer for.

    Args:
        run: The completed extraction run under test.

    Returns:
        Mapping of payload file path to its parsed JSON content, in file-name order.

    Raises:
        AssertionError: If the run emitted no payload file at all.
        Failed: If a payload file does not parse as JSON.
    """
    paths = sorted(run.output_dir.glob(MQC_PAYLOAD_GLOB))
    assert_that(paths).described_as(f"MultiQC payload files under {run.output_dir}").is_not_empty()

    payloads: dict[Path, Any] = {}
    for path in paths:
        try:
            payloads[path] = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            pytest.fail(f"{path} must parse as JSON, but does not: {error}")
    return payloads


def expected_mqc_file_names(prefix: str, stems: Iterable[str]) -> set[str]:
    """
    Build the payload file names a run with this prefix is expected to emit.

    Args:
        prefix: Output file prefix shared by every file the run produced.
        stems: Payload stems, one per payload the run's stages emit.

    Returns:
        The expected file names, as a set for direct comparison against the glob.
    """
    return {f"{prefix}.{stem}_mqc.json" for stem in stems}


def primd_pad_length() -> int:
    """
    Derive how many bases a primd read carries ahead of the carmack_custom_seq_1_0 layout.

    Measured as the offset between where the first component the two chemistries share sits
    in each of their read structures, rather than restated from PRIMER_D's length. Read
    geometry in this repo is derived from the ``ReadStructure`` and never from a
    per-chemistry constant, so a change to that component's length moves this fixture with
    it instead of leaving it silently building the wrong shape.

    Returns:
        The number of bases to prepend to a carmack_custom_seq_1_0 read to shape it like a
        primd one.
    """
    base_structure = ChemistryFactory.get_chemistry(CUSTOM_SEQ_CHEMISTRY).read_structure
    primd_structure = ChemistryFactory.get_chemistry(PRIMD_CHEMISTRY).read_structure

    base_start = base_structure.get_component_by_name(PRIMD_SHARED_COMPONENT).start
    primd_start = primd_structure.get_component_by_name(PRIMD_SHARED_COMPONENT).start

    return primd_start - base_start


def write_primd_input(source_fastq: Path, destination_fastq: Path, pad_length: int) -> Path:
    """
    Write a primd-shaped R1 by padding a carmack_custom_seq_1_0 R1 out to the primd layout.

    Each record's sequence and quality lines gain ``pad_length`` leading characters and its
    header and separator lines are copied through untouched, which shifts every component by
    exactly the offset PRIMER_D introduces and leaves the read otherwise as committed.

    Args:
        source_fastq: Gzipped carmack_custom_seq_1_0 R1 to pad.
        destination_fastq: Gzipped file to write the padded reads to.
        pad_length: Number of bases to prepend to every read.

    Returns:
        The path written, for direct use as a run's input.
    """
    with (
        gzip.open(source_fastq, "rt") as source,
        gzip.open(destination_fastq, "wt") as destination,
    ):
        for index, line in enumerate(source):
            position = index % FASTQ_RECORD_LINES
            if position == FASTQ_SEQUENCE_LINE:
                destination.write(f"{PRIMD_PAD_BASE * pad_length}{line}")
            elif position == FASTQ_QUALITY_LINE:
                destination.write(f"{PRIMD_PAD_QUALITY * pad_length}{line}")
            else:
                destination.write(line)

    return destination_fastq


@pytest.fixture(scope="module")
def custom_seq_small_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes, UMIs and target indices, then prepare reads, from the 200-read
    carmack_custom_seq_1_0 input.

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
        r2_input_name="custom_seq_1_0_small_R2.fastq.gz",
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
def custom_seq_primd_small_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes, UMIs and target indices, then prepare reads, from a primd-shaped input
    built at run time.

    There is no committed primd fixture, and the committed custom_seq input cannot stand in
    for one as it is: PRIMER_D shifts every component after it, so primd matches nothing in
    that input, and the conditional payloads the contract most needs to see are all
    suppressed. Padding that R1 out to the primd layout instead gives a run of the same
    strength as its custom_seq counterpart, reaching every stage and emitting every payload.
    The committed R2 is paired in unmodified, because R2 carries no barcode structure and so
    has nothing to shift.

    Args:
        tmp_path_factory: Session-scoped factory supplying both the padded input's directory
            and the run's output directory.

    Returns:
        The completed run, shared by every test in the primd class.
    """
    padded_input = write_primd_input(
        GOLDEN_INPUT_DIR / "custom_seq_1_0_small_R1.fastq.gz",
        tmp_path_factory.mktemp("primd_input") / "custom_seq_1_0_primd_small_R1.fastq.gz",
        primd_pad_length(),
    )

    return execute_golden_run(
        tmp_path_factory,
        input_name=padded_input,
        chemistry_name=PRIMD_CHEMISTRY,
        extract_umis=True,
        assign_targets=True,
        r2_input_name="custom_seq_1_0_small_R2.fastq.gz",
    )


@pytest.fixture(scope="module")
def custom_seq_full_run(tmp_path_factory: pytest.TempPathFactory) -> GoldenRun:
    """
    Extract barcodes, UMIs and target indices, then prepare reads, from the 2000-read
    carmack_custom_seq_1_0 input.

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
        r2_input_name="custom_seq_1_0_R2.fastq.gz",
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

    def test_barcode_rank_payload_and_plot_are_both_written(self, golden_run: GoldenRun) -> None:
        """
        Test that one run writes both the barcode rank MultiQC payload and the rank PNG.

        The two are separate outputs for separate readers and neither replaces the other:
        the payload feeds the interactive barcode rank curve in the MultiQC report, while
        the PNG is the standalone image this stage has always produced and the fallback for
        a standalone CLI run with no MultiQC to read the payload. Asserted on the same run
        so that dropping either one is caught, rather than being masked by the other still
        being there.

        Args:
            golden_run: The extraction run under test.
        """
        (payload_name,) = expected_mqc_file_names(golden_run.prefix, ["extraction_barcode_rank"])
        payload = golden_run.output_dir / payload_name
        plot = golden_run.produced("bc_rank.png")
        assert_that(payload.exists()).described_as(
            f"MultiQC barcode rank payload {payload}"
        ).is_true()
        assert_that(plot.exists()).described_as(f"static barcode rank plot {plot}").is_true()


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


class PrepareReadsGoldenOutputChecks:
    """
    Per-file golden checks for the six prepare-reads outputs.

    The shipped custom_seq whitelist carries exactly one target (``CUSTOM_SEQ_TGIDX_VALUE``),
    so this class checks that one bucket's R1/R2 pair by name rather than looping over the
    whitelist. Only chemistries with a committed R2 golden fixture reach this stage, so
    HyDrop test classes do not inherit it. This class is not collected itself: it has no
    Test prefix.
    """

    def test_unmatched_r1_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the unmatched (scRNA) arm's trimmed R1 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "none.r1.fastq.gz", "none.r1.fastq")

    def test_unmatched_r2_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the unmatched (scRNA) arm's passthrough R2 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(golden_run, "none.r2.fastq.gz", "none.r2.fastq")

    def test_unmatched_barcodes_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the unmatched arm's synthesized barcodes FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(
            golden_run, "none.barcodes.fastq.gz", "none.barcodes.fastq"
        )

    def test_matched_bucket_r1_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the one real target bucket's trimmed R1 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(
            golden_run,
            f"{CUSTOM_SEQ_TGIDX_VALUE}.r1.fastq.gz",
            f"{CUSTOM_SEQ_TGIDX_VALUE}.r1.fastq",
        )

    def test_matched_bucket_r2_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the one real target bucket's passthrough R2 FASTQ matches the golden file.

        Args:
            golden_run: The extraction run under test.
        """
        assert_gzip_output_matches_golden(
            golden_run,
            f"{CUSTOM_SEQ_TGIDX_VALUE}.r2.fastq.gz",
            f"{CUSTOM_SEQ_TGIDX_VALUE}.r2.fastq",
        )

    def test_prepare_stats_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the prepare-reads stats report matches the golden file once run details
        are stripped.

        Args:
            golden_run: The extraction run under test.
        """
        assert_report_output_matches_golden(golden_run, "prepare_stats.txt")

    def test_detected_targets_matches_golden(self, golden_run: GoldenRun) -> None:
        """
        Test that the detected-targets list matches the golden file verbatim.

        Compared verbatim rather than normalised: this file carries no version or
        timestamp line to strip, which is the point of it.

        Args:
            golden_run: The extraction run under test.
        """
        assert_text_output_matches_golden(golden_run, "detected_targets.txt")


class MqcPayloadContractChecks:
    """
    Contract checks holding for every MultiQC payload file a run emits, whatever wrote it.

    MultiQC reads a custom-content file's top-level ``data`` and never walks nested
    payloads, so a file bundling several payloads under keys of a stage's own choosing is
    discarded whole -- and discarded with a warning rather than an error, which is how a run
    can look entirely healthy while the report it produced is missing most of what was
    measured. A handful of per-stage assertions would close the handful of instances we
    happen to know about; asserting this contract over every file a real run emits closes
    the class of defect, for the stages carmack has today and for the ones it grows.

    Only the top level of ``data`` is asserted non-empty, never the per-sample level. Target
    assignment deliberately never suppresses its target distribution, so a run in which
    nothing matched legitimately emits ``data`` of ``{prefix: {}}``: empty per sample,
    truthy at the top, and rendered by MultiQC exactly as intended. Asserting per-sample
    non-emptiness would fail that run, which is the one whose distribution a reader most
    needs to see.

    Parametrising over the emitted files is not possible here. Parametrisation is resolved
    at collection time, and these files exist only once the module-scoped run fixture has
    executed, so each check walks the glob inside its own body instead. ``load_mqc_payloads``
    asserts that glob non-empty, so none of these checks can pass by walking nothing.

    A test class opts in by inheriting this and overriding both the `golden_run` fixture with
    the run it covers and the `expected_mqc_stems` fixture with the payloads that run should
    emit. This class is not collected itself: it has no Test prefix.
    """

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Provide the payload stems the run under test is expected to emit.

        This is a guard implementation: a test class inheriting these checks is responsible
        for overriding it with the stems its chemistry's stages produce.

        Returns:
            Payload stems, one per expected payload file.
        """
        pytest.fail(
            "Test classes inheriting MqcPayloadContractChecks must override the "
            "'expected_mqc_stems' fixture"
        )

    def test_mqc_payloads_are_json_objects(self, golden_run: GoldenRun) -> None:
        """
        Test that the run emitted at least one payload and that each parses to an object.

        MultiQC keys everything it reads off the top level of the file, so a payload that is
        not a JSON object gives it nothing to key on at all.

        Args:
            golden_run: The extraction run under test.
        """
        for path, payload in load_mqc_payloads(golden_run).items():
            assert_that(payload).described_as(
                f"{path} must hold a JSON object at its top level"
            ).is_instance_of(dict)

    def test_mqc_payloads_declare_a_namespaced_id(self, golden_run: GoldenRun) -> None:
        """
        Test that every payload names itself with a non-empty, carmack-namespaced id.

        The id is what MultiQC anchors a section on, and namespacing it keeps carmack's
        sections from colliding with those of any other tool in the same report.

        Args:
            golden_run: The extraction run under test.
        """
        for path, payload in load_mqc_payloads(golden_run).items():
            payload_id = payload.get("id")
            assert_that(payload_id).described_as(
                f"{path} must carry a top-level id that is a non-empty string"
            ).is_instance_of(str).is_not_empty()
            assert_that(payload_id).described_as(
                f"{path} must namespace its id under {MQC_ID_PREFIX!r}"
            ).starts_with(MQC_ID_PREFIX)

    def test_mqc_payloads_declare_a_plot_type(self, golden_run: GoldenRun) -> None:
        """
        Test that every payload says how MultiQC should draw it.

        The plot type is what MultiQC renders the section as; without it there is nothing to
        draw and the file contributes nothing to the report.

        Args:
            golden_run: The extraction run under test.
        """
        for path, payload in load_mqc_payloads(golden_run).items():
            assert_that(payload.get("plot_type")).described_as(
                f"{path} must carry a non-empty top-level plot_type"
            ).is_not_none().is_not_empty()

    def test_mqc_payloads_carry_top_level_data(self, golden_run: GoldenRun) -> None:
        """
        Test that every payload carries a non-empty ``data`` section at its top level.

        This is the assertion the whole contract exists for. MultiQC takes a custom-content
        file's top-level ``data`` and never walks nested payloads, so a file bundling
        several payloads under a stage's own keys has no top-level ``data`` to take and is
        discarded whole, with a warning rather than an error.

        The check stops at the top level deliberately. Target assignment never suppresses
        its target distribution, so a run that matched no read at all legitimately emits
        ``{prefix: {}}`` here: empty per sample, truthy at the top, and rendered. Asserting
        per-sample non-emptiness would fail that entirely correct run.

        Args:
            golden_run: The extraction run under test.
        """
        for path, payload in load_mqc_payloads(golden_run).items():
            assert_that(payload.get("data")).described_as(
                f"{path} must carry a non-empty top-level data section, because MultiQC "
                "reads the top level only and discards a file whose payloads are nested"
            ).is_not_none().is_not_empty()

    def test_mqc_payload_file_names_are_derived_from_their_ids(
        self, golden_run: GoldenRun
    ) -> None:
        """
        Test that every payload's file is named from the payload's own id.

        The id is what MultiQC anchors the section on, so naming the file from it means the
        file and the section it defines cannot drift apart, and one payload can never
        overwrite another's file.

        Args:
            golden_run: The extraction run under test.
        """
        for path, payload in load_mqc_payloads(golden_run).items():
            payload_id = payload.get("id")
            assert_that(payload_id).described_as(
                f"{path} must carry an id for its file name to be derived from"
            ).is_instance_of(str)
            stem = str(payload_id).removeprefix(MQC_ID_PREFIX)
            assert_that(path.name).described_as(
                f"{path} must be named from its own id {payload_id!r}"
            ).is_equal_to(f"{golden_run.prefix}.{stem}_mqc.json")

    def test_mqc_payload_ids_are_unique(self, golden_run: GoldenRun) -> None:
        """
        Test that no two payload files in a run claim the same id.

        MultiQC anchors a section on the id, so two files sharing one describe the same
        section twice and only one of them survives into the report.

        Args:
            golden_run: The extraction run under test.
        """
        claimed: dict[Any, Path] = {}
        for path, payload in load_mqc_payloads(golden_run).items():
            payload_id = payload.get("id")
            assert_that(claimed).described_as(
                f"{path} must claim an id no other payload in the run claims, but "
                f"{payload_id!r} is already claimed by {claimed.get(payload_id)}"
            ).does_not_contain_key(payload_id)
            claimed[payload_id] = path

    def test_mqc_payload_set_matches_the_stages_that_ran(
        self, golden_run: GoldenRun, expected_mqc_stems: tuple[str, ...]
    ) -> None:
        """
        Test that the run emitted exactly the payload files its stages should have.

        Asserted as an exact set rather than as a subset, so a payload that silently stops
        being written is caught too: a chart vanishing from the report is as damaging as a
        malformed one and considerably harder to notice.

        Args:
            golden_run: The extraction run under test.
            expected_mqc_stems: Payload stems this run's stages are expected to emit.
        """
        produced = {path.name for path in load_mqc_payloads(golden_run)}
        assert_that(produced).described_as(
            f"MultiQC payload files under {golden_run.output_dir}"
        ).is_equal_to(expected_mqc_file_names(golden_run.prefix, expected_mqc_stems))


class TestCustomSeqSmallGoldenOutputs(
    BarcodeGoldenOutputChecks,
    UmiGoldenOutputChecks,
    TargetGoldenOutputChecks,
    PrepareReadsGoldenOutputChecks,
    MqcPayloadContractChecks,
):
    """Golden outputs for 200 reads of carmack_custom_seq_1_0: barcodes, UMIs, targets and prepared reads."""

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

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Declare the payloads a run through every stage should emit.

        Returns:
            The four stages' payload stems.
        """
        return FULL_PIPELINE_MQC_STEMS


class TestHydropSmallGoldenOutputs(BarcodeGoldenOutputChecks, MqcPayloadContractChecks):
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

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Declare the payloads a HyDrop run should emit.

        HyDrop declares neither a UMI component nor a target index, so the run stops after
        barcode extraction and only that stage's payloads exist to emit.

        Returns:
            The barcode extraction payload stems.
        """
        return BARCODE_ONLY_MQC_STEMS


@pytest.mark.only_run_with_direct_target
class TestCustomSeqFullScaleGoldenOutputs(
    BarcodeGoldenOutputChecks,
    UmiGoldenOutputChecks,
    TargetGoldenOutputChecks,
    PrepareReadsGoldenOutputChecks,
    MqcPayloadContractChecks,
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

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Declare the payloads a run through every stage should emit.

        Returns:
            The four stages' payload stems.
        """
        return FULL_PIPELINE_MQC_STEMS


@pytest.mark.only_run_with_direct_target
class TestHydropFullScaleGoldenOutputs(BarcodeGoldenOutputChecks, MqcPayloadContractChecks):
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

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Declare the payloads a HyDrop run should emit.

        HyDrop declares neither a UMI component nor a target index, so the run stops after
        barcode extraction and only that stage's payloads exist to emit.

        Returns:
            The barcode extraction payload stems.
        """
        return BARCODE_ONLY_MQC_STEMS


class TestPrimdGoldenOutputs(MqcPayloadContractChecks):
    """
    MultiQC payload contract for 200 reads shaped like carmack_custom_seq_1_0_primd.

    This class inherits the payload contract alone and deliberately none of the
    golden-comparison mix-ins. Its input is padded out of the custom_seq fixture at run time
    rather than committed, so there is nothing here a golden could meaningfully be blessed
    from; what it does cover is the third registered chemistry's payloads, which is the
    whole reason the run exists.
    """

    @pytest.fixture
    def golden_run(self, custom_seq_primd_small_run: GoldenRun) -> GoldenRun:
        """
        Bind the inherited checks to the padded primd run.

        Args:
            custom_seq_primd_small_run: The module-scoped extraction run.

        Returns:
            The run the inherited checks assert against.
        """
        return custom_seq_primd_small_run

    @pytest.fixture
    def expected_mqc_stems(self) -> tuple[str, ...]:
        """
        Declare the payloads a run through every stage should emit.

        primd shares carmack_custom_seq_1_0's components beyond the leading PRIMER_D, so it
        runs the same four stages and should emit the same payloads.

        Returns:
            The four stages' payload stems.
        """
        return FULL_PIPELINE_MQC_STEMS
