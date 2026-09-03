"""
Tests for the repeatable pipeline benchmark harness.

The subject is what happens around a child process, so every test but one points
``--carmack-command`` at a throwaway script that records the argv it was handed.

The exception carries ``only_run_with_direct_target`` and drives the real CLI: a stand-in
accepts whatever it is given, so that test alone notices the harness's flag construction
drifting from the CLI's signature. Timing is asserted as lower bounds only, never upper.
"""

import hashlib
import io
import json
import os
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from assertpy import assert_that

import benchmark_pipeline

CHEMISTRY = "carmack_custom_seq_1_0"
PREFIX = "bench"
LABEL = "harness-unit-run"
WORKERS = 4
FAILING_EXIT_CODE = 3
ARGV_LOG_NAME = "argv.jsonl"
STAGE_NAMES = ("extract-barcodes", "extract-umis", "assign-targets")
STAGE_TABLE = {stage.name: stage for stage in benchmark_pipeline.STAGES}
# The FASTQ each stage emits under the run prefix. Chaining is defined entirely by these:
# stage two's input is stage one's file inside the output directory.
STAGE_OUTPUT_NAMES = {
    "extract-barcodes": f"{PREFIX}.r1_annotated.fastq.gz",
    "extract-umis": f"{PREFIX}.r1_umi.fastq.gz",
    "assign-targets": f"{PREFIX}.r1_tgidx.fastq.gz",
}
# Not a version the real CLI could report, so a recorded value proves the harness asked the
# command it actually ran rather than reading an import.
FAKE_VERSION = "9.9.9-fake"

FASTQ_BYTES = "".join(f"@read{n}\nACGTACGTACGT\n+\n############\n" for n in range(8)).encode()
FASTQ_SHA256 = hashlib.sha256(FASTQ_BYTES).hexdigest()
# Short enough to keep the suite fast, long enough to dominate scheduler noise.
SPIN_SECONDS = 0.05
# No interpreter has a resident set below this, so a harness that recorded raw Linux
# ru_maxrss KiB as though they were bytes would fall under it.
MIN_RSS_BYTES = 1_000_000

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CLI_FASTQ = REPO_ROOT / "tests" / "data" / "golden" / "custom_seq_1_0_small_R1.fastq.gz"

# A console script on PATH can predate a stage, so the checkout is driven through the
# running interpreter instead, with the repo put ahead of site-packages for the child.
REAL_CLI_COMMAND = shlex.join([sys.executable, "-m", "carmack"])

# The stand-in for the carmack CLI: it records the argv of every stage it is asked to run,
# answers --version, spins for a set interval, and writes the files that stage would.
FAKE_CARMACK_SOURCE = """\
import gzip, json, sys, time
from pathlib import Path
config = json.loads(Path(sys.argv[0]).with_name("config.json").read_text())
args = sys.argv[1:]
# A leading option rather than a stage name is a metadata query, and is not logged, so the
# recorded argv holds stage invocations only.
if not args or args[0].startswith("-"):
    if "--version" in args:
        print(config["version_line"])
    sys.exit(0)
stage = args[0]
with Path(config["argv_log"]).open("a", encoding="utf-8") as handle:
    print(json.dumps(args), file=handle)
deadline = time.monotonic() + config["spin_seconds"]
while time.monotonic() < deadline:
    pass
# A failing stage writes nothing, which is what leaves the next stage without an input.
exit_code = config["exit_codes"].get(stage, 0)
if exit_code == 0:
    output_dir = Path(args[args.index("-o") + 1])
    prefix = args[args.index("-p") + 1]
    with gzip.open(output_dir / config["output_names"][stage], "wb") as gzip_handle:
        gzip_handle.write(config["payload"].encode())
    (output_dir / f"{prefix}.{stage}.stats.txt").write_text(config["stats_text"])
sys.exit(exit_code)
"""


@dataclass(frozen=True)
class StageCall:
    """One invocation of the carmack command, as the child itself received it."""

    stage: str
    options: dict[str, str]
    input_path: Path


@dataclass(frozen=True)
class BenchmarkRun:
    """Everything one in-process call to the harness entry point produced."""

    exit_code: int
    console: str
    output_dir: Path
    working_dir: Path
    input_fastq: Path
    calls: tuple[StageCall, ...]

    @property
    def record(self) -> dict[str, Any]:
        """The run record the harness wrote."""
        return json.loads((self.output_dir / "benchmark.json").read_text())

    @property
    def tsv_rows(self) -> list[list[str]]:
        """The table the harness wrote, header row first, split on tabs."""
        table = (self.output_dir / "benchmark.tsv").read_text()
        return [row.split("\t") for row in table.splitlines()]

    @property
    def stages_run(self) -> list[str]:
        """The stage names the harness actually launched, in order."""
        return [call.stage for call in self.calls]


def parse_stage_call(argv: list[str]) -> StageCall:
    """Split one recorded child argv into its stage name, its options and its input."""
    # Flag/value pairs followed by the input, the shape the command-construction test pins.
    flags = [token for token in argv[1:-1] if token.startswith("-")]
    options = {flag: argv[argv.index(flag) + 1] for flag in flags}

    return StageCall(stage=argv[0], options=options, input_path=Path(argv[-1]).resolve())


def build_fake_carmack(
    directory: Path, *, spin_seconds: float = 0.0, exit_codes: dict[str, int] | None = None
) -> str:
    """Write a carmack stand-in into a directory and return the command that runs it."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_carmack.py"
    script.write_text(FAKE_CARMACK_SOURCE)
    config = {
        "argv_log": str(directory / ARGV_LOG_NAME),
        "version_line": f"carmack, version {FAKE_VERSION}",
        "spin_seconds": spin_seconds,
        "exit_codes": exit_codes or {},
        "payload": "@read\nACGTACGTACGT\n+\n############\n" * 16,
        "stats_text": "reads\t8\nassigned\t7\n",
        "output_names": STAGE_OUTPUT_NAMES,
    }
    (directory / "config.json").write_text(json.dumps(config))

    return shlex.join([sys.executable, str(script)])


def harness_argv(**options: str) -> list[str]:
    """Turn keyword options into harness arguments, underscores becoming dashes."""
    pairs = ((f"--{name.replace('_', '-')}", value) for name, value in options.items())
    return [token for pair in pairs for token in pair]


def invoke_harness(argv: list[str]) -> tuple[int, str]:
    """Call the harness in-process, returning its exit code and both console streams."""
    console = io.StringIO()

    # A refusal is a refusal whether the entry point returns a code or raises SystemExit.
    with redirect_stdout(console), redirect_stderr(console):
        try:
            code: Any = benchmark_pipeline.main(argv)
        except SystemExit as exit_signal:
            code = exit_signal.code

    return (code if isinstance(code, int) else int(code is not None), console.getvalue())


def drive_harness(
    root: Path,
    *,
    stages: str | None = None,
    spin_seconds: float = 0.0,
    exit_codes: dict[str, int] | None = None,
    working_dir: Path | None = None,
) -> BenchmarkRun:
    """Run the harness against a fresh stand-in command inside a scratch directory."""
    command = build_fake_carmack(root / "cli", spin_seconds=spin_seconds, exit_codes=exit_codes)
    input_fastq = root / "reads.fastq"
    input_fastq.write_bytes(FASTQ_BYTES)
    output_dir = root / "out"
    # Unless told otherwise the run happens in an empty directory, leaving a stray relative
    # write nowhere to land unnoticed.
    ran_in = working_dir or (root / "cwd")
    ran_in.mkdir(parents=True, exist_ok=True)

    argv = harness_argv(
        fastq=str(input_fastq), chemistry=CHEMISTRY, output_dir=str(output_dir),
        workers=str(WORKERS), prefix=PREFIX, label=LABEL, carmack_command=command,
    )  # fmt: skip
    if stages is not None:
        argv += ["--stages", stages]

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ran_in)
        exit_code, console = invoke_harness(argv)

    log = root / "cli" / ARGV_LOG_NAME
    logged = log.read_text().splitlines() if log.is_file() else []

    return BenchmarkRun(
        exit_code=exit_code, console=console, output_dir=output_dir.resolve(),
        working_dir=ran_in, input_fastq=input_fastq.resolve(),
        calls=tuple(parse_stage_call(json.loads(line)) for line in logged if line.strip()),
    )  # fmt: skip


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkRun:
    """Every stage, spinning a known interval, run in the repo so git metadata is real."""
    root = tmp_path_factory.mktemp("full")
    return drive_harness(root, spin_seconds=SPIN_SECONDS, working_dir=REPO_ROOT)


@pytest.fixture(scope="module")
def subset_run(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkRun:
    """A selection starting partway along the chain, run in an empty working directory."""
    return drive_harness(tmp_path_factory.mktemp("subset"), stages="extract-umis,assign-targets")


@pytest.fixture(scope="module")
def failed_run(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkRun:
    """The whole chain, with its middle stage exiting non-zero."""
    root = tmp_path_factory.mktemp("failed")
    return drive_harness(root, exit_codes={"extract-umis": FAILING_EXIT_CODE})


class TestStageSelection:
    """Tests for turning a ``--stages`` value into the stages to run."""

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("all", list(STAGE_NAMES)),
            ("extract-barcodes", ["extract-barcodes"]),
            ("extract-umis,assign-targets", ["extract-umis", "assign-targets"]),
            (" extract-umis , assign-targets ", ["extract-umis", "assign-targets"]),
        ],
        ids=["all", "single", "mid-chain-start", "padded"],
    )
    def test_select_stages_accepts_a_contiguous_run(self, spec: str, expected: list[str]) -> None:
        """Test that a contiguous run is selected wherever along the chain it starts."""
        selected = benchmark_pipeline.select_stages(spec)

        assert_that([stage.name for stage in selected]).is_equal_to(expected)

    @pytest.mark.parametrize(
        "spec",
        ["extract-barcodes,assign-targets", "assign-targets,extract-umis", "polish-reads", ""],
        ids=["gap", "out-of-order", "unknown", "nothing-named"],
    )
    def test_select_stages_refuses_a_selection_that_is_not_contiguous(self, spec: str) -> None:
        """Test that a gap is refused rather than guessed at.

        The stage after a gap would otherwise be handed a FASTQ that never went through the
        stage that was skipped.
        """
        assert_that(benchmark_pipeline.select_stages).raises(ValueError).when_called_with(spec)


class TestStageCommandConstruction:
    """Tests for the argv the harness builds for one stage."""

    @pytest.mark.parametrize(
        "stage_name, worker_flags",
        [("extract-barcodes", ["-n", str(WORKERS)]), ("extract-umis", []), ("assign-targets", [])],
    )
    def test_build_stage_command_builds_the_argv_each_stage_accepts(
        self, tmp_path: Path, stage_name: str, worker_flags: list[str]
    ) -> None:
        """Test the exact argv per stage, the input last as the sole positional."""
        reads, out = tmp_path / "reads.fastq", tmp_path / "out"

        command = benchmark_pipeline.build_stage_command(
            STAGE_TABLE[stage_name], ["python", "-m", "carmack"], reads, CHEMISTRY, out, PREFIX,
            WORKERS,
        )  # fmt: skip

        expected = ["python", "-m", "carmack", stage_name, "-c", CHEMISTRY, "-o", str(out)]
        assert_that(command).is_equal_to([*expected, "-p", PREFIX, *worker_flags, str(reads)])


class TestFullRun:
    """Tests for a complete run of every stage against the stand-in command."""

    def test_each_stage_child_gets_the_last_stage_output_and_its_own_flags(
        self, full_run: BenchmarkRun
    ) -> None:
        """Test the chain and the worker count from the argv the children received.

        A stage other than ``extract-barcodes`` seeing ``-n`` would abort rather than merely
        be untidy, and only an argv assertion catches it being passed one stage too far.
        """
        first, second, third = (call.input_path for call in full_run.calls)
        workers = {call.stage: call.options.get("-n") for call in full_run.calls}

        assert_that(full_run.exit_code).described_as(full_run.console).is_equal_to(0)
        assert_that(full_run.stages_run).is_equal_to(list(STAGE_NAMES))
        assert_that(full_run.record["ok"]).is_true()
        assert_that(first).is_equal_to(full_run.input_fastq)
        assert_that(second).is_equal_to(full_run.output_dir / f"{PREFIX}.r1_annotated.fastq.gz")
        assert_that(third).is_equal_to(full_run.output_dir / f"{PREFIX}.r1_umi.fastq.gz")
        assert_that(workers).is_equal_to(
            {"extract-barcodes": str(WORKERS), "extract-umis": None, "assign-targets": None}
        )

    def test_record_carries_what_makes_two_runs_comparable(self, full_run: BenchmarkRun) -> None:
        """Test that the run's identity, its machine, its knobs and its input are recorded."""
        record = full_run.record

        assert_that(record["schema_version"]).is_equal_to(1)
        assert_that(record["label"]).is_equal_to(LABEL)
        assert_that(record["environment"]["host"]).is_not_empty()
        assert_that(record["environment"]["cpu_count"]).is_greater_than_or_equal_to(1)
        assert_that(record["environment"]["carmack_version"]).is_equal_to(FAKE_VERSION)
        assert_that(record["environment"]["git_commit"]).matches(r"^[0-9a-f]{40}$")
        assert_that(record["invocation"]).contains_entry(
            {"chemistry": CHEMISTRY}, {"workers": WORKERS}, {"prefix": PREFIX}
        )
        assert_that(record["invocation"]["stages"]).is_equal_to(list(STAGE_NAMES))
        assert_that(Path(record["input"]["path"])).is_equal_to(full_run.input_fastq)
        assert_that(record["input"]["bytes"]).is_equal_to(len(FASTQ_BYTES))
        assert_that(record["input"]["sha256"]).is_equal_to(FASTQ_SHA256)

    def test_record_measures_every_stage(self, full_run: BenchmarkRun) -> None:
        """Test that each stage's cost is recorded and plausible, on lower bounds alone.

        A loaded machine makes any ceiling a future flake.
        """
        for entry in full_run.record["stages"]:
            wall, cpu, rss = entry["wall_seconds"], entry["cpu_percent"], entry["max_rss_bytes"]
            described = f"stage {entry['name']}"

            assert_that(entry["exit_code"]).described_as(described).is_equal_to(0)
            assert_that(wall).described_as(described).is_greater_than_or_equal_to(SPIN_SECONDS)
            assert_that(cpu).described_as(described).is_greater_than(0)
            assert_that(rss).described_as(described).is_greater_than(MIN_RSS_BYTES)

        total = full_run.record["total_wall_seconds"]
        assert_that(total).is_greater_than_or_equal_to(SPIN_SECONDS * len(STAGE_NAMES))

    def test_record_inventories_what_each_stage_produced(self, full_run: BenchmarkRun) -> None:
        """Test that a stage's outputs are attributed to it, and the harness's own are not."""
        for entry in full_run.record["stages"]:
            names = {output["name"] for output in entry["outputs"]}
            described = assert_that(names).described_as(entry["name"])
            stats_name = f"{PREFIX}.{entry['name']}.stats.txt"

            # The stats file as well as the chained FASTQ, so an inventory built by diffing
            # the directory is visibly not a hard-coded per-stage file list.
            described.contains(STAGE_OUTPUT_NAMES[entry["name"]], stats_name)
            described.does_not_contain("benchmark.json", "benchmark.tsv", f"{entry['name']}.log")
            for output in entry["outputs"]:
                assert_that(output["bytes"]).described_as(output["name"]).is_greater_than(0)

    def test_tsv_has_a_header_and_one_row_per_stage(self, full_run: BenchmarkRun) -> None:
        """Test the table's shape, and that the run metadata repeats on every row."""
        rows = full_run.tsv_rows
        column = {name: index for index, name in enumerate(rows[0])}
        body = rows[1:]

        assert_that(rows[0]).is_equal_to(list(benchmark_pipeline.TSV_COLUMNS))
        assert_that(rows).is_length(len(STAGE_NAMES) + 1)
        assert_that([row[column["stage"]] for row in body]).is_equal_to(list(STAGE_NAMES))
        assert_that({row[column["label"]] for row in body}).is_equal_to({LABEL})
        assert_that([row[column["workers"]] for row in body]).is_equal_to([str(WORKERS), "", ""])


class TestPartialRun:
    """Tests for a run that starts partway along the chain."""

    def test_mid_chain_selection_runs_only_named_stages(self, subset_run: BenchmarkRun) -> None:
        """Test that the first selected stage reads ``--fastq`` and the rest never run."""
        selected = ["extract-umis", "assign-targets"]
        rows = subset_run.tsv_rows

        assert_that(subset_run.exit_code).described_as(subset_run.console).is_equal_to(0)
        assert_that(subset_run.stages_run).is_equal_to(selected)
        assert_that(subset_run.record["invocation"]["stages"]).is_equal_to(selected)
        assert_that(subset_run.calls[0].input_path).is_equal_to(subset_run.input_fastq)
        assert_that([row[rows[0].index("stage")] for row in rows[1:]]).is_equal_to(selected)

    def test_nothing_is_written_outside_the_output_dir(self, subset_run: BenchmarkRun) -> None:
        """Test that the run leaves its working directory and its scratch root untouched."""
        working = subset_run.working_dir

        assert_that(list(working.iterdir())).described_as(str(working)).is_empty()
        assert_that({path.name for path in subset_run.output_dir.parent.iterdir()}).is_equal_to(
            {"cli", "out", "cwd", "reads.fastq"}
        )


class TestStageFailure:
    """Tests for a stage that exits non-zero partway through the chain."""

    def test_failing_stage_stops_the_chain_and_is_recorded(self, failed_run: BenchmarkRun) -> None:
        """Test that it exits non-zero, runs no later stage, and still writes both records."""
        stages = failed_run.record["stages"]

        assert_that(failed_run.exit_code).described_as(failed_run.console).is_not_equal_to(0)
        assert_that(failed_run.stages_run).is_equal_to(["extract-barcodes", "extract-umis"])
        assert_that(failed_run.record["ok"]).is_false()
        assert_that([entry["name"] for entry in stages]).is_equal_to(failed_run.stages_run)
        assert_that(stages[0]["exit_code"]).is_equal_to(0)
        assert_that(stages[1]["exit_code"]).is_equal_to(FAILING_EXIT_CODE)
        assert_that(Path(stages[1]["log"]).is_file()).is_true()
        assert_that(failed_run.tsv_rows).is_length(3)


class TestRefusals:
    """Tests for command lines the harness turns down before running anything."""

    @pytest.mark.parametrize(
        "extra, named",
        [({"stages": "extract-barcodes,assign-targets"}, "--stages"), ({}, "no-such-reads.fastq")],
        ids=["non-contiguous-stages", "missing-input-file"],
    )
    def test_a_bad_command_line_is_refused_by_name_and_writes_nothing(
        self, tmp_path: Path, extra: dict[str, str], named: str
    ) -> None:
        """Test that the harness names what was wrong and leaves no output directory behind."""
        output_dir = tmp_path / "out"

        exit_code, console = invoke_harness(
            harness_argv(
                fastq=str(tmp_path / "no-such-reads.fastq"), chemistry=CHEMISTRY,
                output_dir=str(output_dir), **extra,
            )  # fmt: skip
        )

        assert_that(exit_code).described_as(console).is_not_equal_to(0)
        assert_that(console).contains(named)
        assert_that(output_dir.exists()).is_false()


class TestRealCarmackCli:
    """Tests driving the real CLI, selected with ``-k`` only so the suite stays fast."""

    @pytest.mark.only_run_with_direct_target
    def test_real_carmack_cli_runs_every_stage_of_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that every flag the harness builds is one the real CLI still accepts.

        The stand-in the rest of this file drives records whatever argv it is handed and
        exits zero, so a renamed or removed option satisfies every other test here. Click
        aborts on an option it does not know, so a stage exiting zero had its ``-c``, ``-o``,
        ``-p`` and ``-n`` all still current.
        """
        working_dir = tmp_path / "cwd"
        working_dir.mkdir()
        output_dir = tmp_path / "out"

        # The child resolves ``-m carmack`` against its own import path, so the checkout goes
        # ahead of site-packages rather than relying on where pytest started. An install on
        # PATH can predate a stage entirely.
        monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT), prepend=os.pathsep)
        monkeypatch.chdir(working_dir)

        exit_code, console = invoke_harness(
            harness_argv(
                fastq=str(REAL_CLI_FASTQ), chemistry=CHEMISTRY, output_dir=str(output_dir),
                workers="1", prefix=PREFIX, label="real-cli-run",
                carmack_command=REAL_CLI_COMMAND,
            )  # fmt: skip
        )

        assert_that(exit_code).described_as(console).is_equal_to(0)
        record = json.loads((output_dir / "benchmark.json").read_text())

        assert_that(record["ok"]).described_as(console).is_true()
        assert_that([entry["name"] for entry in record["stages"]]).is_equal_to(list(STAGE_NAMES))

        # "unknown" is what the harness records when --version goes unanswered, so a real
        # version proves the metadata command reached the CLI that ran the stages.
        assert_that(record["environment"]["carmack_version"]).is_not_equal_to("unknown")

        for entry in record["stages"]:
            described = f"{entry['name']}, see {entry['log']}"
            produced = {output["name"] for output in entry["outputs"]}
            expected_output = STAGE_OUTPUT_NAMES[entry["name"]]

            assert_that(entry["exit_code"]).described_as(described).is_equal_to(0)
            assert_that(entry["max_rss_bytes"]).described_as(described).is_greater_than(0)
            assert_that(produced).described_as(described).contains(expected_output)

        assert_that(list(working_dir.iterdir())).described_as(str(working_dir)).is_empty()
