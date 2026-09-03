#!/usr/bin/env python
r"""
Repeatable benchmark harness for the carmack pipeline.

The harness drives the installed ``carmack`` CLI as a black box: it never imports the
package, it shells out and measures each stage from the kernel. It reports, it does not
gate -- no pass/fail threshold is baked in. Run it with ``--help`` for the options.

Nothing is written outside ``--output-dir``: each run leaves ``benchmark.json`` (the full
record), ``benchmark.tsv`` (one row per stage) and one ``<stage>.log`` holding that stage's
combined stdout and stderr. Both records are machine-readable, so two runs are compared
with whatever already diffs JSON or TSV.

Two honesty points about the numbers. ``ru_maxrss`` is reported in KiB on Linux and in
bytes on macOS, so it is converted explicitly on ``sys.platform`` before being recorded as
bytes. And peak RSS is a maximum over the process tree, not a sum: ``wait4`` on the direct
child folds in the descendants that child reaped itself, so the figure is "the largest
single process in the tree" rather than "total concurrent footprint". GNU time reports the
same quantity, so comparability between runs is unaffected.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RECORD_NAME = "benchmark.json"
TABLE_NAME = "benchmark.tsv"

DEFAULT_PREFIX = "bench"
DEFAULT_CARMACK_COMMAND = "carmack"
DEFAULT_WORKERS = 1
DEFAULT_STAGE_SPEC = "all"

BYTES_PER_MIB = 1024 * 1024

# ru_maxrss is KiB everywhere the pipeline runs except macOS, which reports bytes.
RSS_BYTES_PER_UNIT = 1 if sys.platform == "darwin" else 1024


@dataclass(frozen=True)
class Stage:
    """
    One pipeline stage the harness knows how to drive.

    Attributes:
        name: The carmack subcommand name.
        output_suffix: Suffix appended to the run prefix to name the FASTQ this stage
            emits, which is the next stage's input.
        supports_workers: Whether the stage's CLI accepts ``-n/--cpu_count``.
    """

    name: str
    output_suffix: str
    supports_workers: bool


STAGES: tuple[Stage, ...] = (
    Stage(name="extract-barcodes", output_suffix=".r1_annotated.fastq.gz", supports_workers=True),
    Stage(name="extract-umis", output_suffix=".r1_umi.fastq.gz", supports_workers=False),
    Stage(name="assign-targets", output_suffix=".r1_tgidx.fastq.gz", supports_workers=False),
)

# The harness writes these into the output directory itself, so the snapshot diff that
# attributes files to stages must never mistake one of them for a pipeline output.
HARNESS_FILE_NAMES = frozenset({RECORD_NAME, TABLE_NAME} | {f"{s.name}.log" for s in STAGES})

TSV_COLUMNS = (
    "label git_commit carmack_version host cpu_count stage workers exit_code "
    "wall_seconds cpu_percent max_rss_mib"
).split()


def capture_output(command: Sequence[str]) -> str | None:
    """
    Run a metadata command and return its stripped stdout, or None if it did not work.
    """
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True, check=False)
    except OSError:
        return None

    return completed.stdout.strip() if completed.returncode == 0 else None


def hash_file(path: Path) -> str:
    """
    Return the hex sha256 of a file's bytes on disk.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_stages(spec: str) -> tuple[Stage, ...]:
    """
    Turn a ``--stages`` value of "all", or of comma-separated names, into stages to run.

    The selection must name a contiguous run of the canonical order, in that order. A gap
    is refused rather than guessed at: the stage after the gap would otherwise be handed a
    FASTQ that never went through the one that was skipped.

    Raises:
        ValueError: If the selection is not a contiguous run of known stage names.
    """
    if spec.strip() == DEFAULT_STAGE_SPEC:
        return STAGES

    names = [part.strip() for part in spec.split(",") if part.strip()]
    order = [stage.name for stage in STAGES]

    for start in range(len(order)):
        if names and names == order[start : start + len(names)]:
            return STAGES[start : start + len(names)]

    raise ValueError(f"--stages must be a contiguous run of {' -> '.join(order)}, not '{spec}'")


def build_stage_command(
    stage: Stage,
    carmack_command: list[str],
    input_path: Path,
    chemistry: str,
    output_dir: Path,
    prefix: str,
    workers: int,
) -> list[str]:
    """
    Build the argv for one stage, led by the tokens naming the carmack CLI.

    The worker count is passed only to a stage whose CLI accepts it; handing it to one that
    does not would abort that stage rather than merely be untidy.
    """
    command = [*carmack_command, stage.name, "-c", chemistry, "-o", str(output_dir), "-p", prefix]

    if stage.supports_workers:
        command += ["-n", str(workers)]

    command.append(str(input_path))

    return command


def measure_command(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    """
    Run a command to completion and take its cost from the kernel.

    Both child streams go to a file rather than a pipe: the harness blocks in ``wait4`` and
    so cannot drain a pipe, and every carmack invocation prints a banner to stderr that
    would otherwise bury the summary.
    """
    with log_path.open("wb") as log_handle:
        started = time.monotonic()
        # Deliberately not a `with` block: the context manager's exit calls wait(), and this
        # child is reaped below by wait4 instead.
        # pylint: disable=consider-using-with
        process = subprocess.Popen(list(command), stdout=log_handle, stderr=subprocess.STDOUT)
        # wait4 hands back (pid, status, rusage), and the pid is already known.
        status, usage = os.wait4(process.pid, 0)[1:]
        wall_seconds = time.monotonic() - started

    # Recording the exit code on the Popen stops __del__ reaping a pid wait4 already reaped.
    process.returncode = os.waitstatus_to_exitcode(status)
    cpu_seconds = usage.ru_utime + usage.ru_stime

    return {
        "exit_code": process.returncode,
        "wall_seconds": round(wall_seconds, 6),
        "user_seconds": round(usage.ru_utime, 6),
        "system_seconds": round(usage.ru_stime, 6),
        "cpu_percent": round(cpu_seconds / wall_seconds * 100.0 if wall_seconds > 0 else 0.0, 1),
        "max_rss_bytes": int(usage.ru_maxrss) * RSS_BYTES_PER_UNIT,
    }


def snapshot_output_files(output_dir: Path) -> set[str]:
    """
    List the pipeline files in the output directory, excluding the harness's own artefacts.
    """
    return {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in HARNESS_FILE_NAMES
    }


def collect_environment(carmack_command: Sequence[str]) -> dict[str, Any]:
    """
    Describe the machine and the code the run happened on, which is what makes "these two
    runs are comparable" checkable rather than assumed.
    """
    # Click prints "<prog>, version <number>", and the prog name changes with how the CLI
    # was invoked, so only the number is kept.
    reported = capture_output([*carmack_command, "--version"])

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "carmack_version": reported.split()[-1] if reported else "unknown",
        "git_commit": capture_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(capture_output(["git", "status", "--porcelain"])),
    }


def render_tsv(record: dict[str, Any]) -> str:
    """
    Render the run as one row per stage, header included, ending in a newline.

    The run metadata is repeated on every row so that two runs' rows concatenate into one
    self-describing before/after table.
    """
    environment = record["environment"]
    lines = ["\t".join(TSV_COLUMNS)]

    for entry in record["stages"]:
        lines.append(
            f"{record['label']}\t{environment['git_commit'] or ''}\t"
            f"{environment['carmack_version']}\t{environment['host']}\t"
            f"{environment['cpu_count']}\t{entry['name']}\t"
            f"{'' if entry['workers'] is None else entry['workers']}\t{entry['exit_code']}\t"
            f"{entry['wall_seconds']:.3f}\t{entry['cpu_percent']:.1f}\t"
            f"{entry['max_rss_bytes'] / BYTES_PER_MIB:.1f}"
        )

    return "\n".join(lines) + "\n"


def run_stage(stage: Stage, command: list[str], output_dir: Path, workers: int) -> dict[str, Any]:
    """
    Run one stage's command and record everything known about it.

    The files the stage produced are found by snapshotting the output directory either side
    of the run and diffing, which is more robust than a hard-coded per-stage file list.
    """
    log_path = output_dir / f"{stage.name}.log"
    before = snapshot_output_files(output_dir)
    measurement = measure_command(command, log_path)
    produced = sorted(snapshot_output_files(output_dir) - before)

    return {
        "name": stage.name,
        "command": command,
        "workers": workers if stage.supports_workers else None,
        **measurement,
        "log": str(log_path),
        "outputs": [{"name": n, "bytes": (output_dir / n).stat().st_size} for n in produced],
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """
    Run the selected stages, measuring each, and write the run record.

    A stage that exits non-zero is recorded, stops the chain, and makes the harness exit
    non-zero, but the measurements taken before it are still written out.
    """
    try:
        stages = select_stages(args.stages)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    input_path = args.fastq.resolve()
    if not input_path.is_file():
        print(f"error: no input file at {input_path}.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    carmack = shlex.split(args.carmack_command)
    # carmack's -o is a click.Path(exists=True), so the directory must be there already.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Hashed before the timed section so it never contaminates a measurement.
    digest = hash_file(input_path)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "total_wall_seconds": 0.0,
        "environment": collect_environment(carmack),
        "invocation": {
            "chemistry": args.chemistry,
            "workers": args.workers,
            "prefix": args.prefix,
            "stages": [stage.name for stage in stages],
            "carmack_command": carmack,
        },
        "input": {"path": str(input_path), "bytes": input_path.stat().st_size, "sha256": digest},
        "stages": [],
    }

    environment = record["environment"]
    dirty = " (dirty)" if environment["git_dirty"] else ""
    print(
        f"carmack benchmark   label={args.label}   host={environment['host']}   "
        f"commit={(environment['git_commit'] or 'unknown')[:8]}{dirty}   "
        f"carmack={environment['carmack_version']}\n"
        f"input   {input_path.name}   {record['input']['bytes'] / BYTES_PER_MIB:.1f} MiB   "
        f"sha256 {digest[:16]}   chemistry={args.chemistry}   workers={args.workers}\n"
    )
    print(f"{'stage':<20}{'wall':>10}{'%CPU':>9}{'peak RSS':>12}   status")

    stage_input = input_path
    for stage in stages:
        # Named before the stage starts so a long run is never silent.
        print(f"{stage.name:<20}", end="", flush=True)
        command = build_stage_command(
            stage, carmack, stage_input, args.chemistry, output_dir, args.prefix, args.workers
        )
        entry = run_stage(stage, command, output_dir, args.workers)
        record["stages"].append(entry)

        status = "ok" if entry["exit_code"] == 0 else f"FAILED (exit {entry['exit_code']})"
        print(
            f"{entry['wall_seconds']:>9.1f} s{entry['cpu_percent']:>8.0f}%"
            f"{entry['max_rss_bytes'] / BYTES_PER_MIB:>8.1f} MiB   {status}"
        )

        if entry["exit_code"] != 0:
            record["ok"] = False
            print(f"stage {stage.name} failed. Read {entry['log']} for what it said.")
            break

        stage_input = output_dir / f"{args.prefix}{stage.output_suffix}"

    record["total_wall_seconds"] = round(sum(e["wall_seconds"] for e in record["stages"]), 6)
    print(f"{'total':<20}{record['total_wall_seconds']:>9.1f} s\n")

    (output_dir / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (output_dir / TABLE_NAME).write_text(render_tsv(record), encoding="utf-8")
    print(f"wrote {output_dir / RECORD_NAME} and {output_dir / TABLE_NAME}")

    return 0 if record["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark_pipeline.py",
        description="Measure the wall time, CPU use and peak memory of each carmack stage.",
    )
    add = parser.add_argument

    add("--fastq", required=True, type=Path, help="Input FASTQ for the first selected stage.")
    add("--chemistry", required=True, help="Chemistry name, passed to every stage.")
    add("--output-dir", required=True, type=Path, help="Directory to run in. Created if absent.")
    add("--workers", type=int, default=DEFAULT_WORKERS, help="Workers for stages that take one.")
    add("--stages", default=DEFAULT_STAGE_SPEC, help="Contiguous run of stage names, or 'all'.")
    add("--prefix", default=DEFAULT_PREFIX, help="Prefix for generated filenames.")
    add("--label", default="", help="Free-text tag recorded on the run and on every TSV row.")
    add("--carmack-command", default=DEFAULT_CARMACK_COMMAND, help="Command invoking the CLI.")

    return parser


def main(argv: Sequence[str]) -> int:
    """
    Run the harness and return the process exit code, given argv without the program name.
    """
    return run_benchmark(build_parser().parse_args(list(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
