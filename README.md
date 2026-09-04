# carmack
Single-cell multi-omic tools

## Development

### Git hooks

Version-controlled hooks live in `.githooks/`. Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook auto-formats staged Python with `isort` then `black`,
using the config in `pyproject.toml`. Install the dev tools with
`pip install -e ".[dev]"`. Skip it for a single commit with
`git commit --no-verify`.

### Golden output baseline

`tests/test_golden_outputs.py` runs real barcode, UMI and target assignment stages over
committed FASTQ inputs and compares every output file against blessed copies in
`tests/data/golden/expected/`. It exists so that any change to the matchers shows up as
a diff rather than going unnoticed. Each case runs only the stages its chemistry
supports: the `carmack_custom_seq_1_0` cases run all three, the HyDrop cases stop after
barcode extraction.

Two small cases run as part of the normal suite (about 30s). Two full-scale 2000-read
cases are marked `only_run_with_direct_target`, so they are skipped unless `-k` selects
them; the HyDrop one alone takes roughly eleven minutes.

```sh
python -m pytest tests/test_golden_outputs.py            # always-run tier
python -m pytest tests/test_golden_outputs.py -k golden  # adds the full-scale tier
```

A failing golden test means the pipeline's output changed. Work out whether that change
was intended before doing anything else. If it was, re-bless the goldens and review the
resulting diff as part of the change:

```sh
CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py
CARMACK_REGEN_GOLDEN=1 python -m pytest tests/test_golden_outputs.py -k golden
```

Run pytest from the repo root, otherwise an installed copy of `carmack` can shadow the
source tree and produce different files.

### Benchmark harness

`benchmark_pipeline.py` runs the pipeline stages (`extract-barcodes`, `extract-umis`,
`assign-targets`) over a fixed read slice and records the wall time, %CPU and peak RSS of
each, so a change can be measured the same way before and after.

```sh
python benchmark_pipeline.py --fastq path/to/R1.fastq.gz \
    --chemistry carmack_custom_seq_1_0 --workers 16 \
    --output-dir results/baseline --label baseline
```

It writes `benchmark.json`, `benchmark.tsv` and a `<stage>.log` per stage into
`--output-dir`, next to the stages' own outputs, and nothing anywhere else.

Peak RSS is a maximum over the process tree rather than a sum: the largest single process
in it, not the total concurrent footprint.

If a `carmack` on `PATH` shadows the source tree a stage dies with click's `No such
command`, so run from the repo root with `--carmack-command "python -m carmack"`.

### Compressed output

Writing `.gz` output prefers `pigz`, which compresses in parallel and keeps compression
off the critical path of every stage that writes gzip. Without it on `PATH` the stages
still produce identical output, just more slowly, and log a warning saying so. Install
it from your package manager if throughput matters; benchmark numbers are not comparable
between a machine that has it and one that does not.

### UMI correction temp space

UMI correction has to see every read of a cell barcode at once, so `extract-umis` used to
hold one record per accepted read in memory for the whole file. That peaked at 610.4 MiB
over 1M reads, which projects to roughly 200 GB on a 400M-read library against the 236 GB
available — the production run could not complete.

Accepted records are now spilled to temporary shard files during the same extraction pass,
the shard chosen by a crc32 of the cell barcode, and correction then runs one shard at a
time. A barcode group lands entirely in one shard by construction, so correcting a shard in
isolation gives the same answer as correcting the whole file at once; each map row carries
the read's extraction ordinal and the shard maps are merged back on it, so `umi_map.tsv` is
byte-identical to what the in-memory pass wrote, on the standing invariant that a read id
occurs once in the input. Over the same 1M reads the peak is now
155.8 MiB. A `--raw` run, which skips correction entirely, peaks at 146.7 MiB, so correction
now costs about 9 MiB over that floor where it used to cost 464 MiB.

Two options control the spill:

```sh
python -m carmack extract-umis sample.r1_annotated.fastq.gz \
    -c carmack_custom_seq_1_0 --temp-dir /fast/scratch --shard-count 256
```

`--temp-dir` is the directory the shard tree is created in; it must already exist, and
defaults to the system temp directory (`TMPDIR`). `--shard-count` (default 256, maximum
1024) is how many shards the records are spread over; the cap is there because the store
holds one open file descriptor per shard. The tree is removed when the run ends, whether it
succeeded or raised.

Size the temp volume before a production run. The spill measured 96.5 bytes per accepted
read (85.8 MiB over 932,065 accepted reads), so a 400M-read library needs roughly 36 GB.
The spill and the finished `umi_map.tsv` are both on disk near the end of the run, so size
for both if `--temp-dir` and `--output_dir` share a filesystem.

More shards means a lower peak, but the peak is O(N / shard_count), not constant — it still
grows with the library. Measured at 1M reads, the growth over the floor scales with the read
count at any fixed shard count, so the default 256 projects to roughly 1-2 GB at 400M reads
rather than staying at the 155 MiB seen here. That is a chosen budget rather than a hard
bound, and it is what makes the run feasible; it is not constant memory. Past about 256 the
peak stops improving on this data, and 1024 measured slightly worse than 256, because the
store's own per-shard overhead starts to outweigh the smaller resident shard. The peak is
also floored by the largest single cell-barcode group, which is indivisible.

A run killed by a signal cannot clean up after itself: the spill tree survives a `SIGKILL`
from the OOM killer and a `SIGTERM` from a scheduler cancellation or timeout. After a killed
run, remove any leftover `carmack-umi-*` directory under the `--temp-dir` that was used. This
matters at production scale, where the stray tree is tens of gigabytes, and it matters more
when `--temp-dir` is left at a default `TMPDIR` that is RAM-backed tmpfs.
