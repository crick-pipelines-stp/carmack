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

Two small cases run as part of the normal suite (about 10s). Two full-scale 2000-read
cases are marked `only_run_with_direct_target`, so they are skipped unless `-k` selects
them; that tier takes under three minutes, most of it the HyDrop case.

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

### Compressed output

Writing `.gz` output prefers `pigz`, which compresses in parallel and keeps compression
off the critical path of every stage that writes gzip. Without it on `PATH` the stages
still produce identical output, just more slowly, and log a warning saying so. Install
it from your package manager if throughput matters; timings are not comparable
between a machine that has it and one that does not.

A compressor that dies mid-stream now fails the stage, naming the command and its exit
status. Its status used to be waited for and discarded, so the stage returned normally and
exited 0 over a truncated `.gz`, and wrote a report describing reads that never reached
disk; the next stage then read the short file and reported a plausible-looking count. The
triggers are all operational — the output volume filling up mid-write, pigz being
OOM-killed (it runs four threads per stream, and `extract-barcodes` holds three write
streams open at once), or a build of pigz that rejects `-p` — and none of them announce
themselves any other way.

Reads are checked the same way, so a malformed input `.gz` now fails the stage rather than
being silently short-read. That covers a truncated or empty file and, more importantly, one
whose body has a corrupted byte: it decompresses to the full length with the wrong contents,
and the exit status is the only thing that says so. Legitimately concatenated multi-member
files still read normally, which matters because real sequencing FASTQ often arrives that
way, and a read deliberately abandoned part-way through is not a failure either, since
closing a decompressor's pipe early kills it through no fault of its own.

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

### Target assignment workers

`assign-targets` was a single serial loop, and took 139.4 s over a 932,065-read slice. It now
runs on the same order-preserving worker pool as `extract-barcodes`, with `-n`/`--cpu_count`
setting how many processes it uses; the same slice takes 18.5 s at 8 workers and 12.3 s at
16. Output does not depend on that number. Batches are handed on in submission order rather
than in the order the workers happen to finish them, so the annotated FASTQ is byte-identical
to the serial output at every worker count and across repeat runs, and `tgidx_stats.txt`
differs only in its version and timestamp lines.

The default is capped at 16 rather than tracking the machine's core count as
`extract-barcodes` does. Throughput saturates there: past about 16 workers the stage's own
single-threaded parse-and-write loop is the bound, and 32 workers measured 12.17 s against 16
workers' 12.29 s, a difference inside the 11.72-12.29 s spread of repeat runs. Asking for
more than that costs memory and returns no wall time.

Memory is bounded by the in-flight window rather than by the input. At most `2n` batches of
at most 2500 reads are held at once and the input is read no further ahead, so a 400M-read
library peaks where this slice does at the same worker count. Sampling the whole process
tree's proportional set size during a run gives 237.7 MiB at 8 workers, 325.0 MiB at 16 and
542.2 MiB at 32, against roughly 150 MiB for the serial single-process run. Size a machine
against those figures, not against the peak RSS `/usr/bin/time` reports, which is the largest
single process in the tree rather than the sum of them.
