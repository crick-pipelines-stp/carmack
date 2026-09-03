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
