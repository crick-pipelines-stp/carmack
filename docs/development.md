# Development

## Setup

carmack targets Python 3.12. Install it editable with the dev tools and the test
dependencies:

```sh
pip install -e ".[dev,tests]"
```

Install `pigz` too if you want timings that match production; see
[Compressed output](design/compressed-output.md).

## Git hooks

Version-controlled hooks live in `.githooks/`. Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook auto-formats staged Python with `isort` then `black`,
using the config in `pyproject.toml`. Install the dev tools with
`pip install -e ".[dev]"`. Skip it for a single commit with
`git commit --no-verify`.

## Code style

`black` (line length 99) and `isort` (black profile) format the code; `ruff` lints it. All
three read their configuration from `pyproject.toml`.

## Running the tests

```sh
python -m pytest
```

Run pytest from the repo root, otherwise an installed copy of `carmack` can shadow the
source tree and produce different files.

CI runs the suite inside the `Dockerfile`'s `test` target (`.github/workflows/tests.yml`).
To reproduce that locally:

```sh
docker build --target test -t carmack:test .
docker run --rm carmack:test
```

## Golden output baseline

`tests/test_golden_outputs.py` runs real barcode extraction, UMI extraction, target
assignment and prepare-reads stages over committed FASTQ inputs and compares their gzip,
text and plot outputs against blessed copies in `tests/data/golden/expected/`. It exists so
that any change to the matchers shows up as a diff rather than going unnoticed. Each case
runs only the stages its chemistry supports: the `carmack_custom_seq_1_0` cases run all
four, the HyDrop cases stop after barcode extraction, since that chemistry declares neither
a UMI component nor a target index.

The MultiQC `*_mqc.json` payloads are deliberately not blessed. They are asserted instead
against the shape MultiQC requires of a custom-content file, and against the exact set of
payloads the stages that ran should have produced, so a change to what the stats say does
not force a rebless, while a payload MultiQC could not read still fails.

Three small cases run as part of the normal suite (about 15s). One of the three is
`carmack_custom_seq_1_0_primd`, which has no blessed baseline: it is built by padding the
`carmack_custom_seq_1_0` input into the primd layout and runs the MultiQC payload checks
alone, so that all three shipped chemistries are covered without a third set of goldens.
Two full-scale 2000-read cases are marked `only_run_with_direct_target`, so they are
skipped unless `-k` selects them; that tier takes under three minutes, most of it the
HyDrop case.

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

## Documentation

`README.md` is the first-touch document: what carmack is, how to install it, and how to
drive the CLI in general. Keep module-specific detail out of it.

Everything else goes under `docs/`:

- `docs/development.md` (this file) is contributor how-to: setup, tooling, tests.
- `docs/design/<topic>.md` records why an internal works the way it does, one file per
  topic. Add a new file rather than growing an existing one, and link it from
  [`docs/README.md`](README.md).
