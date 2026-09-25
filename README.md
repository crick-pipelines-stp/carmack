# carmack

[![Docker tests](https://github.com/crick-pipelines-stp/carmack/actions/workflows/tests.yml/badge.svg)](https://github.com/crick-pipelines-stp/carmack/actions/workflows/tests.yml)
[![Publish image](https://github.com/crick-pipelines-stp/carmack/actions/workflows/publish.yml/badge.svg)](https://github.com/crick-pipelines-stp/carmack/actions/workflows/publish.yml)
[![ghcr.io](https://img.shields.io/badge/ghcr.io-carmack-2496ED?logo=docker&logoColor=white)](https://github.com/crick-pipelines-stp/carmack/pkgs/container/carmack)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Single-cell multi-omic tools: a command-line toolkit that takes raw sequencing reads from
a barcoded single-cell library through cell barcode, UMI and target-index extraction and
read preparation, then deduplicates the aligned reads and calls cells.

## Installation

### Container image

CI publishes the `runtime` target of the `Dockerfile` to
`ghcr.io/crick-pipelines-stp/carmack` (public, no login needed) after the test
suite passes, from `.github/workflows/publish.yml`:

| Tag | Pushed on | Moves? |
|---|---|---|
| `sha-<7-char commit>` | every push to `main`, every release | never |
| `main` | every push to `main` | yes |
| `X.Y.Z`, `X.Y` | a `vX.Y.Z` git tag | `X.Y` does |
| `latest` | a `vX.Y.Z` git tag that is not a pre-release | yes |

Pin a `sha-*` tag together with its digest (printed in the run's summary), for
example `ghcr.io/crick-pipelines-stp/carmack:sha-01f2d18@sha256:…`; a tag on its
own can in principle be re-pushed. `carmack --version` inside the image reports
`X.Y.Z` for a release, `X.Y.Z.N` for N commits past one, and `0.0.0+<sha>`
before the first release. The image includes `ps`, which Nextflow needs to run a
task at all. No `local` tag is ever published, so a consumer's own
`docker build` of this repo can use that name without colliding.

### From source

carmack needs Python 3.12 or newer.

```sh
pip install git+https://github.com/crick-pipelines-stp/carmack.git
```

Put [`pigz`](https://zlib.net/pigz/) on `PATH` as well. Every stage that writes `.gz`
output uses it to compress in parallel. Without it the output is identical but slower, and
the stage logs a warning.

## Usage

```sh
carmack --help              # list the commands
carmack <command> --help    # options for one command
carmack --version
```

The commands are grouped in `--help`: the user-facing processing stages
(`extract-barcodes`, `extract-umis`, `assign-targets`, `prepare-reads`, `linear-dedup`,
`bam-tag-deduplicate`, `call-cells`) come first, and utilities such as `split-bam` follow.
The read-level stages take a `--chemistry`/`-c` name that selects the read layout and
whitelists shipped in `carmack/data/`.

These options apply to every command and go before the command name:

| Option | Effect |
|---|---|
| `-v`, `--verbose` | Print verbose output to the console. |
| `--hide-progress` | Don't show progress bars. |
| `-l`, `--log-file <filename>` | Save a verbose log to a file. |

```sh
carmack -l run.log --hide-progress extract-barcodes -c <chemistry> -o out/ reads_R1.fastq.gz
```

Commands that run on a worker pool take `-n`/`--cpu_count` to set its size.

## Documentation

See [`docs/`](docs/README.md) for the contributor guide and the design notes behind
individual stages.

## License

carmack is released under the [MIT License](LICENSE).
