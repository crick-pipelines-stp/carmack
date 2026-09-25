# carmack documentation

Start with the top-level [`README.md`](../README.md) for installation and CLI usage.

## Contributing

- [Development](development.md): setup, git hooks, code style, running the tests, and the
  golden output baseline.

## Design notes

Why an internal works the way it does, one topic per file.

- [Compressed output](design/compressed-output.md): `pigz` for writing, and why a failed
  compressor or decompressor fails the stage.
- [Target assignment workers](design/assign-targets-workers.md): the `assign-targets`
  worker pool, its default cap of 16, and its memory footprint.
- [Prepare-reads writer lifecycle](design/prepare-reads-writer-lifecycle.md): the ordering
  of gzip writers and the forked worker pool in `prepare-reads`.
