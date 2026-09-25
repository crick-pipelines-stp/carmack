# Prepare-reads writer lifecycle

`prepare-reads` opens a dynamic number of gzip writers around a forked `ProcessPoolExecutor`:
three fixed files for the scRNA arm plus one `(R1, R2)` pair per entry in the chemistry's
target index whitelist, so the total is not known until the chemistry resolves. Every writer
is entered through a single `contextlib.ExitStack` strictly before the executor, which is
entered last. `ExitStack` unwinds in the reverse of entry order, so the executor is always
the first thing torn down: a `ProcessPoolExecutor` forks its workers on the first batch it is
handed, and a forked worker inherits the write end of every compressor's input pipe already
open at that point, so a writer entered after the executor would never see its pipe closed
by the workers that inherited it, and closing it from the parent alone would wait forever.
This is the same ordering invariant whose violation caused a real production deadlock in
`extract-barcodes` (fixed in `a4e56f25`, "shut the worker pool down before the gzip
writers"), generalized here to a writer count that depends on the chemistry's whitelist
rather than being fixed.

R2 is never handed to a worker process: everything a writer needs is computed from R1 alone,
so R2 is threaded around the pool instead, in a sidecar deque kept aligned with the R1
batches by the in-order driver's strict submission-order guarantee. Only R1 pays the cost of
pickling across the pool boundary.

The `-n`/`--cpu_count` default of 16 mirrors `assign-targets`'s own measured saturation cap
([Target assignment workers](assign-targets-workers.md))
as a starting point, not a claim that this stage's own throughput has been independently
measured at that number - it has not, and the CLI help says so.
