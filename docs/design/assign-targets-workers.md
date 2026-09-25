# Target assignment workers

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
