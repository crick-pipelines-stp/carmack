# Compressed output

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
