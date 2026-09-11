"""Target index assignment from the UMI-annotated R1 FASTQ.

For each annotated read the assigner locates the anchor homopolymer run, derives
a bounded window off the end of that run, matches the target index lying in the
window against the chemistry whitelist, and re-emits the read carrying either a
whitelist entry with its ``TGIDX_POS`` span or ``TGIDX=NONE``.

The run's start is not recorded on the read. It is computed from the chemistry:
the span end of the nearest anchor whose position barcode extraction wrote down,
plus the fixed bases the layout places between that anchor and the run. So the
only tag this stage consumes is a barcode position tag, and it does not depend
on UMI extraction having run -- though the canonical order still puts UMI
extraction first, because that is what carries the ``UMI`` tag onto the read.

This is an assignment, not a filter. ``NONE`` is the correct answer for every
scRNA read in a mixed library, so no read is ever dropped and reads written
reconciles with reads read. That shapes the failure model: the checks in this
module are all about the chemistry, made once at construction, because a
chemistry that could never assign a target is an operator error worth failing
on, whereas a read that carries no target is simply a read that carries no
target.
"""

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path

from carmack.assign_targets.assign_reporting import AssignCounts, AssignStats
from carmack.assign_targets.tgidx_locator import locate_anchor_run, locate_tgidx_window
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.mqc_report import write_mqc_json
from carmack.parallel import map_batches_in_order
from carmack.utils import get_prefix, progress_bar

log = logging.getLogger(__name__)

# Written to the TGIDX tag of every read no whitelist entry was assigned to.
# It is a value, not an absence: a read carrying it has been through the stage
# and come out unassigned, which is a different thing from a read that never
# reached it, and every downstream consumer keys on this literal to tell them
# apart.
NO_TARGET = "NONE"

# Upper bound on the reads a single batch carries. It bounds resident memory: a
# batch is held whole while it is tallied and again while it is written, so
# raising it raises the stage's memory ceiling by the same factor.
MAX_READS_PER_BATCH = 2500

# Measured saturation point for this stage: beyond about sixteen workers the
# parent's own single-threaded parse-and-write loop is the bound, and thirty-two
# workers measured no faster than sixteen. The number is a measurement, not a
# preference, so the command line caps its default here rather than at every
# core the machine happens to have.
DEFAULT_MAX_WORKERS = 16

# Set once per worker process by the pool initializer, then read-only. Holding
# the assigner here rather than binding it into each submitted batch is what
# keeps the k-mer index out of the per-batch payload.
WORKER_ASSIGNER: "TargetAssigner | None" = None


class TargetAssigner:
    """Assigns target indices to annotated R1 reads using its chemistry layout.

    The assigner mirrors the UMI extractor's shape: it resolves the chemistry,
    validates up front that the chemistry could ever locate a target index, and
    streams the reads once to write the annotated FASTQ and stats report.
    """

    def __init__(
        self,
        fastq_file: str,
        chemistry_name: str,
        n_workers: int = 1,
        batch_size: int | None = None,
    ) -> None:
        """Resolve the chemistry and derive the target assignment parameters.

        Args:
            fastq_file: Path to the UMI-annotated R1 FASTQ (produced by UMI
                extraction, carrying the barcode position tags this stage reads).
            chemistry_name: Name of the chemistry describing the read layout.
            n_workers: Width of the process pool the assignment pass runs on.
            batch_size: Reads a submitted batch carries, defaulting to
                ``MAX_READS_PER_BATCH``. A constructor parameter and nothing
                more, deliberately not exposed on the command line: it exists so
                a test fixture can put a small input over more than one batch,
                which is the only way the in-order fold is reached at all.

        Raises:
            ValueError: If the chemistry is unknown, declares no target index
                anchored by a homopolymer run, places no recorded anchor a fixed
                distance 5' of that run, or declares a target index whitelist
                that loads no entries.
        """
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.n_workers = n_workers
        self.batch_size = batch_size or MAX_READS_PER_BATCH
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_target_assignment():
            raise ValueError(
                f"chemistry '{chemistry_name}' declares no target index anchored by a "
                "homopolymer run, so no target can be assigned"
            )

        tgidx = self.chemistry.tgidx_component()
        anchor = self.chemistry.tgidx_anchor()

        # The linchpin of the stage. The anchor run's start is recorded nowhere in
        # the read; it is computed as the end of the nearest recorded anchor span
        # plus the fixed bases the chemistry places between that anchor and the
        # run. Where that distance is not fixed, or where no recorded anchor lies
        # 5' of the run at all, the window would hang off a coordinate belonging
        # to some other component and every read would be scored, silently and
        # plausibly, against arbitrary sequence. There is no downstream symptom to
        # catch that by, so it has to be fatal here.
        try:
            located = self.chemistry.resolve_anchor_offset(anchor)
        except ValueError as error:
            raise ValueError(
                f"chemistry '{chemistry_name}' cannot locate the start of the anchor run "
                f"'{anchor.name}' that its target index hangs off: {error}"
            ) from error

        # Closed here rather than in the chemistry because
        # ChemistryBase.supports_target_assignment() asks only whether the index
        # is preceded by a homopolymer anchor and never looks at the whitelist.
        # An empty whitelist would sail through it and build a KmerMatcher over
        # an empty seed index, which returns no candidates for any window and so
        # reports every read as NONE - a run that looks complete and is entirely
        # a miss.
        whitelist = self.chemistry.tgidx_whitelist()
        if not whitelist:
            raise ValueError(
                f"chemistry '{chemistry_name}' loads no target index whitelist entries, so "
                "every read would be reported as unassigned"
            )

        self.tgidx_name = tgidx.name
        self.tgidx_length = tgidx.length
        self.anchor_base = anchor.homopolymer_base
        self.anchor_min_run = anchor.min_run
        self.max_errors = self.chemistry.max_errors.tgidx
        # Scalars rather than the AnchorOffset itself: the whole assigner is
        # pickled out to every worker in the pool.
        self.anchor_pos_key = located.position_key
        self.run_offset = located.offset

        # Built once, here, because the k-mer index is built inside the
        # matcher's constructor: a matcher created per read would rebuild that
        # index for every read in the FASTQ and dominate the stage's runtime.
        self.matcher = KmerMatcher(
            whitelist=whitelist,
            component=tgidx,
            chemistry=self.chemistry,
            max_errors=self.max_errors,
        )

    def validate_header(self, ann: ReadAnnotation) -> None:
        """Validate that an annotated read carries the tag this stage reads.

        Checks the first annotated read against the supplied chemistry: it must
        carry the position tag of the anchor the run start is measured from,
        since that span's end plus a fixed offset is where the anchor run is
        found. Its absence means the FASTQ was produced with a different
        chemistry than the one supplied, or was never put through barcode
        extraction at all.

        Args:
            ann: The parsed header of the first annotated read.

        Raises:
            ValueError: When the anchor position tag is absent from the header.
        """
        if ann.get(self.anchor_pos_key) is None:
            raise ValueError(
                f"Annotated read '{ann.read_id}' is missing the expected "
                f"'{self.anchor_pos_key}' tag for chemistry '{self.chemistry_name}'. The "
                "annotated FASTQ may have been produced with a different chemistry, or "
                "without running the 'extract-barcodes' stage."
            )

    def assign_read(self, name: str, seq: str, counts: AssignCounts) -> str:
        """Assign a target index to one read and tally the outcome it took.

        The read takes exactly one of four mutually exclusive outcomes - matched,
        no anchor position tag, short window, no match - and each bumps one tally,
        so the four always sum to the reads tallied.

        Args:
            name: The read's annotated header, without its leading ``@``.
            seq: The read's sequence.
            counts: Accumulator the outcome is tallied into, mutated in place so
                a run allocates no per-read outcome object.

        Returns:
            The rendered header, carrying either the assigned whitelist entry
            with its position span or the unassigned sentinel.
        """
        counts.total += 1
        ann = ReadAnnotation.parse(name)
        attempt = None
        window = None

        pos = ann.get(self.anchor_pos_key)
        if pos is None:
            # Fatal on the first read, merely unassigned here: by this point the
            # chemistry has been shown to match the FASTQ, so a read missing the
            # tag is a read, not an operator error.
            counts.no_left_anchor_pos += 1
        else:
            # The chemistry says the run begins a fixed number of bases after the
            # anchor's end. UMI extraction slices its UMI off the same
            # coordinate, so this is the index its own run tally is measured at.
            expected_start = parse_span(pos)[1] + self.run_offset
            run = locate_anchor_run(seq, expected_start, self.anchor_base, self.anchor_min_run)
            run_end = run.end
            counts.run_counts[run.length] += 1
            # The seed length is passed explicitly, never defaulted. Both
            # defaults are 4 today so omitting it would work by coincidence, and
            # TrimWindow.is_matchable only keeps a window the matcher would raise
            # on away from the matcher while the two agree. Threading the
            # matcher's own k through is what stops them drifting apart
            # unnoticed.
            window = locate_tgidx_window(
                seq,
                run_end,
                self.tgidx_length,
                self.max_errors,
                k=self.matcher.k,
            )
            if not window.is_matchable:
                counts.short_window += 1
            else:
                attempts = self.matcher.match(window.sequence)
                # A tie and a total miss both come back as one untouched
                # attempt, so a null match is the only thing to test.
                if len(attempts) == 1 and attempts[0].match is not None:
                    attempt = attempts[0]
                else:
                    counts.no_match += 1

        if attempt is None:
            ann.set(self.tgidx_name, NO_TARGET)
        else:
            counts.matched += 1
            ann.set(self.tgidx_name, attempt.match)
            # The span bounds the sequence found in the read, not the entry it
            # verified against; the two coincide only at edit distance zero.
            # Window coordinates become read coordinates by adding the window's
            # own start.
            ann.set(
                position_key(self.tgidx_name),
                format_span(
                    window.start + attempt.read_idx[0],
                    window.start + attempt.read_idx[1],
                ),
            )
            counts.target_counts[attempt.match] += 1
            counts.edit_distance_counts[attempt.edit_distance] += 1

        return ann.render()

    def assign_targets(self, output_dir: str = ".", prefix: str | None = None) -> AssignStats:
        """Stream the reads, assign target indices and write the output files.

        Every input read is re-emitted exactly once, in input order, with its
        sequence and quality untouched: this stage annotates, it never filters.
        That holds at every worker count, because the driver yields each batch's
        results in submission order rather than in the order the workers finished
        them. ``NONE`` is a valid and expected outcome rather than a failure - it
        is the correct answer for every scRNA read in a mixed library - which is
        what makes reads written reconcile with reads read. That is the sharp
        difference from ``extract-umis``, whose annotated FASTQ holds only the
        reads it succeeded on, so its output is a subset of its input.

        Args:
            output_dir: Directory for the generated files.
            prefix: Prefix for the generated files (default: derived from the
                input filename).

        Returns:
            The reconciling :class:`AssignStats` for the run.

        Raises:
            ValueError: If the first annotated read carries no anchor position
                tag, raised before any output file is opened.
        """
        log.info(f"Assigning target indices in {self.fastq.filename}...")

        prefix = prefix or get_prefix(self.fastq.filename)
        output_path = Path(output_dir)
        tgidx_fastq_path = output_path / f"{prefix}.r1_tgidx.fastq.gz"
        tgidx_stats_path = output_path / f"{prefix}.tgidx_stats.txt"
        tgidx_stats_mqc_path = output_path / f"{prefix}.tgidx_stats_mqc.json"
        tgidx_edit_distance_mqc_path = output_path / f"{prefix}.tgidx_edit_distance_mqc.json"
        tgidx_anchor_run_mqc_path = output_path / f"{prefix}.tgidx_anchor_run_mqc.json"

        counts = AssignCounts()

        # Validate the first read's header before opening either output file, so
        # a FASTQ that never went through extract-barcodes fails fast and leaves
        # no truncated outputs behind to be mistaken for a completed run. The read
        # is pushed back onto the iterator rather than consumed, since validating
        # it is not the same as processing it.
        reads = self.fastq.open_read_iterator(as_string=True)
        first_read = next(reads, None)
        if first_read is not None:
            self.validate_header(ReadAnnotation.parse(first_read[0]))
            reads = chain([first_read], reads)

        # The gzip stream must be entered BEFORE the executor. Pool workers are
        # forked on first submit and inherit the stream's pipe write end, so the
        # compressor only sees EOF once the pool is gone. `with` unwinds in
        # reverse, so the executor has to be innermost for its shutdown to run
        # first -- otherwise the stream close waits on a compressor the workers
        # keep alive, and the run hangs rather than fails.
        with (
            GzipFile(str(tgidx_fastq_path)).open_write_stream() as tgidx_stream,
            ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=init_assign_worker,
                initargs=(self,),
            ) as executor,
        ):
            # The driver submits its opening window as soon as it is called, and
            # that first submit is when the pool forks its workers. Calling it
            # before the progress bar starts its refresh thread keeps the fork
            # single-threaded -- forking a multi-threaded process can leave an
            # inherited lock held for ever in the child.
            results = map_batches_in_order(
                executor,
                assign_read_batch,
                iter_read_batches(reads, self.batch_size),
                self.n_workers,
            )

            with progress_bar(unit="reads") as pbar:
                # No total: the read count is not known without a second
                # decompress pass over the input, which costs more than it tells
                # the operator.
                task = pbar.add_task("Assigning target indices...", total=None)

                for assigned, batch_counts in results:
                    for header, seq, qual in assigned:
                        # The single write every read reaches. Each arrives
                        # exactly once because assign_read returns exactly one
                        # header per read, and in input order because the driver
                        # drains its batches in submission order rather than in
                        # the order the workers happened to finish them.
                        FastqFile.write_read(tgidx_stream, header, seq, qual)
                    counts.add(batch_counts)
                    pbar.update(task, advance=batch_counts.total)

        stats = counts.to_stats(self.anchor_base)

        with tgidx_stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        write_mqc_json(
            tgidx_stats_mqc_path,
            {
                "general_stats": stats.to_mqc_general_stats(prefix),
                "breakdown": stats.to_mqc_breakdown(prefix),
                "target_distribution": stats.to_mqc_target_distribution(prefix),
            },
        )
        edit_distance_payload = stats.to_mqc_edit_distance(prefix)
        if edit_distance_payload is not None:
            write_mqc_json(tgidx_edit_distance_mqc_path, edit_distance_payload)
        anchor_run_payload = stats.to_mqc_anchor_run(prefix)
        if anchor_run_payload is not None:
            write_mqc_json(tgidx_anchor_run_mqc_path, anchor_run_payload)

        log.info(f"Assigned target indices to {counts.matched}/{counts.total} reads")
        return stats


def init_assign_worker(assigner: "TargetAssigner") -> None:
    """Install the parent's assigner as this worker process's assigner.

    Run once per worker process by the pool, before that process is handed any
    batch. The parent's already-constructed assigner is what is handed over
    rather than the arguments to rebuild one from, so each process holds one
    k-mer index for the life of the pool instead of rebuilding it per batch.

    Args:
        assigner: The parent's assigner, inherited by this worker process.
    """
    global WORKER_ASSIGNER
    WORKER_ASSIGNER = assigner


def assign_read_batch(
    reads_batch: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], AssignCounts]:
    """Assign target indices to one batch of reads inside a worker process.

    Takes the batch as its only argument and reads its assigner off the module
    global the pool initializer filled in, so it stays the single-argument
    callable the in-order driver submits as it stands, and the k-mer index never
    travels in a batch's payload.

    Args:
        reads_batch: The batch's ``(header, seq, qual)`` reads, in input order.

    Returns:
        The batch's reads carrying their assigned headers, in input order, with
        sequence and quality untouched, and the tallies of the outcomes they
        took.
    """
    counts = AssignCounts()
    assigned = [
        (WORKER_ASSIGNER.assign_read(name, seq, counts), seq, qual)
        for name, seq, qual in reads_batch
    ]
    return assigned, counts


def iter_read_batches(
    reads: Iterable[tuple[str, ...]], batch_size: int
) -> Iterator[list[tuple[str, str, str]]]:
    """Lazily group an annotated read iterator into batches.

    Takes the iterator rather than opening the FASTQ itself, which is where this
    differs from ``BarcodeExtractor.iter_batches``: this stage pulls the first
    read off to validate its header and chains it back, so the reads to batch
    are already part-consumed and only the caller holds them.

    At most one batch of reads is pulled before that batch is yielded, so a
    caller that bounds its in-flight window bounds resident memory with it.

    Args:
        reads: Reads to group, each yielding at least ``(name, seq, qual)``;
            any further elements are dropped.
        batch_size: Number of reads a full batch carries.

    Yields:
        The next batch of ``(name, seq, qual)`` reads, in input order. The final
        batch is short when the reads do not divide exactly, and no empty batch
        is ever yielded.
    """
    batch: list[tuple[str, str, str]] = []

    for name, seq, qual, *_ in reads:
        batch.append((name, seq, qual))

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
