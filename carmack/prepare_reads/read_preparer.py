"""Per-read dispatch and the streaming driver for the prepare-reads stage.

For each UMI- and target-annotated R1 read, ``ReadPreparer.prepare_read`` decides which of
two arms the read belongs to and tallies the outcome into a shared ``PrepareCounts``
accumulator. A read carrying no real target index (no ``TGIDX`` tag, or one set to
``NO_TARGET``) is unmatched and stays untrimmed -- a later scRNA writer trims it off the
UMI span, not this stage. A read carrying a real target index is matched: its insert start
is computed by the same chemistry-agnostic ``insert_start`` arithmetic the scRNA arm uses,
just anchored off ``TGIDX_POS`` instead of ``UMI_POS``, and its scTIP header is rendered up
front so a later writer needs nothing but the outcome to write the read.

``ReadPreparer.prepare_reads`` is the streaming driver: it opens the paired input FASTQs,
pools ``prepare_read`` across a dynamic number of gzip writers -- three fixed files for the
scRNA arm plus one (R1, R2) pair per whitelisted target bucket -- and dispatches each
outcome to whichever writer its arm owns. R2 is never handed to a worker process: nothing
`prepare_read` computes depends on it, so shipping it through the pool would only pay
pickling and IPC cost for bytes the parent process already holds unchanged. Instead it is
threaded around the pool through a sidecar deque, kept aligned with the matching R1 batch by
the strict submission-order guarantee ``map_batches_in_order`` gives: a batch's R2 half is
pushed onto the sidecar the instant its R1 half is handed to the driver, and popped back off
only once that same batch's result has been drained, so the two halves can never come back
out of step even though only one of them ever crosses into a worker.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from carmack.assign_targets.target_assigner import NO_TARGET
from carmack.chemistry.annotation import parse_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.parallel import map_batches_in_order
from carmack.prepare_reads.insert_locator import insert_start
from carmack.prepare_reads.prepare_reporting import PrepareCounts, PrepareStats
from carmack.prepare_reads.scrna_writer import ScrnaWriter
from carmack.prepare_reads.sctip_writer import (
    SctipBucketWriters,
    render_sctip_header,
    write_sctip_read,
)
from carmack.utils import get_prefix, progress_bar

# Restated rather than imported: this stage's own saturation point has not been
# independently measured, so these are mirrored from assign-targets's measured values as
# a starting point for this stage, not a claim that they have been re-measured here.
MAX_READS_PER_BATCH = 2500
DEFAULT_MAX_WORKERS = 16

Read = tuple[str, str, str]
ReadPair = tuple[Read, Read]

# Set once per worker process by the pool initializer, then read-only. Holding the
# preparer here rather than binding it into each submitted batch is what keeps its
# resolved chemistry -- and everything derived from it -- out of the per-batch payload.
WORKER_PREPARER: "ReadPreparer | None" = None


@dataclass(frozen=True)
class UnmatchedOutcome:
    """The scRNA (unmatched) arm's per-read result.

    Carries the read's full, untrimmed sequence and quality: unlike the matched arm,
    trimming this read down to its cDNA insert needs the UMI span, which a later scRNA
    writer reads for itself rather than this stage recomputing it.

    Attributes:
        ann: The read's parsed header.
        r1_seq: The full, untrimmed R1 sequence.
        r1_qual: The full, untrimmed R1 quality string.
    """

    ann: ReadAnnotation
    r1_seq: str
    r1_qual: str


@dataclass(frozen=True)
class MatchedOutcome:
    """One scTIP target bucket arm's per-read result.

    Attributes:
        tgidx: The read's assigned target index value.
        header: The rendered scTIP read name, ready to write unchanged.
        r1_seq: The R1 sequence trimmed down to its insert.
        r1_qual: The R1 quality string trimmed down to its insert.
    """

    tgidx: str
    header: str
    r1_seq: str
    r1_qual: str


class ReadPreparer:
    """Dispatches one annotated R1 read at a time to the unmatched or a matched arm.

    Resolves and caches every chemistry-derived parameter dispatch needs once, at
    construction, so that no per-read call ever re-derives them.
    """

    def __init__(
        self,
        r1_fastq: str,
        r2_fastq: str,
        chemistry_name: str,
        n_workers: int = 1,
        batch_size: int | None = None,
    ) -> None:
        """Resolve the chemistry and derive the dispatch parameters this stage needs.

        Args:
            r1_fastq: Path to the UMI- and target-annotated R1 FASTQ.
            r2_fastq: Path to the paired R2 FASTQ.
            chemistry_name: Name of the chemistry describing the read layout.
            n_workers: Width of the process pool a later driver runs dispatch on.
            batch_size: Reads a submitted batch carries, defaulting to
                ``MAX_READS_PER_BATCH``.

        Raises:
            ValueError: If the chemistry is unknown, declares no UMI component anchored
                on its 5' side, or declares a target index whose whitelist loads no
                entries.
        """
        self.r1_fastq = FastqFile(r1_fastq)
        self.r2_fastq = FastqFile(r2_fastq)
        self.chemistry_name = chemistry_name
        self.n_workers = n_workers
        self.batch_size = batch_size or MAX_READS_PER_BATCH
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_umi_extraction():
            raise ValueError(
                f"chemistry '{chemistry_name}' declares no UMI component anchored on its "
                "5' side, so prepare-reads cannot locate the UMI it needs to dispatch reads"
            )

        if self.chemistry.supports_target_assignment():
            whitelist = self.chemistry.tgidx_whitelist()
            if not whitelist:
                raise ValueError(
                    f"chemistry '{chemistry_name}' loads no target index whitelist "
                    "entries, so no read could ever be validly dispatched to a matched arm"
                )
            tgidx = self.chemistry.tgidx_component()
            self.tgidx_key: str | None = tgidx.name
            self.tgidx_right_anchor: ReadComponent | None = self.chemistry.tgidx_right_anchor()
        else:
            self.tgidx_key = None
            self.tgidx_right_anchor = None

        umi = self.chemistry.umi_component()
        self.umi_name = umi.name
        self.barcode_names = [
            comp.name
            for comp in self.chemistry.read_structure.get_components_by_type(
                ReadComponentType.BARCODE
            )
        ]

        # Built once, here, rather than per read or per worker: everything it resolves
        # (the UMI span key, its right anchor, the barcode span keys) is fixed for the
        # lifetime of the chemistry, so a fresh writer per unmatched read would only
        # re-derive the same cached values on every call.
        self.scrna_writer = ScrnaWriter(self.chemistry)

    def prepare_read(
        self, name: str, seq: str, qual: str, counts: PrepareCounts
    ) -> UnmatchedOutcome | MatchedOutcome:
        """Dispatch one read to the unmatched or a matched arm and tally the outcome.

        A chemistry with no target index support never even looks for a ``TGIDX`` tag,
        since ``self.tgidx_key`` is ``None`` for it. Otherwise a read with no ``TGIDX``
        tag, or one set to ``NO_TARGET``, is unmatched; any other value is matched, and
        its insert start is computed off the end of its ``TGIDX_POS`` span.

        Args:
            name: The read's annotated header, without its leading ``@``.
            seq: The read's full sequence.
            qual: The read's full quality string.
            counts: Accumulator the outcome is tallied into, mutated in place so a run
                allocates no per-read outcome-tally object.

        Returns:
            The unmatched or matched outcome for this read.

        Raises:
            ValueError: If the read carries a real target index but no ``TGIDX_POS``
                span for it -- a corrupt input, since a correctly run assign-targets
                stage always writes a span alongside a real target value.
        """
        counts.total += 1
        ann = ReadAnnotation.parse(name)

        tgidx = ann.get(self.tgidx_key) if self.tgidx_key is not None else None
        if tgidx is None or tgidx == NO_TARGET:
            counts.unmatched += 1
            return UnmatchedOutcome(ann=ann, r1_seq=seq, r1_qual=qual)

        pos_key = position_key(self.tgidx_key)
        pos = ann.get(pos_key)
        if pos is None:
            raise ValueError(
                f"Read '{ann.read_id}' carries target index '{tgidx}' but no '{pos_key}' tag"
            )

        _, tgidx_end = parse_span(pos)
        cut = insert_start(reference=tgidx_end, anchor=self.tgidx_right_anchor, seq=seq)

        barcodes = {barcode_name: ann.get(barcode_name) for barcode_name in self.barcode_names}
        umi = ann.get(self.umi_name)
        header = render_sctip_header(self.chemistry, ann.read_id, barcodes, umi)

        counts.target_counts[tgidx] += 1
        return MatchedOutcome(tgidx=tgidx, header=header, r1_seq=seq[cut:], r1_qual=qual[cut:])

    def prepare_reads(self, output_dir: str = ".", prefix: str | None = None) -> PrepareStats:
        """Stream the paired reads, dispatch every one and write the output files.

        Every input read is written exactly once, to exactly one output arm: the scRNA
        arm's three files for an unmatched read, or one target bucket's (R1, R2) pair for
        a matched one. This stage never filters, so reads written always reconciles with
        reads read -- the invariant :class:`PrepareStats` states and checks.

        The output side opens a dynamic number of gzip writers: three fixed files plus
        one (R1, R2) pair per entry in the chemistry's target index whitelist, all of
        them held open for the whole run through a single :class:`~contextlib.ExitStack`
        rather than a fixed `with` tuple, since the whitelist's size is not known until
        the chemistry is resolved. Every one of those writers is entered before the
        :class:`~concurrent.futures.ProcessPoolExecutor` that follows them, with no
        exception: a `ProcessPoolExecutor` forks its workers on the first batch it is
        handed, and a forked worker inherits the write end of every compressor's input
        pipe that was already open at that point. A gzip writer entered AFTER the
        executor would therefore never see its pipe closed by the workers that inherited
        it, and closing it from the parent alone waits forever for a pipe some other
        process still holds open. `ExitStack` unwinds in the reverse of entry order, so
        entering the executor last is what makes it the first thing torn down -- releasing
        every inherited pipe end before any writer is asked to finish -- on every exit
        path, including the exception path a mismatched R1/R2 pair takes partway through
        a run.

        R2 never crosses into a worker process: `prepare_read` computes everything a
        writer needs from R1 alone, so R2 is threaded around the pool instead, held in a
        sidecar deque that `r1_batches_with_r2_sidecar` keeps aligned with the R1 batches
        actually submitted. `map_batches_in_order` yields a batch's outcome only once it
        has drained that batch's result, in the order the batches were submitted, so the
        sidecar's next entry is always the R2 half belonging to the outcome just yielded.

        Args:
            output_dir: Directory the generated files are written into.
            prefix: Prefix for the generated files, defaulting to the R1 input's own
                prefix.

        Returns:
            The reconciling :class:`PrepareStats` for the run.

        Raises:
            ValueError: If the first pulled pair's read ids disagree, or if one stream
                is exhausted before the other -- raised before any output file is
                opened, so a mismatched input leaves no truncated output behind.
        """
        prefix = prefix or get_prefix(self.r1_fastq.filename)

        r1_reads = self.r1_fastq.open_read_iterator(as_string=True)
        r2_reads = self.r2_fastq.open_read_iterator(as_string=True)
        pairs = iter_paired_reads(r1_reads, r2_reads)

        # Pulled and validated before anything is opened, so a mismatch on the very
        # first pair -- the one place `next()` can raise here -- leaves no executor, no
        # writer and no partial output file behind for the caller to clean up.
        first_pair = next(pairs, None)
        if first_pair is not None:
            pairs = chain([first_pair], pairs)

        output_path = Path(output_dir)
        none_r1_path = output_path / f"{prefix}.none.r1.fastq.gz"
        none_r2_path = output_path / f"{prefix}.none.r2.fastq.gz"
        none_barcodes_path = output_path / f"{prefix}.none.barcodes.fastq.gz"

        with ExitStack() as stack:
            none_r1_stream = stack.enter_context(GzipFile(str(none_r1_path)).open_write_stream())
            none_r2_stream = stack.enter_context(GzipFile(str(none_r2_path)).open_write_stream())
            none_barcodes_stream = stack.enter_context(
                GzipFile(str(none_barcodes_path)).open_write_stream()
            )

            # One (R1, R2) writer pair per whitelisted target, opened here -- before the
            # executor below -- one whitelist entry at a time, so every bucket a
            # matched read could ever be dispatched to is already open by the time the
            # first batch is submitted. Empty for a chemistry with no target index
            # support, since `tgidx_whitelist()` itself resolves to `()` for one.
            bucket_writers: dict[str, SctipBucketWriters] = {}
            for tgidx_value in self.chemistry.tgidx_whitelist():
                bucket_r1_stream = stack.enter_context(
                    GzipFile(
                        str(output_path / f"{prefix}.{tgidx_value}.r1.fastq.gz")
                    ).open_write_stream()
                )
                bucket_r2_stream = stack.enter_context(
                    GzipFile(
                        str(output_path / f"{prefix}.{tgidx_value}.r2.fastq.gz")
                    ).open_write_stream()
                )
                bucket_writers[tgidx_value] = SctipBucketWriters(
                    r1=bucket_r1_stream, r2=bucket_r2_stream
                )

            # Entered LAST, after every writer above: see the writer-before-executor
            # ordering explained above the `with` block.
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=self.n_workers,
                    initializer=init_prepare_worker,
                    initargs=(self,),
                )
            )

            r2_sidecar: deque = deque()
            results = map_batches_in_order(
                executor,
                prepare_read_batch,
                r1_batches_with_r2_sidecar(pairs, self.batch_size, r2_sidecar),
                self.n_workers,
            )

            counts = PrepareCounts()
            with progress_bar(unit="reads") as pbar:
                # No total: the read count is not known without a second decompress
                # pass over the input, which costs more than it tells the operator.
                task = pbar.add_task("Preparing reads...", total=None)

                for outcomes, batch_counts in results:
                    r2_batch = r2_sidecar.popleft()
                    for outcome, r2_read in zip(outcomes, r2_batch, strict=True):
                        if isinstance(outcome, UnmatchedOutcome):
                            r2_name, r2_seq, r2_qual = r2_read
                            self.scrna_writer.write_read(
                                outcome.ann,
                                outcome.r1_seq,
                                outcome.r1_qual,
                                r2_name,
                                r2_seq,
                                r2_qual,
                                none_r1_stream,
                                none_r2_stream,
                                none_barcodes_stream,
                            )
                        else:
                            _, r2_seq, r2_qual = r2_read
                            write_sctip_read(
                                bucket_writers,
                                outcome.tgidx,
                                outcome.header,
                                outcome.r1_seq,
                                outcome.r1_qual,
                                r2_seq,
                                r2_qual,
                            )
                    counts.add(batch_counts)
                    pbar.update(task, advance=batch_counts.total)

        stats = counts.to_stats()

        with (output_path / f"{prefix}.prepare_stats.txt").open("w") as report_file:
            report_file.write(stats.get_report())

        return stats


def init_prepare_worker(preparer: "ReadPreparer") -> None:
    """Install the parent's preparer as this worker process's preparer.

    Run once per worker process by the pool, before that process is handed any batch.
    The parent's already-constructed preparer is what is handed over rather than the
    arguments to rebuild one from, so each worker process resolves its chemistry once,
    for the life of the pool, instead of once per batch.

    Args:
        preparer: The parent's preparer, inherited by this worker process.
    """
    global WORKER_PREPARER
    WORKER_PREPARER = preparer


def prepare_read_batch(
    r1_batch: list[tuple[str, str, str]],
) -> tuple[list["UnmatchedOutcome | MatchedOutcome"], PrepareCounts]:
    """Dispatch one batch of R1 reads inside a worker process.

    Takes the batch as its only argument and reads its preparer off the module global
    the pool initializer filled in, so it stays the single-argument callable the
    in-order driver submits as it stands. Only R1 is taken: `prepare_read` never reads
    R2, so the worker never needs it and it never has to be pickled across the pool
    boundary.

    Args:
        r1_batch: The batch's ``(name, seq, qual)`` R1 reads, in input order.

    Returns:
        The batch's outcomes, in input order, and the tallies of the outcomes they
        took.
    """
    counts = PrepareCounts()
    outcomes = [
        WORKER_PREPARER.prepare_read(name, seq, qual, counts) for name, seq, qual in r1_batch
    ]
    return outcomes, counts


def iter_paired_reads(r1_reads: Iterable[Read], r2_reads: Iterable[Read]) -> Iterator[ReadPair]:
    """Lazily zip an R1 read stream with its R2 stream, validating every pulled pair.

    At most one read is pulled from each stream before its pair is validated and
    yielded, so a mismatch anywhere in the streams is raised at exactly that point,
    with every pair already yielded left unaffected and nothing beyond the mismatch
    ever pulled from either stream.

    Args:
        r1_reads: R1 reads, each yielding at least ``(name, seq, qual)``.
        r2_reads: R2 reads, each yielding at least ``(name, seq, qual)``.

    Yields:
        The next ``(r1_read, r2_read)`` pair, in stream order.

    Raises:
        ValueError: If a pulled pair's read ids disagree, or if one stream is
            exhausted before the other.
    """
    r1_iter = iter(r1_reads)
    r2_iter = iter(r2_reads)
    sentinel = object()

    while True:
        r1_read = next(r1_iter, sentinel)
        r2_read = next(r2_iter, sentinel)

        if r1_read is sentinel and r2_read is sentinel:
            return
        if r1_read is sentinel or r2_read is sentinel:
            raise ValueError("R1 and R2 streams do not carry the same number of reads")

        r1_id = ReadAnnotation.parse(r1_read[0]).read_id
        r2_id = ReadAnnotation.parse(r2_read[0]).read_id
        if r1_id != r2_id:
            raise ValueError(f"R1 read '{r1_id}' does not pair with R2 read '{r2_id}'")

        yield r1_read, r2_read


def iter_read_pair_batches(
    pairs: Iterable[ReadPair], batch_size: int
) -> Iterator[tuple[list[Read], list[Read]]]:
    """Lazily group a paired-read stream into same-length ``(r1_batch, r2_batch)`` batches.

    At most one batch of pairs is pulled before that batch is yielded, so a caller that
    bounds its in-flight window bounds resident memory with it.

    Args:
        pairs: ``(r1_read, r2_read)`` pairs to group, such as ``iter_paired_reads``
            yields.
        batch_size: Number of pairs a full batch carries.

    Yields:
        The next ``(r1_batch, r2_batch)`` pair of same-length batches, in input order.
        The final batch is short when the pairs do not divide exactly, and no empty
        batch is ever yielded.
    """
    r1_batch: list[Read] = []
    r2_batch: list[Read] = []

    for r1_read, r2_read in pairs:
        r1_batch.append(r1_read)
        r2_batch.append(r2_read)

        if len(r1_batch) >= batch_size:
            yield r1_batch, r2_batch
            r1_batch = []
            r2_batch = []

    if r1_batch:
        yield r1_batch, r2_batch


def r1_batches_with_r2_sidecar(
    pairs: Iterable[ReadPair],
    batch_size: int,
    r2_sidecar: "deque[list[Read]]",
) -> Iterator[list[Read]]:
    """Yield each batch's R1 half while stashing its R2 half in a shared sidecar.

    This is the seam that keeps R2 out of the process pool without losing track of it.
    `prepare_reads` hands only the R1 half of each batch to `map_batches_in_order`, so
    only R1 is ever pickled to a worker; the R2 half a batch's outcomes still need for
    writing is appended to `r2_sidecar` here, strictly before this generator yields that
    batch's R1 half, so a caller driving both this generator and the in-order results it
    feeds can always assume a batch's sidecar entry is already waiting by the time that
    batch's result comes back to be drained.

    Args:
        pairs: ``(r1_read, r2_read)`` pairs to group and split, such as
            ``iter_paired_reads`` yields.
        batch_size: Number of pairs a full batch carries.
        r2_sidecar: Shared queue this generator appends each batch's R2 half onto, in
            the same order its R1 half is yielded in. Owned by the caller, which is
            responsible for popping an entry off once it is done with it.

    Yields:
        The next batch's R1 half alone, in input order.
    """
    for r1_batch, r2_batch in iter_read_pair_batches(pairs, batch_size):
        r2_sidecar.append(r2_batch)
        yield r1_batch
