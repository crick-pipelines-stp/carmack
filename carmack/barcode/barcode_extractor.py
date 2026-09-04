import logging
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt

from carmack.barcode.barcode_utils import make_barcode_rank_plot
from carmack.barcode.extraction_dataclasses import MatchMethod, ReadMatchResult
from carmack.barcode.extraction_reporting import ExtractionStatsAccumulator
from carmack.barcode.hybrid_extractor import HybridExtractor
from carmack.barcode.matchers.alignment_matcher import AlignmentMatcher
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher, MatcherBase
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.annotation import format_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponentType
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.utils import get_prefix, progress_bar

log = logging.getLogger(__name__)

MIN_READS_PER_BATCH = 10
MAX_READS_PER_BATCH = 2500


class BarcodeExtractor:
    """
    Main barcode extraction pipeline.

    Processes FASTQ files using parallel workers and generates output files.
    """

    def __init__(
        self,
        fastq_file: str,
        chemistry_name: str,
        kmer_size: int = 4,
        n_workers: int = 1,
        batch_size: int | None = None,
        fast: bool = False,
    ) -> None:
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.kmer_size = kmer_size
        self.n_workers = n_workers
        self.fast = fast

        # Check reads
        self.total_reads = self.fastq.reads_count
        if self.total_reads == 0:
            raise ValueError(f"No reads found in FASTQ file: {fastq_file}")

        # Determine ideal batch size based on total reads and number of workers, but allow override
        self.batch_size = batch_size or self.calc_batch_size()

        # Get chemistry
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        self.whitelists = self.chemistry.barcode_whitelists

        # Init matchers
        self.matchers: dict[MatchMethod, Mapping[str, MatcherBase]] = self.init_matchers()

        log.debug(f"BarcodeExtractor initialized for {fastq_file}")
        log.debug(
            f"Parameters: chemistry={chemistry_name}, kmer_size={self.kmer_size}, batch_size={self.batch_size}, workers={self.n_workers}, fast={self.fast}"
        )

    def calc_batch_size(self) -> int:
        """
        Determine the batch size to use for processing.

        If a batch size was provided during initialization, use that. Otherwise, calculate a default
        batch size based on the number of workers and the total number of reads in the FASTQ file.
        """
        batch_size = max(MIN_READS_PER_BATCH, ceil(self.total_reads / self.n_workers))
        batch_size = min(batch_size, MAX_READS_PER_BATCH)
        log.debug(f"Calculated default batch size: {batch_size} (total reads: {self.total_reads})")
        return batch_size

    def init_matchers(self) -> dict[MatchMethod, Mapping[str, MatcherBase]]:
        """
        Initialize the barcode matchers based on the chemistry definition.

        This sets up the necessary matchers for each barcode component according to the read structure
        defined by the chemistry. Matchers are stored in a dictionary for easy access during extraction.
        """

        # {bc_name: Matcher}
        fixed_matchers: dict[str, FixedPositionMatcher] = {}
        kmer_matchers: dict[str, KmerMatcher] = {}
        alignment_matchers: dict[str, AlignmentMatcher] = {}

        for component in self.chemistry.read_structure.components:
            if component.type is not ReadComponentType.BARCODE:
                continue

            common_kwargs = {
                "component": component,
                "whitelist": self.whitelists[component.name],
                "chemistry": self.chemistry,
            }

            fixed_matchers[component.name] = FixedPositionMatcher(**common_kwargs)
            kmer_matchers[component.name] = KmerMatcher(
                **common_kwargs,
                k=self.kmer_size,
                max_errors=self.chemistry.max_errors.barcode,
            )
            if not self.fast:
                alignment_matchers[component.name] = AlignmentMatcher(
                    **common_kwargs,
                    max_errors=self.chemistry.max_errors.barcode,
                )

        # The order of matchers is important
        # We want to try the fastest methods first to reduce search space for slower methods
        matchers = {
            MatchMethod.EXACTMATCH: fixed_matchers,
            MatchMethod.KMERMATCH: kmer_matchers,
        }

        if not self.fast:
            matchers[MatchMethod.ALIGNMATCH] = alignment_matchers

        return matchers

    def iter_batches(self) -> Iterator[list[tuple[str, str, str]]]:
        """Lazily yield read batches from the FASTQ file.

        Streams reads so the full FASTQ is never held in memory at once.
        """
        current_batch: list[tuple[str, str, str]] = []

        for name, seq, qual, *_ in self.fastq.open_read_iterator(as_string=True):
            current_batch.append((name, seq, qual))

            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch

    def generate_batches(self) -> list[list[tuple[str, str, str]]]:
        """Materialise all read batches into a list. Prefer iter_batches() for streaming."""
        return list(self.iter_batches())

    def extract_barcodes(
        self,
        output_dir: str = ".",
        prefix: str | None = None,
    ) -> None:
        """
        Run the full extraction pipeline.

        Args:
            output_dir: Output directory for result files
            prefix: Prefix for output files (default: derived from input filename)
        """

        log.info("Starting barcode extraction pipeline...")

        prefix = prefix or get_prefix(self.fastq.filename)

        # Prepare output file paths
        output_path = Path(output_dir)
        bc_all_path = output_path / f"{prefix}.bc_all.txt.gz"
        bc_valid_path = output_path / f"{prefix}.bc_valid.txt.gz"
        bc_annotated_path = output_path / f"{prefix}.r1_annotated.fastq.gz"
        bc_counts_path = output_path / f"{prefix}.bc_counts.csv"
        bc_rank_plot_path = output_path / f"{prefix}.bc_rank.png"
        bc_stats_path = output_path / f"{prefix}.bc_stats.txt"

        log.debug(
            f"Output paths: {bc_all_path}, {bc_valid_path}, {bc_annotated_path}, {bc_counts_path}, {bc_rank_plot_path}, {bc_stats_path}"
        )

        stats_acc = ExtractionStatsAccumulator(self.matchers)

        # The gzip streams must be entered BEFORE the executor. Pool workers are
        # forked on first submit and inherit the streams' pipe write ends, so
        # `gzip` only sees EOF once the pool is gone. `with` unwinds in reverse,
        # so the executor has to be innermost for its shutdown to run first --
        # otherwise the stream close waits on a `gzip` the workers keep alive.
        with (
            GzipFile(str(bc_all_path)).open_write_stream() as bc_all_f,
            GzipFile(str(bc_valid_path)).open_write_stream() as bc_valid_f,
            GzipFile(str(bc_annotated_path)).open_write_stream() as bc_annotated_f,
            ProcessPoolExecutor(max_workers=self.n_workers) as executor,
        ):
            hybrid_extractor = HybridExtractor(chemistry=self.chemistry, matchers=self.matchers)

            # Bounded in-flight window keeps memory usage proportional to
            # max_in_flight * batch_size rather than the full FASTQ. Results
            # are folded into the accumulator and written to disk as they
            # arrive, so no list of ReadMatchResults is retained.
            batch_iter = self.iter_batches()
            max_in_flight = max(self.n_workers * 2, 2)
            log.debug(
                f"Streaming batches with batch size {self.batch_size}, max in-flight {max_in_flight}"
            )

            futures = {
                executor.submit(process_read_batch, batch, hybrid_extractor=hybrid_extractor)
                for batch in islice(batch_iter, max_in_flight)
            }

            with progress_bar(unit="reads") as pbar:
                task = pbar.add_task("Extracting barcodes...", total=self.fastq.reads_count)

                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_results, batch_len = future.result()
                        for r in batch_results:
                            annotated = r.get_annotated_readname()
                            GzipFile.write_string(bc_all_f, f"{annotated}\n")
                            if r.success and r.full_barcode is not None:
                                GzipFile.write_string(bc_valid_f, f"{annotated}\n")
                                ann = ReadAnnotation.parse(r.read_name)
                                for bc in r.bc_results:
                                    attempt = bc.attempts[-1]
                                    ann.set(bc.bc_name, attempt.match)
                                    start, end = attempt.read_idx
                                    ann.set(position_key(bc.bc_name), format_span(start, end))
                                FastqFile.write_read(bc_annotated_f, ann.render(), r.read, r.qual)
                            stats_acc.update(r)
                        pbar.update(task, advance=batch_len)

                    for batch in islice(batch_iter, len(done)):
                        futures.add(
                            executor.submit(
                                process_read_batch, batch, hybrid_extractor=hybrid_extractor
                            )
                        )

        ex_stats = stats_acc.finalize()

        with progress_bar(unit="files") as pbar:
            task = pbar.add_task("Writing summary files...", total=3)

            with bc_counts_path.open("w") as f:
                for bc, count in stats_acc.full_barcode_counts.most_common():
                    f.write(f"{bc},{count}\n")
            pbar.update(task, advance=1)

            fig = make_barcode_rank_plot(stats_acc.full_barcode_counts)
            fig.savefig(bc_rank_plot_path)
            plt.close(fig)
            pbar.update(task, advance=1)

            with bc_stats_path.open("w") as f:
                f.write(ex_stats.get_report())
            pbar.update(task, advance=1)

        log.info(f"Completed barcode extraction for {ex_stats.overall.total_reads} reads")
        log.info(
            f"Overall success rate: {(ex_stats.overall.perfect + ex_stats.overall.corrok) / ex_stats.overall.total_reads:.2%}"
        )
        log.info(
            f"Perfect matches: {ex_stats.overall.perfect} ({ex_stats.overall.perfect / ex_stats.overall.total_reads:.2%})"
        )
        log.info(
            f"Corrected matches: {ex_stats.overall.corrok} ({ex_stats.overall.corrok / ex_stats.overall.total_reads:.2%})"
        )


# Module-level function for processing a batch of reads, used for multiprocessing
def process_read_batch(
    reads_batch, hybrid_extractor: HybridExtractor
) -> tuple[list[ReadMatchResult], int]:
    """
    Process a batch of reads to extract barcodes.

    This function is designed to be run in parallel across multiple processes. It takes a batch of reads,
    applies the defined matchers, and returns the results for each read.
    """
    results = []
    batch_len = len(reads_batch)

    for read_name, read_seq, qual in reads_batch:
        results.append(hybrid_extractor.process_read(read_name, read_seq, qual))

    return results, batch_len
