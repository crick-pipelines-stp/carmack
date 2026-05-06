import logging
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt

from carmack.barcode.barcode_utils import make_barcode_rank_plot
from carmack.barcode.extraction_dataclasses import MatchMethod, ReadMatchResult
from carmack.barcode.extraction_reporting import ExtractionStats
from carmack.barcode.hybrid_extractor import HybridExtractor
from carmack.barcode.matchers.alignment_matcher import AlignmentMatcher
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher, MatcherBase
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponentType
from carmack.io.fastq_file import FastqFile
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
                "barcode_component": component,
                "whitelist": self.whitelists[component.name],
                "chemistry": self.chemistry,
            }

            fixed_matchers[component.name] = FixedPositionMatcher(**common_kwargs)
            kmer_matchers[component.name] = KmerMatcher(
                **common_kwargs,
                k=self.kmer_size,
            )
            if not self.fast:
                alignment_matchers[component.name] = AlignmentMatcher(**common_kwargs)

        # The order of matchers is important
        # We want to try the fastest methods first to reduce search space for slower methods
        matchers = {
            MatchMethod.EXACTMATCH: fixed_matchers,
            MatchMethod.KMERMATCH: kmer_matchers,
        }

        if not self.fast:
            matchers[MatchMethod.ALIGNMATCH] = alignment_matchers

        return matchers

    def generate_batches(self) -> list[list[tuple[str, str, str]]]:
        """Generate read batches from the FASTQ file. Used for multiprocessing."""
        batches = []
        current_batch: list[tuple[str, str, str]] = []

        for name, seq, qual, *_ in self.fastq.open_read_iterator(as_string=True):
            current_batch.append((name, seq, qual))

            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

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
        bc_all_path = output_path / f"{prefix}.bc_all.txt"
        bc_valid_path = output_path / f"{prefix}.bc_valid.txt"
        bc_counts_path = output_path / f"{prefix}.bc_counts.csv"
        bc_rank_plot_path = output_path / f"{prefix}.bc_rank.png"
        bc_stats_path = output_path / f"{prefix}.bc_stats.txt"

        log.debug(
            f"Output paths: {bc_all_path}, {bc_valid_path}, {bc_counts_path}, {bc_rank_plot_path}, {bc_stats_path}"
        )

        results: list[ReadMatchResult] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            read_batches = self.generate_batches()
            log.debug(
                f"Generated {len(read_batches)} read batches with batch size {self.batch_size} for processing"
            )

            hybrid_extractor = HybridExtractor(chemistry=self.chemistry, matchers=self.matchers)

            futures = [
                executor.submit(process_read_batch, batch, hybrid_extractor=hybrid_extractor)
                for batch in read_batches
            ]

            with progress_bar(unit="reads") as pbar:
                task = pbar.add_task("Extracting barcodes...", total=self.fastq.reads_count)

                for future in as_completed(futures):
                    batch_results, batch_len = future.result()
                    results.extend(batch_results)
                    pbar.update(task, advance=batch_len)

        # Write output files
        with progress_bar(unit="files") as pbar:
            task = pbar.add_task("Writing output files...", total=5)

            self.write_bc_all(bc_all_path, results)
            pbar.update(task, advance=1)

            self.write_bc_valid(bc_valid_path, results)
            pbar.update(task, advance=1)

            self.write_bc_counts(bc_counts_path, results)
            pbar.update(task, advance=1)

            self.write_bc_rank_plot(bc_rank_plot_path, results)
            pbar.update(task, advance=1)

            self.write_bc_stats(bc_stats_path, results)
            pbar.update(task, advance=1)

    def get_barcode_counts(self, results: list[ReadMatchResult]) -> Counter[str]:
        """Count successful full barcodes across all reads."""
        return Counter(r.full_barcode for r in results if r.success and r.full_barcode is not None)

    def write_bc_all(self, bc_all_path: Path, results: list[ReadMatchResult]) -> None:
        """Write the full barcode extraction results for all reads to a TXT file."""
        with bc_all_path.open("w") as f:
            for r in results:
                f.write(f"{r.get_annotated_readname()}\n")

    def write_bc_valid(self, bc_valid_path: Path, results: list[ReadMatchResult]) -> None:
        """Write the valid barcodes (those that matched the whitelist) to a TXT file."""
        with bc_valid_path.open("w") as f:
            for r in results:
                if r.success and r.full_barcode is not None:
                    f.write(f"{r.get_annotated_readname()}\n")

    def write_bc_counts(self, bc_counts_path: Path, results: list[ReadMatchResult]) -> None:
        """Write the counts of each unique full barcode to a CSV file."""
        barcode_counts = self.get_barcode_counts(results)

        with bc_counts_path.open("w") as f:
            for bc, count in barcode_counts.most_common():
                f.write(f"{bc},{count}\n")

    def write_bc_rank_plot(self, bc_rank_plot_path: Path, results: list[ReadMatchResult]) -> None:
        """Write a barcode-rank plot of successful full-barcode counts to a PNG file."""
        fig = make_barcode_rank_plot(self.get_barcode_counts(results))
        fig.savefig(bc_rank_plot_path)
        plt.close(fig)

    def write_bc_stats(
        self, bc_stats_path: Path, results: list[ReadMatchResult], log_stats: bool = True
    ) -> None:
        """Write summary statistics of the barcode extraction results to a TXT file."""
        ex_stats: ExtractionStats = ExtractionStats.from_results(results, self.matchers)
        stats_out: str = ex_stats.get_report()

        with bc_stats_path.open("w") as f:
            f.write(stats_out)

        if log_stats:
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
