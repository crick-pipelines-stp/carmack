import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Optional, Set

import pysam

from carmack.utils import get_prefix, progress_bar


log = logging.getLogger(__name__)


class BamSplitter:
    """
    Class that splits a BAM file into two separate files based on barcode (BC) tag value.
    """

    def __init__(self, bam: str, bai: Optional[str]) -> None:
        self.bam = bam
        self.bai = bai

        log.debug(f"BamSplitter object created with BAM: {self.bam} and BAI: {self.bai}")

    def get_barcode_tag(self, read: pysam.AlignedSegment) -> str:
        """
        Get the barcode tag value from a read.
        """
        if not read.has_tag("BC"):
            raise ValueError("Read does not have a barcode tag (BC).")

        barcode = read.get_tag("BC")

        if len(barcode) == 0:
            raise ValueError("Barcode tag (BC) is empty.")

        return barcode

    def split_bam(
        self, stack: ExitStack, tagged_bam: pysam.AlignmentFile, output_dir: str, prefix: str
    ) -> Set[str]:
        """
        Segregate reads based on barcode tag value and save to separate files.
        Additionally, write the barcode counts to a CSV file.

        Returns set of split filenames.
        """
        log.info("Splitting reads in BAM file.")

        SPLIT_BAM_SUFFIX = "split.bam"
        COUNT_CSV_PATH = os.path.join(output_dir, f"{prefix}_split_counts.csv")

        file_handles = {}
        barcode_counter = {}
        split_files = set()

        total_reads = tagged_bam.count()
        with progress_bar(unit="reads") as pbar:
            task = pbar.add_task("Splitting reads", total=total_reads)
            for read in tagged_bam.fetch():
                barcode = self.get_barcode_tag(read)

                # Create a new file handle for the barcode
                if barcode not in file_handles.keys():
                    split_file = os.path.join(output_dir, f"{prefix}_{barcode}.{SPLIT_BAM_SUFFIX}")
                    split_files.add(split_file)
                    file_handles[barcode] = stack.enter_context(
                        pysam.AlignmentFile(
                            split_file,
                            "wb",
                            header=tagged_bam.header,
                        )
                    )
                    barcode_counter[barcode] = 0

                    log.debug(f"File and count entry created for barcode: {barcode}")

                # Write the read to the corresponding barcode file and increment the counter
                file_handles[barcode].write(read)
                barcode_counter[barcode] += 1
                pbar.advance(task)

        log.info(
            f"Finished splitting reads in BAM file. Total files created = {len(file_handles)}"
        )

        # Sort the barcode counts in descending order
        barcode_counter = dict(sorted(barcode_counter.items(), key=lambda x: x[1], reverse=True))

        # Write the barcode counts to a CSV file
        count_csv = stack.enter_context(open(COUNT_CSV_PATH, "w", newline=""))
        count_csv_writer = csv.writer(count_csv)

        count_csv_writer.writerow(["barcode", "count"])
        for barcode, count in barcode_counter.items():
            count_csv_writer.writerow([barcode, count])

        log.info(f"Barcode counts written to {COUNT_CSV_PATH}")

        return split_files

    def split(self, output_dir: str, prefix: Optional[str] = None, cpu_count: int = 1) -> None:
        """
        Split barcode-tagged BAM file into separate files based on barcode tag (BC) value.

        The output files are saved to the output directory with corresponding sorted BAM and index
        BAI files.
        Each file is named according to the barcode tag (BC) value.
        """
        # Set prefix, if non specified
        if prefix is None:
            prefix = get_prefix(self.bam)

        # Split BAM
        with ExitStack() as stack:
            bam = stack.enter_context(pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai))
            split_files = self.split_bam(stack, bam, output_dir, prefix)

        # Sort and index split BAM files
        log.info(f"Sorting and indexing split BAM files (using {cpu_count} threads).")
        SORTED_BAM_SUFFIX = "split.sorted.bam"

        def sort_index(split_file: str, logger: logging.Logger = log) -> None:
            log.debug(f"Sorting and indexing split file: {split_file}")
            sorted_file = os.path.join(output_dir, f"{get_prefix(split_file)}.{SORTED_BAM_SUFFIX}")
            pysam.sort("-o", sorted_file, split_file)
            pysam.index(sorted_file)

        # Multi-threaded sorting and indexing
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            with progress_bar(unit="files") as pbar:
                task = pbar.add_task("Sorting and indexing", total=len(split_files))
                for _ in executor.map(sort_index, split_files):
                    pbar.advance(task)

        log.info("Finished sorting and indexing split BAM files.")
