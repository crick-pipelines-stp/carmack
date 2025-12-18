import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

import pysam

from carmack.utils import get_prefix, progress_bar


log = logging.getLogger(__name__)


class BamSplitter:
    """
    Class that splits a BAM file into two separate files based on barcode (BC) tag value.
    """

    def __init__(self, bam: str, bai: str | None) -> None:
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
    ) -> set[str]:
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

        with progress_bar(unit="reads") as progress:
            task = progress.add_task("Splitting reads", total=tagged_bam.count())
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
                progress.update(task, advance=1)

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

    def split(self, output_dir: str, prefix: str | None = None, cpu_count: int = 1) -> None:
        """
        Split barcode-tagged BAM file into separate files based on barcode tag (BC) value.

        The output files are saved to the output directory with corresponding sorted BAM and index
        BAI files.
        Each file is named according to the barcode tag (BC) value.
        """
        # Set prefix, if none specified
        prefix = prefix or get_prefix(self.bam)

        # Split BAM
        with ExitStack() as stack:
            bam = stack.enter_context(pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai))
            split_files = self.split_bam(stack, bam, output_dir, prefix)

        # Sort and index split BAM files
        log.info(f"Sorting and indexing split BAM files (using {cpu_count} threads).")
        SORTED_BAM_SUFFIX = "split.sorted.bam"

        def sort_index(split_file: str, logger: logging.Logger = log) -> None:
            logger.debug(f"Sorting and indexing split file: {split_file}")
            sorted_file = os.path.join(output_dir, f"{get_prefix(split_file)}.{SORTED_BAM_SUFFIX}")
            pysam.sort("-o", sorted_file, split_file)
            pysam.index(sorted_file)

        # Multi-threaded sorting and indexing
        with (
            ThreadPoolExecutor(max_workers=cpu_count) as executor,
            progress_bar(unit="files") as progress,
        ):
            task = progress.add_task("Sorting and indexing BAM files", total=len(split_files))
            futures = [executor.submit(sort_index, split_file, log) for split_file in split_files]

            for _ in as_completed(futures):
                progress.update(task, advance=1)

        log.info("Finished sorting and indexing split BAM files.")
