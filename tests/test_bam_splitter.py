import csv
import fnmatch
import os
from typing import Dict

import pysam
import pytest

from carmack.split_reads.split_reads import BamSplitter
from carmack.utils import get_prefix


BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1_dup.dedup.tagged.bam"

BAI_PATH = BAM_PATH + ".bai"
PREFIX = get_prefix(BAM_PATH)
UNSORTED_BAM_SUFFIX = "split.bam"
SORTED_BAM_SUFFIX = "split.sorted.bam"
SORTED_BAI_SUFFIX = "split.sorted.bam.bai"
CSV_FILE = PREFIX + "_split_counts.csv"


class TestBamSplitter:
    @pytest.fixture(scope="class")
    def shared_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("split-reads")

    @pytest.fixture(scope="class")
    def barcodes(self) -> Dict[str, int]:
        """
        Returns a dictionary of barcodes and their counts computed from the BAM file.
        """
        barcodes_counter = {}
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as bam:
            for read in bam:
                barcode = read.get_tag("BC")
                if barcode not in barcodes_counter:
                    barcodes_counter[barcode] = 0
                barcodes_counter[barcode] += 1
        return barcodes_counter

    def get_filename(self, path, suffix, barcode, shared_path) -> str:
        """
        Get filename that contains the barcode and ends with the suffix.
        """
        pattern = f"*{barcode}*.{suffix}"
        file_match = fnmatch.filter(os.listdir(path), pattern)
        assert len(file_match) != 0
        return os.path.join(shared_path, file_match[0])

    def test_output_file_creation(self, shared_path, barcodes):
        # Initialize BamSplitter
        splitter = BamSplitter(BAM_PATH, BAI_PATH)

        # Work
        splitter.split(shared_path, prefix=PREFIX, cpu_count=1)

        # Check files count
        # For each barcode, expect 3 files: unsorted bam, sorted bam, sorted bai + one CSV file
        files = len(os.listdir(shared_path))
        assert files == len(barcodes) * 3 + 1

    def test_alignment_files(self, shared_path, barcodes):
        """
        Check if output alignment files contain the correct number of reads and barcodes.
        """
        for barcode in barcodes.keys():
            for suffix in (UNSORTED_BAM_SUFFIX, SORTED_BAM_SUFFIX):
                file = self.get_filename(shared_path, suffix, barcode, shared_path)
                with pysam.AlignmentFile(file, "rb") as bam:
                    read_counter = 0
                    for read in bam:
                        assert read.get_tag("BC") == barcode  # Check if barcodes match
                        read_counter += 1
                    # Check if number of reads match
                    assert read_counter == barcodes[barcode]  # bam.count() requires index file

    def test_csv_file(self, shared_path, barcodes):
        """
        Check if the CSV file contains the correct barcodes and their counts.
        """
        csv_file_path = os.path.join(shared_path, CSV_FILE)
        with open(csv_file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            assert header == ["barcode", "count"]  # Check header
            for row in csv_reader:  # Check each barcode and count
                barcode, count = row
                assert barcode in barcodes
                assert int(count) == barcodes[barcode]
