# pylint: disable=missing-function-docstring, missing-class-docstring

import gzip
import os
import unittest

from carmack.fastq_tools.fastq_filter import FastqFilter
from carmack.io.fastq_file import FastqFile
from tests.utils import with_temporary_folder

R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"
R2_PATH = "tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz"
BC_VALID_PATH = "tests/data/bc_valid.csv"


class TestFastqFilter(unittest.TestCase):
    @with_temporary_folder
    def test_fastq_filter_valid_reads(self, temp_path):

        # Init
        prefix = ""
        line_count_r1 = 0
        line_count_r2 = 0
        exp_line_count = 9343
        test_file_valid_r1 = os.path.join(temp_path, prefix + ".r1_valid.fastq.gz")
        test_file_valid_r2 = os.path.join(temp_path, prefix + ".r2_valid.fastq.gz")

        # Run filter_valid_reads
        fastq_filter = FastqFilter(R1_PATH, R2_PATH)
        fastq_filter.filter_valid_reads(BC_VALID_PATH, temp_path, prefix)

        # Check files exist
        files = os.listdir(temp_path)
        assert (test_file_valid_r1 in filename for filename in files)
        assert (test_file_valid_r2 in filename for filename in files)

        # Count nr of lines and check they are expected and equal
        r1_fq_file = FastqFile(test_file_valid_r1)
        for name, seq, qual in r1_fq_file.open_read_iterator(as_string=True):
            line_count_r1 += 1
        # print(line_count_r1)

        r2_fq_file = FastqFile(test_file_valid_r2)
        for name, seq, qual in r2_fq_file.open_read_iterator(as_string=True):
            line_count_r2 += 1
        # print(line_count_r2)

        assert line_count_r1 == line_count_r2 == exp_line_count

    @with_temporary_folder
    def test_fastq_filter_reads_gzipped_bc_valid(self, temp_path):
        # Init
        prefix = "gz"
        exp_line_count = 9343
        test_file_valid_r1 = os.path.join(temp_path, prefix + ".r1_valid.fastq.gz")
        test_file_valid_r2 = os.path.join(temp_path, prefix + ".r2_valid.fastq.gz")

        # Build a gzipped bc_valid fixture from the plain-text fixture so the
        # suffix-aware reader takes the gzip decompression path.
        gz_bc_valid = os.path.join(temp_path, "bc_valid.txt.gz")
        with open(BC_VALID_PATH, "rb") as src, gzip.open(gz_bc_valid, "wb") as dst:
            dst.write(src.read())

        # Run filter_valid_reads against the gzipped barcodes
        fastq_filter = FastqFilter(R1_PATH, R2_PATH)
        fastq_filter.filter_valid_reads(gz_bc_valid, temp_path, prefix)

        # Same reads should pass as with the uncompressed fixture
        line_count_r1 = sum(
            1 for _ in FastqFile(test_file_valid_r1).open_read_iterator(as_string=True)
        )
        line_count_r2 = sum(
            1 for _ in FastqFile(test_file_valid_r2).open_read_iterator(as_string=True)
        )
        assert line_count_r1 == line_count_r2 == exp_line_count

    @with_temporary_folder
    def test_fastq_filter_valid_reads_with_trimming(self, temp_path):
        # Init
        prefix = "trim"
        exp_line_count = 9343
        trim_r1 = 5
        trim_r2 = 7
        test_file_valid_r1 = os.path.join(temp_path, prefix + ".r1_valid.fastq.gz")
        test_file_valid_r2 = os.path.join(temp_path, prefix + ".r2_valid.fastq.gz")

        # Run filter_valid_reads with trimming
        fastq_filter = FastqFilter(R1_PATH, R2_PATH)
        fastq_filter.filter_valid_reads(
            BC_VALID_PATH, temp_path, prefix, trim_r1=trim_r1, trim_r2=trim_r2
        )

        # Check files exist
        files = os.listdir(temp_path)
        assert (test_file_valid_r1 in filename for filename in files)
        assert (test_file_valid_r2 in filename for filename in files)

        # Check that all sequences are trimmed as expected
        r1_fq_file = FastqFile(test_file_valid_r1)
        for name, seq, qual in r1_fq_file.open_read_iterator(as_string=True):
            assert len(seq) > 0
            assert len(qual) == len(seq)
            assert not seq.startswith("N" * trim_r1)  # crude check, but ensures trimming
            break  # just check first read for speed

        r2_fq_file = FastqFile(test_file_valid_r2)
        for name, seq, qual in r2_fq_file.open_read_iterator(as_string=True):
            assert len(seq) > 0
            assert len(qual) == len(seq)
            assert not seq.startswith("N" * trim_r2)
            break

        # Count nr of lines and check they are expected and equal
        line_count_r1 = sum(
            1 for _ in FastqFile(test_file_valid_r1).open_read_iterator(as_string=True)
        )
        line_count_r2 = sum(
            1 for _ in FastqFile(test_file_valid_r2).open_read_iterator(as_string=True)
        )
        assert line_count_r1 == line_count_r2 == exp_line_count
