# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import subprocess
import unittest
from pathlib import Path

from carmack.io.fastq_file import FastqFile
from tests.utils import gzip_bytes, with_temporary_folder

TEST_NAME = "NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"
TEST_SEQ = "AGATCTCGGT"
TEST_QUAL = "AAAAAEEEEE"

# A payload large enough that a truncated copy still holds well over a thousand intact
# records, so the records that do come back look exactly like a healthy short run and
# nothing in the data marks the file as cut off part-way.
TRUNCATION_READ_COUNT = 2500
TRUNCATION_TEXT = "".join(
    f"@{TEST_NAME} record {index:08d}\n{TEST_SEQ}\n+\n{TEST_QUAL}\n"
    for index in range(TRUNCATION_READ_COUNT)
)

# Fraction of the gzip member kept when building the truncated copy.
TRUNCATION_FRACTION = 0.6


class TestFastqFile(unittest.TestCase):
    def test_fastq_file_gzip_read(self):
        """Test reading gzip fastq file"""
        fq_path = "tests/data/sc_10k.fastq.gz"
        fq_file = FastqFile(fq_path, paired_end=False)

        for name, seq, qual in fq_file.open_read_iterator(as_string=True):
            self.assertEqual(name, TEST_NAME)
            self.assertEqual(seq, TEST_SEQ)
            self.assertEqual(qual, TEST_QUAL)
            break

        self.assertEqual(fq_file.filename, fq_path)
        self.assertEqual(fq_file.reads_count, 10000)

    def test_fastq_file_raw_read(self):
        """Test reading raw fastq file"""
        fq_path = "tests/data/small.fastq"
        fq_file = FastqFile(fq_path, paired_end=False)

        for name, seq, qual in fq_file.open_read_iterator(as_string=True):
            self.assertEqual(name, TEST_NAME)
            self.assertEqual(seq, TEST_SEQ)
            self.assertEqual(qual, TEST_QUAL)
            break

        self.assertEqual(fq_file.filename, fq_path)
        self.assertEqual(fq_file.reads_count, 2500)

    @with_temporary_folder
    def test_fastq_file_gzip_write(self, tmp_path):
        """Test with write gzip file"""
        filename = os.path.join(tmp_path, "test.gz")
        fq_file = FastqFile(filename, paired_end=False)
        wstream = fq_file.open_write_stream()
        FastqFile.write_read(wstream, TEST_NAME, TEST_SEQ, TEST_QUAL)
        wstream.close()

        fq_file = FastqFile(filename, paired_end=False)

        for name, seq, qual in fq_file.open_read_iterator(as_string=True):
            self.assertEqual(name, TEST_NAME)
            self.assertEqual(seq, TEST_SEQ)
            self.assertEqual(qual, TEST_QUAL)
            break


class TestFastqFileReadFailure(unittest.TestCase):
    def write_truncated_fastq(self, folder):
        """
        Write a gzip FASTQ file whose member has been cut off part-way through.

        Args:
            folder: Directory to write the fixture into.

        Returns:
            The path of the truncated file, as a str.
        """

        healthy = gzip_bytes(TRUNCATION_TEXT)
        path = Path(folder) / "truncated.fastq.gz"
        path.write_bytes(healthy[: int(len(healthy) * TRUNCATION_FRACTION)])
        return str(path)

    def count_reads(self, filename):
        """
        Count the reads in a FASTQ file, giving the property access a caller of its own.

        Args:
            filename: Path of the file to count.

        Returns:
            The number of reads the file holds.
        """

        return FastqFile(filename, paired_end=False).reads_count

    @with_temporary_folder
    def test_fastq_file_read_iterator_reports_a_truncated_file(self, tmp_path):
        """Test reading a truncated FASTQ to the end is reported rather than accepted"""
        filename = self.write_truncated_fastq(tmp_path)

        reads = []
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            for read in FastqFile(filename, paired_end=False).open_read_iterator(as_string=True):
                reads.append(read)

        self.assertEqual(caught.exception.returncode, 1)

        # Every record that did arrive is intact, so a caller looking only at the data
        # sees a perfectly well-formed FASTQ that simply happens to be shorter than the
        # run that produced it. The exit status is the only thing that says otherwise.
        self.assertGreater(len(reads), 0)
        self.assertLess(len(reads), TRUNCATION_READ_COUNT)
        self.assertEqual(reads[0][1], TEST_SEQ)
        self.assertEqual(reads[0][2], TEST_QUAL)

    @with_temporary_folder
    def test_fastq_file_reads_count_reports_a_truncated_file(self, tmp_path):
        """Test counting the reads in a truncated FASTQ is reported as well"""
        filename = self.write_truncated_fastq(tmp_path)

        # reads_count opens a stream of its own rather than going through the read
        # iterator, so it needs the check independently or a count taken over a truncated
        # input is quietly reported as the size of the run.
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.count_reads(filename)

        self.assertEqual(caught.exception.returncode, 1)
