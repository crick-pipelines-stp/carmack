# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest

from carmack.io.fastq_file import FastqFile
from tests.utils import with_temporary_folder


TEST_NAME = "NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"
TEST_SEQ = "AGATCTCGGT"
TEST_QUAL = "AAAAAEEEEE"


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
