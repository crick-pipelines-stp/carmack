# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest

from carmack.io.gzip_file import GzipFile
from tests.utils import with_temporary_folder

TEST_NAME = "@NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"


class TestGzipFile(unittest.TestCase):
    def test_gzip_file_read(self):
        """Test reading gzip fastq file"""
        gz_file = GzipFile("tests/data/sc_10k.fastq.gz")

        for line in gz_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break

    def test_gzip_file_read_raw(self):
        """Test reading gzip file thats actually not compressed"""
        raw_file = GzipFile("tests/data/small.fastq")

        for line in raw_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break

    @with_temporary_folder
    def test_gzip_file_write(self, tmp_path):
        """Test with write gzip file"""
        filename = os.path.join(tmp_path, "test.gz")
        gz_file = GzipFile(filename)
        wstream = gz_file.open_write_stream()
        GzipFile.write_string(wstream, TEST_NAME)
        wstream.close()

        gz_file = GzipFile(filename)

        for line in gz_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break
