import os

from carmack.io.fastq_file import FastqFile
from ..utils import with_temporary_folder

TEST_NAME = "NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"
TEST_SEQ = "AGATCTCGGT"
TEST_QUAL = "AAAAAEEEEE"


def test_fastq_file_gzip_read(self):
    """Test reading gzip fastq file"""
    fq_file = FastqFile("tests/data/sc_10k.fastq.gz", paired_end=False)

    for (name, seq, qual) in fq_file.open_read_iterator(as_string=True):
        self.assertEqual(name, TEST_NAME)
        self.assertEqual(seq, TEST_SEQ)
        self.assertEqual(qual, TEST_QUAL)
        break


def test_fastq_file_raw_read(self):
    """Test reading raw fastq file"""
    fq_file = FastqFile("tests/data/small.fastq", paired_end=False)

    for (name, seq, qual) in fq_file.open_read_iterator(as_string=True):
        self.assertEqual(name, TEST_NAME)
        self.assertEqual(seq, TEST_SEQ)
        self.assertEqual(qual, TEST_QUAL)
        break


@with_temporary_folder
def test_fastq_file_gzip_write(self, tmp_path):
    """Test with write gzip file"""
    filename = os.path.join(tmp_path, "test.gz")
    fq_file = FastqFile(filename, paired_end=False)
    wstream = fq_file.open_write_stream()
    FastqFile.write_read(wstream, TEST_NAME, TEST_SEQ, TEST_QUAL)
    wstream.close()

    fq_file = FastqFile(filename, paired_end=False)

    for (name, seq, qual) in fq_file.open_read_iterator(as_string=True):
        self.assertEqual(name, TEST_NAME)
        self.assertEqual(seq, TEST_SEQ)
        self.assertEqual(qual, TEST_QUAL)
        break
