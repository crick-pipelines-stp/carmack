import logging
import os

from ..io.fastq_file import FastqFile
from ..io.gzip_file import GzipFile


log = logging.getLogger(__name__)


class FastqFilter:
    """
    Class that filters fastq files for valid reads
    """

    def __init__(self, read1: str, read2: str) -> None:
        self.read1 = read1
        self.read2 = read2

    def filter_valid_reads(self, bc_valid, output_dir, prefix=None, trim_r1=0, trim_r2=0):
        """
        Filter reads in read1/read2 by valid barcodes, with optional trimming of first n bases from each read.

        Args:
            bc_valid (str): Path to file with valid barcodes.
            output_dir (str): Output directory for filtered fastq files.
            prefix (str, optional): Prefix for output files.
            trim_r1 (int, optional): Number of bases to trim from start of read1. Default 0.
            trim_r2 (int, optional): Number of bases to trim from start of read2. Default 0.
        """
        log.info("VALID BARCODE READ FILTER")

        # Init
        valid_barcodes = set()

        # Load valid barcodes. GzipFile selects its codec from the filename
        # suffix, so this reads both gzipped (.txt.gz) and legacy plain-text
        # (.txt/.csv) barcode files.
        for line in GzipFile(bc_valid).open_read_iterator(as_string=True):
            barcode = line.split(",")[0].split(" ")[0]
            valid_barcodes.add(barcode)

        r1_fq = FastqFile(self.read1)
        r2_fq = FastqFile(self.read2)

        filtered_r1 = os.path.join(output_dir, prefix + ".r1_valid.fastq.gz")
        filtered_r2 = os.path.join(output_dir, prefix + ".r2_valid.fastq.gz")
        r1_fq_filtered = FastqFile(filtered_r1)
        r2_fq_filtered = FastqFile(filtered_r2)

        wstream_r1 = r1_fq_filtered.open_write_stream()
        for name, seq, qual in r1_fq.open_read_iterator(as_string=True):
            name_split_read = name.split(" ")

            if name_split_read[0] in valid_barcodes:
                if trim_r1 > 0:
                    seq = seq[trim_r1:]
                    qual = qual[trim_r1:]
                FastqFile.write_read(wstream_r1, name, seq, qual)
        wstream_r1.close()

        wstream_r2 = r2_fq_filtered.open_write_stream()
        for name, seq, qual in r2_fq.open_read_iterator(as_string=True):
            name_split_read = name.split(" ")

            if name_split_read[0] in valid_barcodes:
                if trim_r2 > 0:
                    seq = seq[trim_r2:]
                    qual = qual[trim_r2:]
                FastqFile.write_read(wstream_r2, name, seq, qual)
        wstream_r2.close()
