import os
import logging

from ..io.fastq_file import FastqFile

class FastqFilter:
    """
    Class that filters fastq files for valid reads
    """

    def __init__(self, read1: str, read2: str) -> None: 
        self.read1 = read1
        self.read2 = read2

    def filter_valid_reads(self, bc_valid, output_dir, prefix=None): 
        logging.info("VALID BARCODE READ FILTER")

        # Init
        valid_barcodes = set()

        # Load valid barcodes
        with open(bc_valid, "r") as bc_valid_file:
            for line in bc_valid_file:
                barcode = line.split(',')[0].split(' ')[0]
                valid_barcodes.add(barcode)

        r1_fq = FastqFile(self.read1) 
        r2_fq = FastqFile(self.read2) 

        filtered_r1 = os.path.join(output_dir, prefix + ".r1_valid.fastq.gz")
        filtered_r2 = os.path.join(output_dir, prefix + ".r2_valid.fastq.gz")
        r1_fq_filtered = FastqFile(filtered_r1)
        r2_fq_filtered = FastqFile(filtered_r2)

        wstream_r1 = r1_fq_filtered.open_write_stream()
        for (name, seq, qual) in r1_fq.open_read_iterator(as_string=True):
            name_split_read = name.split(' ')
            
            if name_split_read[0] in valid_barcodes:
                FastqFile.write_read(wstream_r1, name, seq, qual)
        wstream_r1.close()
        
        wstream_r2 = r2_fq_filtered.open_write_stream()
        for (name, seq, qual) in r2_fq.open_read_iterator(as_string=True):
            name_split_read = name.split(' ')
            
            if name_split_read[0] in valid_barcodes:
                FastqFile.write_read(wstream_r2, name, seq, qual)
        wstream_r2.close()
