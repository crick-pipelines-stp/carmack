import os
import logging

# import rich_click as click # or import click
from ..io.fastq_file import FastqFile
from ..barcode.barcode_extractor import BarcodeExtractor

class FastqFilter:
    """
    Class that filters fastq files for valid reads
    """

    def __init__(self, read1: str, read2: str) -> None: #  cell_barcode: str, chemistry: str, paired_end: bool = False
        self.read1 = read1
        self.read2 = read2
        # self.cell_barcode = cell_barcode
        # self.paired_end = paired_end
        # self.chemistry = ChemistryFactory.get_chemistry(chemistry)

    def filter_valid_reads(self, bc_valid, output_dir, prefix=None): # is_index=False
        logging.info("VALID BARCODE READ FILTER")

        # Init
        bc_dict = {}

        # Load valid barcodes
        with open(bc_valid, "r") as bc_valid_file:
            for line in bc_valid_file:
                line_split = line.split(',')
                name_split_bc = line_split[0].split(' ')
                bc_dict[name_split_bc[0]] = line_split[1]

        r1_fq_file = FastqFile(self.read1) 
        r2_fq_file = FastqFile(self.read2) 

        filtered_r1 = os.path.join(output_dir, prefix + ".r1_valid.fastq.gz")
        filtered_r2 = os.path.join(output_dir, prefix + ".r2_valid.fastq.gz")
        fq_r1_file = FastqFile(filtered_r1)
        fq_r2_file = FastqFile(filtered_r2)

        wstream_r1 = fq_r1_file.open_write_stream()
        for (name, seq, qual) in r1_fq_file.open_read_iterator(as_string=True):
            name_split_read = name.split(' ')
            
            if name_split_read[0] in bc_dict:
                FastqFile.write_read(wstream_r1, name, seq, qual)
                # evt. if else statement using is_index argument.
        wstream_r1.close()
        
        wstream_r2 = fq_r2_file.open_write_stream()
        for (name, seq, qual) in r2_fq_file.open_read_iterator(as_string=True):
            name_split_read = name.split(' ')
            
            if name_split_read[0] in bc_dict:
                FastqFile.write_read(wstream_r2, name, seq, qual)
                # evt. if else statement using is_index argument.
        wstream_r2.close()
