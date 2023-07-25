import os
import logging
import pysam
import hashlib

from .subprocess_stream import SubprocessStream


class DuplicateRemoval:
    """
    Class that removes duplicate reads from bam files
    """

    def __init__(self, bam: str) -> None: 
        self.bam = bam 
        # self.bam_read1 = bam_read1 
        # self.bam_read2 = bam_read2
    
    def process_bam_file(self): # as_string: bool = False
        """
        Open a bam file for reading, and yield the start position, end position and barcode for each read.
        """
        line_index = 0
        stream = None

        if self.compressor is not None:
            stream = SubprocessStream([self.compressor, "-c", "-d", self.filename], mode="r")
        else:
            stream = open(self.bam, "r")
        
        with stream as bam_file:
            for line in bam_file:
                # for each read, strip terminal spaces, and 
                # subset the start, end positions and barcode
                read = 'CAAGGTCGATGATGCCTCAATTGAGTTCTC'
                barcode = 'CTATAGTCTT'

                start = 1
                end = 30
                # Adjust start and end positions for soft clipping

                yield (read, start, end, barcode) # evt. return more data from the other columns


    def label_unique_reads(self): 
        # Init
        unique_barcodes = set()

        # Iterate over each line, and store the barcode in a set
        # For all lines, the first instance of a unique barcode hash causes hat line to be labelled as unique, and the others as duplicates

        for (read, start, end, barcode) in self.process_bam_file():

            if barcode in unique_barcodes:
                # Add barcode to set of unique barcodes (hashes)
                unique_barcodes.add(barcode)
                duplicate = False
            else:
                duplicate = True




        # Return labelled reads

