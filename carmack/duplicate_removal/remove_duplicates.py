import os
import logging
import pysam
import hashlib
import csv


class DuplicateRemoval:
    """
    Class that removes duplicate reads from bam files
    """

    def __init__(self, bam: str, bai: str, bc_valid_csv: str) -> None: 
        self.bam = bam 
        self.bai = bai
        self.bc_valid_csv = bc_valid_csv
    
    def tag_barcodes(self, output_dir, prefix=None):
        if prefix == None:
            prefix = self.bam.rsplit("/", 1)[-1].split(".", 1)[0]

        out_bam = os.path.join(output_dir, prefix + ".tagged.bam")

        # Open the BAM file for reading
        with pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai) as bam_file:
            # Open a new BAM file for writing with the barcode tag added
            with pysam.AlignmentFile(out_bam, "wb", header=bam_file.header) as output_bam:
                # Iterate over each read in the BAM file
                for read in bam_file:
                    # Get the read ID
                    read_id = read.query_name

                    # Check if the read ID is in the valid barcodes by iterating over the CSV file
                    with open(self.bc_valid_csv, 'r') as valid_barcodes:
                        csv_reader = csv.reader(valid_barcodes)
                        for line in csv_reader:
                            # get the barcode_id
                            barcode_id = line[0] 

                            if read_id in barcode_id:
                                # Get the barcode
                                barcode = line[1]
                                read.set_tag('BC', barcode)
                                break
                    
                    # Write the read to the output BAM file
                    output_bam.write(read)



