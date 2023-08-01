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
    
    # def tag_barcodes(self, output_dir, prefix=None):
    #     if prefix == None:
    #         prefix = self.bam.rsplit("/", 1)[-1].split(".", 1)[0]

    #     out_bam = os.path.join(output_dir, prefix + ".tagged.bam")

    #     # Open the BAM file for reading
    #     with pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai) as bam_file:
    #         # Open a new BAM file for writing with the barcode tag added
    #         with pysam.AlignmentFile(out_bam, "wb", header=bam_file.header) as output_bam:
    #             # Iterate over each read in the BAM file
    #             for read in bam_file:
    #                 # Get the read ID
    #                 read_id = read.query_name

    #                 # Check if the read ID is in the valid barcodes by iterating over the CSV file
    #                 with open(self.bc_valid_csv, 'r') as valid_barcodes:
    #                     csv_reader = csv.reader(valid_barcodes)
    #                     for line in csv_reader:
    #                         # get the barcode_id
    #                         barcode_id = line[0] 

    #                         if read_id in barcode_id:
    #                             # Get the barcode
    #                             barcode = line[1]
    #                             read.set_tag('BC', barcode)
    #                             break
                    
    #                 # Write the read to the output BAM file
    #                 output_bam.write(read)
        
    def get_unique_readpairs(self, output_dir, prefix=None):
        if prefix == None:
            prefix = self.bam.rsplit("/", 1)[-1].split(".", 1)[0]

        # Init
        out_bam = os.path.join(output_dir, prefix + ".tagged.bam")
        out_unique = os.path.join(output_dir, prefix + ".unique_valid_reads.txt")
        unique_ids = {}
        line_nr = 0
        prev_read = None

        # Open the unsorted BAM file for reading
        with pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai) as bam_file:
            # Open a new BAM file for writing with the barcode tag added
            with pysam.AlignmentFile(out_bam, "wb", header=bam_file.header) as output_bam:
                # Iterate over each read in the BAM file
                for read in bam_file:
                    # Get the read ID
                    read_id = read.query_name
                    line_nr += 1

                    # Process each read pair of the bam file
                    if prev_read is not None and prev_read.query_name == read.query_name:
                        # get the summed mapping quality
                        map_qual_sum = read.mapping_quality + prev_read.mapping_quality 
                        
                        # Get the start and end position 
                        start = min(read.reference_start, prev_read.reference_start) 
                        end = start + abs(read.template_length)

                        # if the start and end positions are the same, continue on to the next read pair
                        if start == end:
                            continue

                        # Get the barcode
                        # Check if the read ID is in the valid barcodes by iterating over the CSV file
                        with open(self.bc_valid_csv, 'r') as valid_barcodes:
                            csv_reader = csv.reader(valid_barcodes)
                            for line in csv_reader:
                                # get the barcode_id
                                barcode_id = line[0] 

                                if read_id in barcode_id:
                                    # Get the barcode 
                                    barcode = line[1]

                                    # Tag reads with the barcode
                                    prev_read.set_tag('BC', barcode)
                                    read.set_tag('BC', barcode)

                                    # get a hashed string of the read pair 
                                    string_to_hash = f"{start}-{end}-{barcode}"
                                    hash_object = hashlib.md5(string_to_hash.encode())
                                    hash_value = hash_object.hexdigest()

                                    # If not in set of unique read_pair ids, add it and mark as unique, if in set of unique read_pair ids, mark as duplicate
                                    if hash_value not in unique_ids:
                                        # Add the read_id and q-score to the dictionary
                                        unique_ids[hash_value] = [read_id, map_qual_sum]
                                    
                                    # If ID already exists, still update the entry if the mapping quality is higher.
                                    elif unique_ids[hash_value][1] < map_qual_sum: 
                                         print(unique_ids[hash_value][1])
                                         unique_ids[hash_value] = [read_id, map_qual_sum] 
                                    break
                        
                        # Write the reads to the output BAM file
                        output_bam.write(prev_read)
                        output_bam.write(read)

                    prev_read = read
        
        valid_unique_read_ids = [i[1][0] for i in unique_ids.items()]

        with open(out_unique, "w") as valid_unique_readpairs:
            for unique_id in valid_unique_read_ids:
                valid_unique_readpairs.write("%s\n" % unique_id)