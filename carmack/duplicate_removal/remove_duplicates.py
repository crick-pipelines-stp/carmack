import os
import logging
import pysam
import csv

log = logging.getLogger()
class DuplicateRemoval:
    """
    Class that removes duplicate reads from bam files
    """

    def __init__(self, bam: str, bai: str, bc_valid_csv: str) -> None: 
        self.bam = bam 
        self.bai = bai
        self.bc_valid_csv = bc_valid_csv

    def tag_and_deduplicate_reads(self, log_progress, output_dir, dedup=True, prefix=None):
        if prefix == None:
            prefix = self.bam.rsplit("/", 1)[-1].split(".", 1)[0]

        # Init
        out_bam = os.path.join(output_dir, prefix + ".tagged.bam")
        unique_ids = set() 
        bc_dict = {}
        line_nr = 0
        start_equals_end_count = 0
        unique_count = 0
        duplicate_count = 0
        valid_bc_count = 0
        prev_read = None

        # Make valid barcodes dictionary with structure barcode_id:barcode
        with open(self.bc_valid_csv, 'r') as valid_barcodes:
            csv_reader = csv.reader(valid_barcodes)
            for line in csv_reader:
                valid_bc_count += 1
                barcode_id = line[0].split(' ', 1)[0] 
                barcode = line[1]
                bc_dict[barcode_id] = barcode
        
        # calculate ~1% of total valid barcodes for logging and set initial threshold to 10%
        one_perc = valid_bc_count / 100
        curr_perc_thresh = 0.1

        # Open the filtered BAM file for reading
        with pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai) as bam_file:
            # Open a new BAM file for writing with the barcode tag added
            with pysam.AlignmentFile(out_bam, "wb", header=bam_file.header) as tagged_bam:
                # Iterate over each read in the BAM file
                for read in bam_file:
                    # Get the read ID
                    read_id = read.query_name
                    line_nr += 1

                    # Log progress
                    if log_progress and line_nr % one_perc:
                        curr_perc = line_nr / valid_bc_count
                        
                        if curr_perc > curr_perc_thresh:
                            logging.info(f"LINES_PROCESSED: {line_nr}")
                            curr_perc_thresh += 0.1

                    # Process each read pair of the bam file
                    if prev_read is not None and prev_read.query_name == read.query_name:

                        # Get the start and end position 
                        start = min(read.reference_start, prev_read.reference_start) 
                        end = start + abs(read.template_length)

                        # if the start and end positions are the same, continue on to the next read pair
                        if start == end:
                            start_equals_end_count += 1
                            continue

                        # Check if the read ID is in the valid barcodes dictionary
                        if read_id in bc_dict:
                            # Get the barcode 
                            barcode = bc_dict[read_id]

                            # Tag reads in bam file with the barcode
                            prev_read.set_tag('BC', barcode)
                            read.set_tag('BC', barcode)

                            if dedup:
                                # If dedup is True, deduplicate the tagged reads before writing to the output BAM file
                                read_pair_string = f"{start}-{end}-{barcode}"

                                # If not in set of unique read_pair ids, mark as unique and write to file with unique read pairs
                                if read_pair_string not in unique_ids:
                                    unique_count += 1

                                    # Add the read_id and q-score to the dictionary
                                    unique_ids.add(read_pair_string)

                                    # Set custom is_duplicate tag to False
                                    prev_read.set_tag("DU", False)
                                    read.set_tag("DU", False)

                                    # Write to file containing only unique read pairs
                                    tagged_bam.write(prev_read)
                                    tagged_bam.write(read)

                                # If in set of unique read_pair ids, mark as duplicate and don't write to file 
                                else:
                                    duplicate_count += 1
                                    
                            else:
                                # If dedup is False, write the tagged reads to the output BAM file without prior deduplication
                                tagged_bam.write(prev_read)
                                tagged_bam.write(read)

                        else:
                            log.error(f"read ID: {read_id} not in valid barcode dictionary!")

                    prev_read = read
        
         # Write unique and duplicate read counts to a separate output file for multiqc reporting.
        if dedup:
            with open(os.path.join(output_dir, prefix + '.dedup.stats_mqc.log'), "w") as dedup_counts_stats_file: 
                dedup_counts_stats_file.write('unique_read_count,duplicate_read_count\n')
                dedup_counts_stats_file.write(f"{unique_count},{duplicate_count}")        