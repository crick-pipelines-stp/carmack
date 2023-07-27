import os
import csv
import pysam
import carmack.utils as utils

from carmack.duplicate_removal.remove_duplicates import DuplicateRemoval

from ..utils import with_temporary_folder

BAM_PATH = 'tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam'
BAI_PATH = 'tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam.bai'
CSV_PATH = 'tests/data/bc_valid.csv'

@with_temporary_folder
def test_bam_file_contents(self, temp_path):
    # Init
    test_file = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tagged.bam')
    expected_hash = 'b44f0f15c69a3d95a128c61d384510f9'

    duplicate_rem = DuplicateRemoval(BAM_PATH, BAI_PATH, CSV_PATH)

    duplicate_rem.tag_barcodes(temp_path)
    
    # hash = utils.file_md5(test_file)
    # print(hash)

    utils.validate_file_md5(test_file, expected_hash)                        


# @with_temporary_folder
# def test_bam_file_contents(self, temp_path):
#     # Init
#     test_file = os.path.join(temp_path, 'output.bam')
#     expected_hash = 'b44f0f15c69a3d95a128c61d384510f9'
#     # line_nr = 0

#     # Open the BAM file for reading
#     with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as bam_file:
#         # Open a new BAM file for writing with the barcode tag added
#         with pysam.AlignmentFile(test_file, "wb", header=bam_file.header) as output_bam:
#                 # Check if the read ID is in the valid barcodes by iterating over the CSV file
#                 with open(CSV_PATH, 'r') as csv_file:
#                     csv_reader = csv.reader(csv_file)
#                     for line in csv_reader:
#                         # Get the barcode_id
#                         barcode_id = line[0].split(' ')[0] 
#                         print(barcode_id)

#                         # Get the barcode
#                         barcode = line[1]
#                         print(barcode)

#                         for read in bam_file.fetch(query_name=barcode_id):
#                             print(read)
#                             # Tag with barcode
#     #                         read.set_tag('BC', barcode)

#     #                         output_bam.write(read)
    
#     # # # hash = utils.file_md5(test_file)
#     # # # print(hash)
#     # utils.validate_file_md5(test_file, expected_hash)  
