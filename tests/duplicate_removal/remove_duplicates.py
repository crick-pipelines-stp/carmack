import os
import csv
import pysam
import carmack.utils as utils
import hashlib

from carmack.duplicate_removal.remove_duplicates import DuplicateRemoval

from ..utils import with_temporary_folder

BAM_PATH = 'tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam'
BAM_PATH_UNSORT = 'tests/data/hydrop_scatac_1_S1_R1.bam'
BAI_PATH = 'tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam.bai'
CSV_PATH = 'tests/data/bc_valid.csv'

@with_temporary_folder
def test_barcode_tagging(self, temp_path):
    # Init
    test_file = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tagged.bam')
    expected_hash = 'b44f0f15c69a3d95a128c61d384510f9'

    duplicate_rem = DuplicateRemoval(BAM_PATH, BAI_PATH, CSV_PATH)

    duplicate_rem.tag_barcodes(temp_path)
    
    # hash = utils.file_md5(test_file)
    # print(hash)

    utils.validate_file_md5(test_file, expected_hash)                        

# @with_temporary_folder
# def test_get_unique_readpairs(self, temp_path):
#     # Init
#     test_out_bam = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tagged.bam') 
#     test_out_unique = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.unique_valid_reads.txt') 
#     expected_hash_out_bam = '64e8cceae5f27d1febda6f642cbeaeb6'
#     expected_hash_out_unique = '50fac7a1cb82a67ba663105e35dadab7'

#     duplicate_rem = DuplicateRemoval(BAM_PATH_UNSORT, BAI_PATH, CSV_PATH)
#     duplicate_rem.get_unique_readpairs(temp_path)
    
#     hash_bam = utils.file_md5(test_out_bam)
#     hash_unique = utils.file_md5(test_out_unique)

#     utils.validate_file_md5(test_out_bam, expected_hash_out_bam)
#     utils.validate_file_md5(test_out_unique, expected_hash_out_unique)                        

