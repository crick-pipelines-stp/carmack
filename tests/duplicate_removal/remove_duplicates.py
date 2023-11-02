import os
import csv
import pysam
import carmack.utils as utils
import hashlib

from carmack.duplicate_removal.remove_duplicates import DuplicateRemoval

from ..utils import with_temporary_folder

BAM_PATH = 'tests/data/hydrop_scatac_1_S1_R1.bam'
BAI_PATH = 'tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam.bai'
CSV_PATH = 'tests/data/bc_valid.csv'  

@with_temporary_folder
def test_tag_reads(self, temp_path): 
    # Init
    test_bam_tagged = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tagged.bam') 
    test_tsv = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tsv') 
    expected_hash_out_bam = '64e8cceae5f27d1febda6f642cbeaeb6' 
    expected_hash_out_tsv = '2a0e2f6df080ef2792c93b5c6afa1c4b'

    duplicate_rem = DuplicateRemoval(BAM_PATH, BAI_PATH, CSV_PATH)
    duplicate_rem.tag_and_deduplicate_reads(True, temp_path, dedup=False)
    # duplicate_rem.get_unique_readpairs("tests/data/")
    
    # hash_bam_tag = utils.file_md5(test_bam_tagged)
    # print(hash_bam_tag) 
    
    # hash_tsv = utils.file_md5(test_tsv)
    # print(hash_tsv) 

    utils.validate_file_md5(test_bam_tagged, expected_hash_out_bam)
    utils.validate_file_md5(test_tsv, expected_hash_out_tsv)


@with_temporary_folder
def test_tag_and_deduplicate_reads(self, temp_path): 
    # Init
    test_bam_tagged = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tagged.bam') 
    test_tsv = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.tsv') 

    test_log = os.path.join(temp_path, 'hydrop_scatac_1_S1_R1.dedup.stats_mqc.log') 
    expected_hash_out_bam = 'a1b1390af85853b2f164fe29a1e78839'  
    expected_hash_out_tsv = '2a0e2f6df080ef2792c93b5c6afa1c4b'
    expected_hash_out_log = '98db428472e315b423c2c8bc75966225' 

    duplicate_rem = DuplicateRemoval(BAM_PATH, BAI_PATH, CSV_PATH)
    duplicate_rem.tag_and_deduplicate_reads(True, temp_path, dedup=True)
    # duplicate_rem.get_unique_readpairs("tests/data/")
    
    # hash_bam_tag_dedup = utils.file_md5(test_bam_tagged)
    # print(hash_bam_tag_dedup) 

    # hash_tsv = utils.file_md5(test_tsv)
    # print(hash_tsv) 

    # hash_log = utils.file_md5(test_log)
    # print(hash_log) 

    utils.validate_file_md5(test_bam_tagged, expected_hash_out_bam)
    utils.validate_file_md5(test_tsv, expected_hash_out_tsv)
    utils.validate_file_md5(test_log, expected_hash_out_log)

