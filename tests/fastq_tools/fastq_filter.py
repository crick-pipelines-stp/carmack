import os
import carmack.utils as utils

from ..io.fastq_file import FastqFile
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.fastq_tools.fastq_filter import FastqFilter

from ..utils import with_temporary_folder

R1_PATH = 'tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz'
R2_PATH = 'tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz'
CB_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

@with_temporary_folder
def test_filter_valid_reads(self, temp_path):

    expected_hash_r1 = '5823bb4932ab8f16f06069c8d0205a59'
    expected_hash_r2 = '13d623f4ed59adf4ab75d7ae69b33093'

    # Init
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    max_corrections = 2
    count = 100
    prefix = ''
    line_count_r1 = 0
    line_count_r2 = 0
    exp_line_count = 9343
    test_file_valid_bc = os.path.join(temp_path, prefix + '.bc_valid.csv')
    test_file_valid_r1 = os.path.join(temp_path, prefix + '.r1_valid.fastq.gz')
    test_file_valid_r2 = os.path.join(temp_path, prefix + '.r2_valid.fastq.gz')

    # Run extract_cell_barcodes
    barcode_ext.extract_cell_barcodes(max_corrections, count, temp_path, prefix)

    # Run filter_valid_reads
    fastq_filter = FastqFilter(R1_PATH, R2_PATH)
    fastq_filter.filter_valid_reads(test_file_valid_bc, temp_path, prefix)

    # check files exist
    files = os.listdir(temp_path)
    assert (test_file_valid_r1 in filename for filename in files)
    assert (test_file_valid_r2 in  filename for filename in files)

    # count nr of lines and check they are expected and equal
    r1_fq_file = FastqFile(test_file_valid_r1)
    for (name, seq, qual) in r1_fq_file.open_read_iterator(as_string=True):
        line_count_r1 += 1
    # print(line_count_r1)

    r2_fq_file = FastqFile(test_file_valid_r2)
    for (name, seq, qual) in r2_fq_file.open_read_iterator(as_string=True):
        line_count_r2 += 1
    # print(line_count_r2)

    assert (line_count_r1 == line_count_r2  == exp_line_count)