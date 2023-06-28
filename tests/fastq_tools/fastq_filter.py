import os

from ..io.fastq_file import FastqFile
from carmack.fastq_tools.fastq_filter import FastqFilter

from ..utils import with_temporary_folder

R1_PATH = 'tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz'
R2_PATH = 'tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz'
BC_VALID_PATH = 'tests/data/bc_valid.csv'

@with_temporary_folder
def test_filter_valid_reads(self, temp_path):
    # Init
    prefix = ''
    line_count_r1 = 0
    line_count_r2 = 0
    exp_line_count = 9343
    test_file_valid_r1 = os.path.join(temp_path, prefix + '.r1_valid.fastq.gz')
    test_file_valid_r2 = os.path.join(temp_path, prefix + '.r2_valid.fastq.gz')

    # Run filter_valid_reads
    fastq_filter = FastqFilter(R1_PATH, R2_PATH)
    fastq_filter.filter_valid_reads(BC_VALID_PATH, temp_path, prefix)

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