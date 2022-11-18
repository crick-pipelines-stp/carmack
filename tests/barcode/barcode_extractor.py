import os

import carmack.utils as utils
from carmack.barcode.barcode_extractor import BarcodeExtractor

from ..utils import with_temporary_folder

R1_PATH = 'tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz'
R2_PATH = 'tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz'
CB_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

@with_temporary_folder
def test_calc_raw_barcode_match_distribution_hydrop(self, temp_path):
    """Test calculation of raw barcode match distribution."""

    expected_hash = '777cf483afbc2db0408f151baa693037'

    # Init
    test_file = os.path.join(temp_path, 'barcode_counts.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_counts = barcode_ext.calc_raw_barcode_match_distribution()

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_counts:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)
