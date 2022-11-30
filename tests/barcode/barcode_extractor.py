import os
import numpy as np

import carmack.utils as utils
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.barcode.barcode_extractor import BarcodeExtractor

from ..utils import with_temporary_folder

R1_PATH = 'tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz'
R2_PATH = 'tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz'
CB_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

@with_temporary_folder
def test_calc_raw_barcode_match_counts_hydrop(self, temp_path):
    """Test calculation of raw barcode match counts."""

    expected_hash = '777cf483afbc2db0408f151baa693037'

    # Init
    test_file = os.path.join(temp_path, 'barcode_counts.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_counts = barcode_ext.calc_raw_barcode_match_counts()

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_counts:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)


@with_temporary_folder
def test_calc_raw_barcode_match_distribution_hydrop(self, temp_path):
    """Test calculation of raw barcode match distribution."""

    expected_hash = '18d6549067432e6ca3d7d7843f02059c'

    # Init
    test_file = os.path.join(temp_path, 'barcode_dist.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_dist:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)

# Tests when number of N's is greater than max dist - should return none
# Test changes in max dist - 

def test_gen_nearby_seqs(self):
    """Test generation of nearby sequences."""

    # Init
    seq = 'TGTAGCAAGN'
    qs = np.array([30,30,30,30,30,30,30,30,30,30])
    maxdist = 1
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Generate nearby sequences
    nearby_seqs = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

    # # Check
    # assert nearby_seqs == [('ACGTACGT', 'IIIIIIII'), ('ACGTACGG', 'IIIIIIII')]