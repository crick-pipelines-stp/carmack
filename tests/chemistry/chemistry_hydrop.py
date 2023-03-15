import os
import pytest
import numpy as np

import carmack.utils as utils
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile

from ..utils import with_temporary_folder

TEST_BC_1 = "TGTAGCAAGT"
TEST_BC_2 = "TTAGTTGGAC"
TEST_BC_3 = "TGACCGTACT"

BC_READS_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

def test_hydrop_load_barcode_set(self):
    """Test loading barcode set for hydrop chemistry."""
    chemistry = ChemistryHydrop()
    barcode_set = chemistry.load_barcode_set()

    self.assertEqual(barcode_set[0][0], TEST_BC_1)
    self.assertEqual(barcode_set[1][0], TEST_BC_2)
    self.assertEqual(barcode_set[2][0], TEST_BC_3)

    self.assertEqual(len(barcode_set[0]), 96)
    self.assertEqual(len(barcode_set[1]), 96)
    self.assertEqual(len(barcode_set[2]), 96)

def test_hydrop_load_barcode_set_with_factory(self):
    """Test loading barcode set for hydrop chemistry."""
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()

    self.assertEqual(barcode_set[0][0], TEST_BC_1)
    self.assertEqual(barcode_set[1][0], TEST_BC_2)
    self.assertEqual(barcode_set[2][0], TEST_BC_3)

    self.assertEqual(len(barcode_set[0]), 96)
    self.assertEqual(len(barcode_set[1]), 96)
    self.assertEqual(len(barcode_set[2]), 96)

def test_hydrop_construct_whitelist_model_case(self):
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)

    # KEEP UPDATED WITH MODEL SEQUENCE CONSTRUCTION
    test_wl = barcode_set[0][0] + barcode_set[1][0] + barcode_set[2][0]
    
    assert barcode_wl[0] == test_wl

@with_temporary_folder
def test_hydrop_construct_whitelist_md5(self, temp_path):
    expected_hash = 'c0126c596b5a3cef10a50195ddc8e224'
    test_file = os.path.join(temp_path, 'barcodes.txt')

    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)

    with open(test_file, 'w') as out_file:
        for bc_wl in barcode_wl:
            out_file.write(bc_wl + '\n')

    utils.validate_file_md5(test_file, expected_hash)

@pytest.mark.parametrize("seq, expected_seq", [
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGGACCGT', None), # Too short
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT', "TGACCGTACTTTAGTTGGACTGTAGCAAGT"), # Exactly 50
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACTGCT', "TGACCGTACTTTAGTTGGACTGTAGCAAGT"), # Long perm 1
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACTAAAAAAAAAAAAAAAAAAAA', "TGACCGTACTTTAGTTGGACTGTAGCAAGT") # Long perm 2
]) 
def test_subset_whitelist_guess_perm(self, seq, expected_seq):
    """Test generating quick whitelist matches from variable length data"""
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    wl_seq = chemistry.subset_whitelist_guess(seq)

    # print("")
    # print(wl_seq)

    assert wl_seq == expected_seq


@pytest.mark.parametrize("seq, qs, expected_seq, expected_qs, expected_msg", [
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGT', None, None, None, "SUBSET:SEQLEN<50"), # Sequence is too short
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGTGTACTCGTGACCGTACT', None, None, None, "SUBSET:SPC1_NOTFND"), # Spacer 1 cant be found
('TGTAGCAAGTGCAGTTGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT', None, None, None, "SUBSET:SPC2_NOTFND"), # Spacer 2 cant be found
('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT', None, ['TEST', 'TEST', 'TEST'], ['TEST', 'TEST', 'TEST'], ""), # Perfect with no indels
]) 
def test_hydrop_subset_barcodes_perm(self, seq, qs, expected_seq, expected_qs, expected_msg):
    """Test subsetting barcodes from sequence for hydrop chemistry with variable permitations"""
    barcode_chunks = None
    qs_chunks = None

    if(qs is None):
        qs = np.full(len(seq), 30)

    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_chunks, qs_chunks = chemistry.subset_barcode_chunks(seq, qs)
        
    print("")
    print(barcode_chunks)
    print(qs_chunks)

    if expected_seq is None:
        assert barcode_chunks is None
    else:
        for idx, bc in enumerate(barcode_chunks):
            assert bc == expected_seq[idx]

    if expected_qs is None:
        assert qs_chunks is None
    else:
        for idx, bc in enumerate(qs_chunks):
            assert bc == expected_qs[idx]


@with_temporary_folder
def test_hydrop_subset_barcodes_md5(self, temp_path):
    """Test subsetting barcodes from sequence for hydrop chemistry."""

    expected_hash = 'a36c5ce249da5d081a469017700c2066'

    test_file = os.path.join(temp_path, 'barcodes.txt')
    fq_file = FastqFile(BC_READS_PATH)
    stream = fq_file.open_read_iterator(as_string=True)
    chemistry = ChemistryFactory.get_chemistry('hydrop')

    with open(test_file, 'w') as out_file:
        for (name, seq, qual) in stream:
            qs = np.full(len(seq), 30)
            barcode_chunks, qs_chunks = chemistry.subset_barcode_chunks(seq, qs)
            line = ','.join(barcode_chunks)
            out_file.write(line + '\n')

    utils.validate_file_md5(test_file, expected_hash)

def test_hydrop_subset_barcodes_qs_subset(self):
    seq = 'TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT'
    qs = np.full(len(seq), 30)

    barcode_chunks, qs_chunks = ChemistryHydrop.subset_barcode_chunks(seq, qs)
    print(barcode_chunks)