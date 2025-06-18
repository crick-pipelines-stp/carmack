# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import pytest
import numpy as np
import unittest

import carmack.utils as utils
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile

from tests.utils import with_temporary_folder

TEST_BC_1 = "TGTAGCAAGT"
TEST_BC_2 = "TTAGTTGGAC"
TEST_BC_3 = "TGACCGTACT"

BC_READS_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

class TestChemistryHydrop(unittest.TestCase):
    def test_chem_hydrop_load_barcode_set(self):
        """Test loading barcode set for hydrop chemistry."""
        chemistry = ChemistryHydrop()
        barcode_set = chemistry.load_barcode_set()
        # print(barcode_set)

        self.assertIn(TEST_BC_1, list(barcode_set[0]))
        self.assertIn(TEST_BC_2, list(barcode_set[1]))
        self.assertIn(TEST_BC_3, list(barcode_set[2]))

        self.assertEqual(len(barcode_set[0]), 96)
        self.assertEqual(len(barcode_set[1]), 96)
        self.assertEqual(len(barcode_set[2]), 96)

    def test_chem_hydrop_load_barcode_set_with_factory(self):
        """Test loading barcode set for hydrop chemistry."""
        chemistry = ChemistryFactory.get_chemistry('hydrop')
        barcode_set = chemistry.load_barcode_set()

        self.assertIn(TEST_BC_1, list(barcode_set[0]))
        self.assertIn(TEST_BC_2, list(barcode_set[1]))
        self.assertIn(TEST_BC_3, list(barcode_set[2]))

        self.assertEqual(len(barcode_set[0]), 96)
        self.assertEqual(len(barcode_set[1]), 96)
        self.assertEqual(len(barcode_set[2]), 96)

    def test_chem_hydrop_construct_whitelist_model_case(self):
        chemistry = ChemistryFactory.get_chemistry('hydrop')
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)

        # KEEP UPDATED WITH MODEL SEQUENCE CONSTRUCTION
        test_wl = list(barcode_set[0])[0] +  list(barcode_set[1])[0] +  list(barcode_set[2])[0]

        assert test_wl in barcode_wl

    @with_temporary_folder
    def test_chem_hydrop_construct_whitelist_md5(self, temp_path):
        expected_hash = 'fa1805f3bd180de020eb7e6c6512ffeb'
        test_file = os.path.join(temp_path, 'barcodes.txt')

        chemistry = ChemistryFactory.get_chemistry('hydrop')
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = sorted(chemistry.construct_whitelist(barcode_set))

        with open(test_file, 'w') as out_file:
            for bc_wl in barcode_wl:
                out_file.write(bc_wl + '\n')
        utils.validate_file_md5(test_file, expected_hash)

    @with_temporary_folder
    def test_chem_hydrop_subset_barcodes_md5(self, temp_path):
        """Test subsetting barcodes from sequence for hydrop chemistry."""

        expected_hash = '79ef6c4886a11509e93929c67aed1e7e'

        test_file = os.path.join(temp_path, 'barcodes.txt')
        fq_file = FastqFile(BC_READS_PATH)
        stream = fq_file.open_read_iterator(as_string=True)
        chemistry = ChemistryFactory.get_chemistry('hydrop')

        with open(test_file, 'w') as out_file:
            for (name, seq, qual) in stream:
                qs = np.full(len(seq), 30)
                barcode_chunks, qs_chunks, msg = chemistry.subset_barcode_chunks(seq, qs)
                
                if barcode_chunks is not None:
                    line = ','.join(barcode_chunks)
                    out_file.write(line + '\n')

        utils.validate_file_md5(test_file, expected_hash)

class TestChemistryHydropFixtures():
    @pytest.mark.parametrize("seq, expected_seq", [
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGGACCGT', None), # Too short
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT', "TGACCGTACTTTAGTTGGACTGTAGCAAGT"), # Exactly 50
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACTGCT', "TGACCGTACTTTAGTTGGACTGTAGCAAGT"), # Long perm 1
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGTACTAAAAAAAAAAAAAAAAAAAA', "TGACCGTACTTTAGTTGGACTGTAGCAAGT") # Long perm 2
    ])
    def test_chem_subset_whitelist_guess_perm(self, seq, expected_seq):
        """Test generating quick whitelist matches from variable length data"""
        chemistry = ChemistryFactory.get_chemistry('hydrop')
        wl_seq = chemistry.subset_whitelist_guess(seq)

        assert wl_seq == expected_seq


    @pytest.mark.parametrize("seq, qs, expected_seq, expected_qs, expected_msg", [
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGGGTACTCGTGACCGT', None, None, None, "SUBSET:SEQLEN<50"), # Sequence is too short
    ('TGTAGCAAGTGCAGTAGCTGTTAGTTGGACAGTGTACTCGTGACCGTACT', None, None, None, "SUBSET:SPC1_NOTFND"), # Spacer 1 cant be found
    ('TGTAGCAAGTGCAGTTGCTGTTAGTTGGACAGGGTACTCGTGACCGTACT', None, None, None, "SUBSET:SPC2_NOTFND"), # Spacer 2 cant be found
    ('GGTTAATCACAGGGTACTCGAATAGCGTGGGCAGTAGCTGCCGTTCGTCCGT', None, ['CCGTTCGTCC', 'AATAGCGTGG', 'GGTTAATCAC'], [np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])], "SUBSET:OK"), # Perfect with no indels
    ('CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT', None, ['GAACAGTAGT', 'ACGGTGGACT', 'CAGTGTGGAA'], [np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])], "SUBSET:OK"), # Perfect with no indels 2
    ('TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA', None, ['CTCCTCATCCG', 'ACCAAGAGA', 'TCCTGATAAG'], [np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])], "SUBSET:INDL"), # Indel 1
    ('GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC', None, ['TTGAGATCGT', 'GGAGCTTGTC', 'GAACTTGTAG'], [np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])], "SUBSET:OK"), # 52 bp seq but no indel
    ('GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC', [0,0,0,0,0,0,0,0,0,0,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,0,0,0,0,0,0,0,0,0,0], ['TTGAGATCGT', 'GGAGCTTGTC', 'GAACTTGTAG'], [[0,0,0,0,0,0,0,0,0,0],[30,30,30,30,30,30,30,30,30,30],[0,0,0,0,0,0,0,0,0,0]], "SUBSET:OK"), # Check QS subset
    ])
    def test_chem_hydrop_subset_barcodes_perm(self, seq, qs, expected_seq, expected_qs, expected_msg):
        """Test subsetting barcodes from sequence for hydrop chemistry with variable permitations"""
        barcode_chunks = None
        qs_chunks = None

        if qs is None:
            qs = np.full(len(seq), 30)

        chemistry = ChemistryFactory.get_chemistry('hydrop')
        barcode_chunks, qs_chunks, msg = chemistry.subset_barcode_chunks(seq, qs)

        # print("")
        # print(barcode_chunks)
        # print(qs_chunks)
        # print(msg)

        if expected_seq is None:
            assert barcode_chunks is None
        else:
            for idx, bc in enumerate(barcode_chunks):
                assert bc == expected_seq[idx]

        if expected_qs is None:
            assert qs_chunks is None
        else:
            for idx, qs in enumerate(qs_chunks):
                np.testing.assert_array_equal(qs, expected_qs[idx])

        assert msg == expected_msg
