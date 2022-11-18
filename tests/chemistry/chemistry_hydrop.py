import os

from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.chemistry_factory import ChemistryFactory

TEST_BC_1 = "TGTAGCAAGT"
TEST_BC_2 = "TTAGTTGGAC"
TEST_BC_3 = "TGACCGTACT"

def test_hydrop_load_barcode_set(self):
    """Test loading barcode set for hydrop chemistry."""
    chemistry = ChemistryHydrop()
    barcode_set = chemistry.load_barcode_set()

    self.assertEqual(barcode_set['bc1'][0], TEST_BC_1)
    self.assertEqual(barcode_set['bc2'][0], TEST_BC_2)
    self.assertEqual(barcode_set['bc3'][0], TEST_BC_3)

    self.assertEqual(len(barcode_set['bc1']), 96)
    self.assertEqual(len(barcode_set['bc2']), 96)
    self.assertEqual(len(barcode_set['bc3']), 96)

def test_hydrop_load_barcode_set_with_factory(self):
    """Test loading barcode set for hydrop chemistry."""
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()

    self.assertEqual(barcode_set['bc1'][0], TEST_BC_1)
    self.assertEqual(barcode_set['bc2'][0], TEST_BC_2)
    self.assertEqual(barcode_set['bc3'][0], TEST_BC_3)

    self.assertEqual(len(barcode_set['bc1']), 96)
    self.assertEqual(len(barcode_set['bc2']), 96)
    self.assertEqual(len(barcode_set['bc3']), 96)
