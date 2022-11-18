import os

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

@with_temporary_folder
def test_hydrop_subset_barcodes(self, temp_path):
    """Test subsetting barcodes from sequence for hydrop chemistry."""

    expected_hash = 'a36c5ce249da5d081a469017700c2066'

    test_file = os.path.join(temp_path, 'barcodes.txt')
    fq_file = FastqFile(BC_READS_PATH)
    stream = fq_file.open_read_iterator(as_string=True)
    chemistry = ChemistryFactory.get_chemistry('hydrop')

    with open(test_file, 'w') as out_file:
        for (name, seq, qual) in stream:
            barcodes = chemistry.subset_barcodes(seq)
            line = ','.join(barcodes)
            out_file.write(line+ '\n')
    
    utils.validate_file_md5(test_file, expected_hash)
