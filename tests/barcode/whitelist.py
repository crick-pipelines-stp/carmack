import os

from carmack.barcode.whitelist import load_barcode_whitelist

TEST_BC = "GCAGTAGCTGTGTAGCAAGTGTACTCTGCG"

def test_load_barcode_whitelist(self):
    """Test loading lists of barcodes"""
    whitelist = load_barcode_whitelist("carmack/resources/data/barcodes/hydrop_whitelist_bc1_96.tsv")

    for bc in whitelist:
        self.assertEqual(bc, TEST_BC)
        break

    self.assertEqual(len(whitelist), 96)

