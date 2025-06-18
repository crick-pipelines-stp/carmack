# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest

from carmack.barcode.whitelist import load_barcode_whitelist


TEST_BC = "GCAGTAGCTGTGTAGCAAGTGTACTCTGCG"


class TestBarcodeWhitelist(unittest.TestCase):
    def test_load_barcode_whitelist(self):
        """Test loading lists of barcodes"""
        whitelist = load_barcode_whitelist(
            "carmack/data/barcodes/hydrop/hydrop_whitelist_bc1_96.tsv"
        )

        for bc in whitelist:
            self.assertEqual(bc, TEST_BC)
            break

        self.assertEqual(len(whitelist), 96)
