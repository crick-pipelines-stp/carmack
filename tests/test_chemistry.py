""" 
Tests covering the chemistry module
"""

import unittest


class TestChemistry(unittest.TestCase):
    """Class for chemistry tests"""

    from .chemistry.chemistry_factory import test_class_noinit
    from .chemistry.chemistry_hydrop import\
        test_hydrop_load_barcode_set,\
        test_hydrop_load_barcode_set_with_factory,\
        test_hydrop_construct_whitelist_model_case,\
        test_hydrop_construct_whitelist_md5,\
        test_hydrop_subset_barcodes_md5


class TestBarcodeFixtures():
    """Class for barcode tests with fixtures"""

    from .chemistry.chemistry_hydrop import\
        test_hydrop_subset_barcodes_perm,\
        test_subset_whitelist_guess_perm