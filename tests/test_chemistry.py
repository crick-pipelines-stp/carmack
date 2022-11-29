""" 
Tests covering the chemistry module
"""

import unittest


class TestChemistry(unittest.TestCase):
    """Class for chemistry tests"""

    from .chemistry.chemistry_factory import test_class_noinit
    from .chemistry.chemistry_hydrop import test_hydrop_load_barcode_set, test_hydrop_load_barcode_set_with_factory, test_hydrop_subset_barcodes
