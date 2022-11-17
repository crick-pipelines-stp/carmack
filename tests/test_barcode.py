""" 
Tests covering the barcode module
"""

import unittest


class TestIo(unittest.TestCase):
    """Class for barcode tests"""

    from .barcode.whitelist import test_load_barcode_whitelist
