""" 
Tests covering the barcode module
"""

import unittest


class TestBarcode(unittest.TestCase):
    """Class for barcode tests"""

    from .barcode.whitelist import test_load_barcode_whitelist
    from .barcode.barcode_extractor import\
        test_calc_raw_barcode_match_counts_hydrop,\
        test_calc_raw_barcode_match_distribution_hydrop,\
        test_correct_barcode_md5,\
        test_get_corrected_barcode_md5


class TestBarcodeFixtures():
    """Class for barcode tests with param fixtures"""

    from .barcode.barcode_extractor import test_gen_nearby_seqs_withn,\
        test_gen_nearby_seqs_withn_none,\
        test_gen_nearby_seqs_no_n,\
        test_gen_nearby_seqs_expected,\
        test_gen_indel_set_n_in_seq,\
        test_gen_indel_set_correct_length,\
        test_gen_indel_set_deletions,\
        test_gen_indel_set_insertions,\
        test_gen_indel_set_qs_deletions,\
        test_correct_barcode_chunk,\
        test_correct_barcode_perm,\
        test_get_corrected_barcode_messages