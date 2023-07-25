""" 
Tests covering the duplicate removal module
"""

import unittest

class TestDuplicateRemoval(unittest.TestCase):
    """Class for fastq filter test"""

    from .duplicate_removal.remove_duplicates import test_bam_file_contents