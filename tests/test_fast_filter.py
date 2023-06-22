""" 
Tests covering the fastq filter module
"""

import unittest


class TestFastqFilter(unittest.TestCase):
    """Class for fastq filter test"""

    from .fastq_tools.fastq_filter import test_filter_valid_reads
        