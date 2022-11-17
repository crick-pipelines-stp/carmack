""" 
Tests covering the io module
"""

import unittest


class TestIo(unittest.TestCase):
    """Class for io tests"""

    ############################################
    # Test of the individual io commands. #
    ############################################

    from .io.log_subprocess import test_log_subprocess

    from .io.subprocess_stream import test_subprocess_stream_gzip_read, test_subprocess_stream_gzip_write

    from .io.fastq_file import test_fastq_file_gzip_read, test_fastq_file_raw_read, test_subprocess_stream_gzip_write

