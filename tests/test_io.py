""" 
Tests covering the io module
"""

import unittest

class TestIo(unittest.TestCase):
    """Class for io tests"""

    ############################################
    # Test of the individual io commands. #
    ############################################

    from .io.log_subprocess import (
        test_log_subprocess
    )
