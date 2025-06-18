# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest

from carmack.io.log_subprocess import LogSubprocess


class TestLogSubprocess(unittest.TestCase):
    def test_log_subprocess(self):
        """Check class init with default values"""
        log_subprocess = LogSubprocess()
        assert log_subprocess is not None
