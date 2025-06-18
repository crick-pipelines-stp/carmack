import sys
import logging
import subprocess

import ctypes
import ctypes.util
from signal import SIGKILL

log = logging.getLogger(__name__)

LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
PR_SET_PDEATHSIG = ctypes.c_int(1)  # <sys/prctl.h>


class LogSubprocess:
    """
    Helper class with functions for logging calls to subprocesses.
    """

    def __init__(self):
        """
        Initialise the LogSubprocess object
        """
        self.pdeathsig = self._child_preexec_set_pdeathsig()

    def _child_preexec_set_pdeathsig(self):
        """
        When used as the preexec_fn argument for subprocess.Popen etc,
        causes the subprocess to recieve SIGKILL if the parent process
        terminates.
        """
        if sys.platform.startswith("linux"):
            zero = ctypes.c_ulong(0)
            return LIBC.prctl(PR_SET_PDEATHSIG, ctypes.c_ulong(SIGKILL), zero, zero, zero)
        else:
            return None

    def check_call(self, *args, **kwargs):
        if not "preexec_fn" in kwargs:
            kwargs["preexec_fn"] = self.pdeathsig
        return subprocess.check_call(*args, **kwargs)

    def check_output(self, *args, **kwargs):
        if not "preexec_fn" in kwargs:
            kwargs["preexec_fn"] = self.pdeathsig
        return subprocess.check_output(*args, **kwargs)

    def call(self, *args, **kwargs):
        if not "preexec_fn" in kwargs:
            kwargs["preexec_fn"] = self.pdeathsig
        return subprocess.call(*args, **kwargs)

    def Popen(self, *args, **kwargs):
        if not "preexec_fn" in kwargs:
            kwargs["preexec_fn"] = self.pdeathsig
        return subprocess.Popen(*args, **kwargs)
