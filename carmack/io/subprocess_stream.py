import logging
import os
import subprocess
import sys

from .log_subprocess import LogSubprocess


log = logging.getLogger(__name__)


class SubprocessStream(object):
    """
    Wrap a subprocess that we stream from or stream to. Acts like an open filehandle by passing down
    next, fileno, write, and close down to its pipe.
    """

    def __init__(self, *args, **kwargs):
        mode = kwargs.pop("mode", "r")
        if mode == "r":
            kwargs["stdout"] = subprocess.PIPE
        elif mode == "w":
            kwargs["stdin"] = subprocess.PIPE
        else:
            raise ValueError("mode %s unsupported" % self.mode)

        kwargs["preexec_fn"] = os.setsid
        sys.stdout.flush()
        sub_proc = LogSubprocess()
        self.proc = sub_proc.Popen(*args, **kwargs)

        if mode == "r":
            self.pipe = self.proc.stdout
        elif mode == "w":
            self.pipe = self.proc.stdin

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.pipe.__next__()

    def fileno(self):
        return self.pipe.fileno()

    def write(self, x):
        self.pipe.write(x)

    def close(self):
        self.pipe.close()
        self.proc.wait()

    def __exit__(self, tp, val, tb):
        self.close()
