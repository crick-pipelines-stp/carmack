import sys
import os
import subprocess
import lib.log_subprocess as tk_subproc
import lib.constants as const


#
# FUNCTION
#
def open_maybe_gzip(filename, mode="r"):
    """Returns an open filehandle to a file that may or may not be compressed with gzip or lz4.
    """
    compressor = None

    if filename.endswith(const.GZIP_SUFFIX):
        compressor = "gzip"
    elif filename.endswith(const.LZ4_SUFFIX):
        compressor = "lz4"
    else:
        return open(filename, mode)

    assert compressor is not None

    if mode == "r":
        return SubprocessStream[compressor, "-c", "-d", filename], mode="r")
    elif mode == "w":
        f = open(filename, "w")
        return SubprocessStream([compressor, "-c"], stdout=f, mode="w")
    else:
        raise ValueError("Unsupported mode for compression: %s" % mode)