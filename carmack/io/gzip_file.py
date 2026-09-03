import logging
from functools import cache
from os import cpu_count
from shutil import which

from .subprocess_stream import SubprocessStream

log = logging.getLogger(__name__)

GZIP_SUFFIX = ".gz"
LZ4_SUFFIX = ".lz4"

# pigz parallelises deflate and is a drop-in for gzip on the command line. Four threads is
# roughly 4.8x gzip's throughput, which puts compression off the critical path for every
# stage, while staying small enough that a stage holding several output streams open at once
# cannot saturate the machine on compression alone.
COMPRESSION_THREADS = 4


@cache
def resolve_gzip_write_command() -> tuple[str, ...]:
    """
    Resolve the command used to compress gzip output, preferring pigz over gzip.

    The result is memoised, so PATH is searched once per process and a fallback is reported
    once rather than once per output file.

    Returns:
        The command and its arguments, reading plaintext on stdin and writing the compressed
        stream on stdout.
    """
    if which("pigz") is None:
        log.warning("pigz not found on PATH, falling back to single-threaded gzip for output.")
        return ("gzip", "-c")

    threads = min(COMPRESSION_THREADS, cpu_count() or 1)
    log.info(f"Compressing output with pigz using {threads} threads.")
    return ("pigz", "-c", "-p", str(threads))


class GzipFile:
    """
    Class that can read/write gzip files
    """

    def __init__(self, filename):
        """
        Initialise the GzipFile object
        """
        self.filename = filename
        self.compressor = None
        self.write_command = None

        # Writing prefers pigz, reading does not: gzip's format is not parallel-decodable,
        # so a parallel decompressor has nothing to win on the read side.
        if filename.endswith(GZIP_SUFFIX):
            self.compressor = "gzip"
            self.write_command = resolve_gzip_write_command()
        elif filename.endswith(LZ4_SUFFIX):
            self.compressor = "lz4"
            self.write_command = (self.compressor, "-c")

    def open_read_iterator(self, as_string: bool = False):
        """
        Open a gzip file for reading. Returns a generator that yields
        """
        stream = None

        if self.compressor is not None:
            stream = SubprocessStream([self.compressor, "-c", "-d", self.filename], mode="r")
        else:
            stream = open(self.filename, "r")

        with stream as file:
            for line in file:
                if as_string:
                    if self.compressor is None:
                        yield line.strip()
                    else:
                        yield line.decode("UTF-8").strip()
                else:
                    yield line

    def open_write_stream(self):
        """
        Open a gzip file for writing. Returns a stream that can be written to
        """
        f = open(self.filename, "w")
        return SubprocessStream(self.write_command, mode="w", stdout=f)

    @staticmethod
    def write_string(file_stream, line: str) -> None:
        """Writes a single line to a gzip file"""
        file_stream.write(line.encode("UTF-8"))
