from .subprocess_stream import SubprocessStream

GZIP_SUFFIX = ".gz"
LZ4_SUFFIX = ".lz4"

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

        if filename.endswith(GZIP_SUFFIX):
            self.compressor = "gzip"
        elif filename.endswith(LZ4_SUFFIX):
            self.compressor = "lz4"

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
        Open a fastq file for writing. Returns a stream that can be written to
        """
        f = open(self.filename, "w")
        return SubprocessStream([self.compressor, "-c"], mode="w", stdout=f)

    @staticmethod
    def write_string(file_stream, line: str) -> None:
        """Writes a single line to a gzip file"""
        file_stream.write(line.encode("UTF-8"))
