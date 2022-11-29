from .gzip_file import GzipFile
from .subprocess_stream import SubprocessStream

GZIP_SUFFIX = ".gz"
LZ4_SUFFIX = ".lz4"


class FastqFile(GzipFile):
    """
    Class that can read/write fastq files in raw or gz format
    """

    def __init__(self, filename: str, paired_end: bool = False):
        """
        Initialise the FastqFile object
        """
        self.paired_end = paired_end
        super().__init__(filename)

    def open_read_iterator(self, as_string: bool = False):
        """
        Open a fastq file for reading. Returns a generator that yields
        """
        line_index = 0
        stream = None

        if self.compressor is not None:
            stream = SubprocessStream([self.compressor, "-c", "-d", self.filename], mode="r")
        else:
            stream = open(self.filename, "r")

        with stream as fastq_file:
            for line in fastq_file:
                if line_index == 0:
                    name1 = line.strip()[1:]
                elif line_index == 1:
                    seq1 = line.strip()
                elif line_index == 3:
                    qual1 = line.strip()
                elif line_index == 4:
                    name2 = line.strip()[1:]
                elif line_index == 5:
                    seq2 = line.strip()
                elif line_index == 7:
                    qual2 = line.strip()

                line_index += 1
                if not (self.paired_end) and line_index == 4:
                    line_index = 0

                if line_index == 8:
                    line_index = 0

                if line_index == 0:
                    if self.paired_end:
                        if as_string:
                            yield (
                                name1.decode("UTF-8"),
                                seq1.decode("UTF-8"),
                                qual1.decode("UTF-8"),
                                name2.decode("UTF-8"),
                                seq2.decode("UTF-8"),
                                qual2.decode("UTF-8"),
                            )
                        else:
                            yield (name1, seq1, qual1, name2, seq2, qual2)
                    else:
                        if as_string:
                            if self.compressor is None:
                                yield (name1, seq1, qual1)
                            else:
                                yield (name1.decode("UTF-8"), seq1.decode("UTF-8"), qual1.decode("UTF-8"))
                        else:
                            yield (name1, seq1, qual1)

    @staticmethod
    def write_read(file_stream, name: str, seq: str, qual: str) -> None:
        """Writes a single read to a fastq file"""
        file_stream.write(("@" + name + "\n").encode("UTF-8"))
        file_stream.write((seq + "\n").encode("UTF-8"))
        file_stream.write(("+\n").encode("UTF-8"))
        file_stream.write((qual + "\n").encode("UTF-8"))
