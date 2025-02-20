from typing import Dict, Generator, Literal
import os


class BedFile:
    """
    Class that can read/write bed files and its variations (handle them as regular bed files).
    """

    def __init__(self, filename: str):
        """
        Initialise the BedFile object
        """
        self.filename = filename

    def format_entry(self, entry: list) -> Dict:
        """
        Format a bed entry into a dictionary.
        """
        if len(entry) < 6:
            raise ValueError(f"BAM file must have at least 6 columns, but got {len(entry)}")

        entry_dict = {
            "chrom": entry[0],
            "start": int(entry[1]),
            "end": int(entry[2]),
            "name": entry[3],
            "score": int(entry[4]),
            "strand": entry[5],  # Literal["+", "-", "."]
        }
        # Check entries
        if not (0 <= entry_dict["score"] <= 1000):
            raise ValueError(f"Score must be in the range 0-1000, but got {entry_dict['score']}")

        return entry_dict

    def open_read_iterator(self) -> Generator[Dict, None, None]:
        """
        Open a bed file for reading. Returns a generator that yields.
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist.")

        with open(self.filename, "r") as bed_file:
            for line in bed_file:
                entry = line.strip().split("\t")
                yield self.format_entry(entry)

    def open_write_stream(self):
        """
        Open a bed file for writing. Returns a file object.
        """
        return open(self.filename, "w")

    @staticmethod
    def write_entry(
        file_stream,
        chrom: str,
        start: int,
        end: int,
        name: str,
        score: int,
        strand: Literal["+", "-", "."],
    ):
        """
        Write a bed entry to a file stream.
        """
        file_stream.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")
