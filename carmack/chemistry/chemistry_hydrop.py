import os

from .chemistry_base import ChemistryBase
from ..io.gzip_file import GzipFile

BC1_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc1_96.tsv'
BC2_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc2_96.tsv'
BC3_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc3_96.tsv'

class ChemistryHydrop(ChemistryBase):
    """
    ChemistryHydrop class.
    """

    def __init__(self):
        """Initialize the ChemistryHydrop class."""
        super().__init__()

    def load_barcode_set(self) -> list:
        """Load barcode set for chemistry."""

        bc1_path = os.path.abspath(BC1_PATH)
        bc2_path = os.path.abspath(BC2_PATH)
        bc3_path = os.path.abspath(BC3_PATH)

        stream1 = GzipFile(bc1_path).open_read_iterator(as_string=True)
        bc1 = [line.strip()[10:-10] for line in stream1]
        stream1.close()

        stream2 = GzipFile(bc2_path).open_read_iterator(as_string=True)
        bc2 = [line.strip()[10:-10] for line in stream2]
        stream2.close()

        stream3 = GzipFile(bc3_path).open_read_iterator(as_string=True)
        bc3 = [line.strip()[15:-10] for line in stream3]
        stream3.close()

        return [ bc1, bc2, bc3 ]

    def subset_barcodes(self, seq: str) -> list:
        """Subset barcodes from sequence for given chemistry where they are supposed to be found."""

        assert len(seq) >= 50, 'Sequence length must be at least 50 bases.'

        # Subset sequences
        bc3 = seq[:-42]
        bc2 = seq[20:-22]
        bc1 = seq[40:-2]

        return [ bc1, bc2, bc3 ]
