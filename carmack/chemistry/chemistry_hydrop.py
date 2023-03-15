import os
import numpy as np

from .chemistry_base import ChemistryBase
from ..io.gzip_file import GzipFile

BC1_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc1_96.tsv'
BC2_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc2_96.tsv'
BC3_PATH = 'carmack/resources/data/barcodes/hydrop_whitelist_bc3_96.tsv'

class ChemistryHydrop(ChemistryBase):
    """
    ChemistryHydrop class.
    """

    SPACER_1 = 'AGGGTACTCG'
    SPACER_2 = 'GCAGTAGCTG'

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
    
    def construct_whitelist(self, barcode_set):
        whitelist = []

        for idx, bc in enumerate(barcode_set[0]):
            curr_wl = bc + barcode_set[1][idx] + barcode_set[2][idx]
            whitelist.append(curr_wl)

        return whitelist
    
    def subset_whitelist_guess(self, seq: str) -> str:
        """Make best guess sequence subset based on standard hydrop chemistry for a whitelist match"""

        # Return if seq too short for hydrop chemistry
        if(len(seq) < 50):
            return None
        
        # Subset seq if more than 50 to the left most 50 bases
        if(len(seq) > 50):
            seq = seq[:50]

        # Subset barcodes
        bc3 = seq[:10]
        bc2 = seq[20:30]
        bc1 = seq[40:50]

        # Return constructed 30 base hydrop whitelist bc
        return bc1 + bc2 + bc3

    def subset_barcode_chunks(self, seq: str, qs: np.ndarray) -> list:
        """Subset barcodes from sequence for given chemistry where they are supposed to be found using locator sequences"""

        # Return nothing if the sequence is too short for hydrop chemistry
        if(len(seq) < 50):
            return None, None, "SUBSET:SEQLEN<50"
        
        # Return nothing if the sequence is too short for hydrop chemistry
        if(len(seq) < 50):
            return None, None, "SUBSET:SEQLEN<50"

        #             idx_rep_seq1 = seq.find(rep_seq_1)
        #     idx_rep_seq2 = seq.find(rep_seq_2)

        # # Subset sequences
        # bc3 = seq[:-42]
        # bc2 = seq[20:-22]
        # bc1 = seq[40:-2]

        # # Subset qs
        # qs1 = qs[:-42]
        # qs2 = qs[20:-22]
        # qs3 = qs[40:-2]

        # return [ bc1, bc2, bc3 ], [ qs1, qs2, qs3 ]
        return None, None
