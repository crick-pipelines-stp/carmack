import os
import numpy as np
from importlib.resources import files
from itertools import product

from .chemistry_base import ChemistryBase
from ..io.gzip_file import GzipFile

BC1_PATH = files('carmack.data.barcodes').joinpath('hydrop_whitelist_bc1_96.tsv')
BC2_PATH = files('carmack.data.barcodes').joinpath('hydrop_whitelist_bc2_96.tsv')
BC3_PATH = files('carmack.data.barcodes').joinpath('hydrop_whitelist_bc3_96.tsv')

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

        stream1 = GzipFile(str(BC1_PATH)).open_read_iterator(as_string=True)
        bc1 = {line.strip()[10:-10] for line in stream1}
        stream1.close()

        stream2 = GzipFile(str(BC2_PATH)).open_read_iterator(as_string=True)
        bc2 = {line.strip()[10:-10] for line in stream2}
        stream2.close()

        stream3 = GzipFile(str(BC3_PATH)).open_read_iterator(as_string=True)
        bc3 = {line.strip()[15:-10] for line in stream3}
        stream3.close()

        return [ bc1, bc2, bc3 ]

    def construct_whitelist(self, barcode_set):
        whitelist = set()

        for b_combination in product(*barcode_set):
            curr_wl = ''.join(b_combination)
            whitelist.add(curr_wl)

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

        # Init
        msg = "SUBSET:OK"

        # Return nothing if the sequence is too short for hydrop chemistry
        if(len(seq) < 50):
            return None, None, "SUBSET:SEQLEN<50"

        # Subset seq if more than 50 to the left most 50 bases
        if(len(seq) > 50):
            seq = seq[:50]

        # Try to find spacer seqs
        idx_spcr_1 = seq.find(self.SPACER_1)
        idx_spcr_2 = seq.find(self.SPACER_2)

        # Error if we cant find them
        if idx_spcr_1 == -1:
            return None, None, "SUBSET:SPC1_NOTFND"
        if idx_spcr_2 == -1:
            return None, None, "SUBSET:SPC2_NOTFND"

        # Set message to indel if detected
        if idx_spcr_1 != 10:
            msg = "SUBSET:INDL"
        if idx_spcr_2 != 30:
            msg = "SUBSET:INDL"

        # Subset the barcodes
        bc1 = seq[idx_spcr_2+10:]
        bc2 = seq[idx_spcr_1+10:(-50 + idx_spcr_2)]
        bc3 = seq[:(-50 + idx_spcr_1)]

        # Subset the qs scores
        qs1 = qs[idx_spcr_2+10:]
        qs2 = qs[idx_spcr_1+10:(-50 + idx_spcr_2)]
        qs3 = qs[:(-50 + idx_spcr_1)]

        return [ bc1, bc2, bc3 ], [ qs1, qs2, qs3 ], msg
