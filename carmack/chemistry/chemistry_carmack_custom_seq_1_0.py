import os
from importlib.resources import files
from itertools import product

import numpy as np

from ..io.gzip_file import GzipFile
from .chemistry_base import ChemistryBase
from carmack.barcode.barcode_utils import find_anchor_hamming

BC_LENGTH = 96
BC_CHUNK_LENGTH = 10
BC1_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath("carmack_custom_seq_1_0_96_bc1.tsv")
BC2_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath("carmack_custom_seq_1_0_96_bc2.tsv")
BC3_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath("carmack_custom_seq_1_0_96_bc3.tsv")
PRIMER_C = "TGTGTATAAGGACCTCGTTGCC"
PRIMER_A = "ATGGAAGCCGACGAATTAGACC"


class ChemistryCarmackCustomSeq10(ChemistryBase):
    """
    ChemistryCarmack class.
    """

    def load_barcode_set(self) -> list:
        """Load barcode set for chemistry."""

        stream1 = GzipFile(str(BC1_PATH)).open_read_iterator(as_string=True)
        bc1 = {line.strip() for line in stream1}
        stream1.close()

        stream2 = GzipFile(str(BC2_PATH)).open_read_iterator(as_string=True)
        bc2 = {line.strip() for line in stream2}
        stream2.close()

        stream3 = GzipFile(str(BC3_PATH)).open_read_iterator(as_string=True)
        bc3 = {line.strip() for line in stream3}
        stream3.close()

        return [bc1, bc2, bc3]

    def construct_whitelist(self, barcode_set):
        whitelist = set()

        # Create combinations of BC1 + BC2 + BC3
        for b_combination in product(*barcode_set):
            curr_wl = "".join(b_combination)
            whitelist.add(curr_wl)

        return whitelist

    def subset_whitelist_guess(self, seq: str) -> str:
        """Make best guess sequence subset based on standard chemistry for a whitelist match"""

        # Return if seq too short for chemistry
        if len(seq) < BC_LENGTH:
            return None

        # Subset seq if more than 50 to the left most 50 bases
        if len(seq) > BC_LENGTH:
            seq = seq[:BC_LENGTH]

        # Subset barcodes
        bc3 = seq[22:32]
        bc2 = seq[54:64]
        bc1 = seq[86:96]

        # Return constructed 30 base hydrop whitelist bc
        return bc1 + bc2 + bc3

    def subset_barcode_chunks(self, seq: str, qs: np.ndarray) -> list:
        """Subset barcodes from sequence for given chemistry where they are supposed to be found using locator sequences"""

        # Init
        msg = "SUBSET:OK"

        # Return nothing if the sequence is too short for hydrop chemistry
        if len(seq) < BC_LENGTH:
            return None, None, "SUBSET:SEQLEN<" + str(BC_LENGTH)

        # Subset seq if more than 96 for efficiency
        if len(seq) > BC_LENGTH:
            seq = seq[:BC_LENGTH]

        # Try to find spacer seqs
        idx_primer_c = find_anchor_hamming(seq, PRIMER_C, 2)
        idx_primer_a = find_anchor_hamming(seq, PRIMER_A, 2)

        # Error if we cant find them
        if idx_primer_c == -1:
            return None, None, "SUBSET:PRIMC_NOTFND"
        if idx_primer_a == -1:
            return None, None, "SUBSET:PRIMA_NOTFND"

        # Set message to indel if detected
        if idx_primer_c != 32:
            msg = "SUBSET:INDL"
        if idx_primer_a != 64:
            msg = "SUBSET:INDL"

        # Subset the barcodes
        bc3 = seq[idx_primer_c - BC_CHUNK_LENGTH : idx_primer_c]
        bc2 = seq[idx_primer_c + len(PRIMER_C) : idx_primer_a]
        bc1 = seq[idx_primer_a + len(PRIMER_A) :]

        # Subset the qs scores
        qs3 = qs[idx_primer_c - BC_CHUNK_LENGTH : idx_primer_c]
        qs2 = qs[idx_primer_c + len(PRIMER_C) : idx_primer_a]
        qs1 = qs[idx_primer_a + len(PRIMER_A) :]

        return [bc1, bc2, bc3], [qs1, qs2, qs3], msg
