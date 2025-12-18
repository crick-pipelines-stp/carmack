from importlib.resources import files
from itertools import product

import numpy as np

from carmack.io.gzip_file import GzipFile
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.barcode.barcode_utils import get_best_barcode

BC_LENGTH = 96
BC_CHUNK_LENGTH = 10
BC1_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc1.tsv"
)
BC2_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc2.tsv"
)
BC3_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc3.tsv"
)
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

    def construct_whitelist(self, barcode_set) -> set:
        whitelist = set()

        # Create combinations of BC1 + BC2 + BC3
        for b_combination in product(*barcode_set):
            curr_wl = "".join(b_combination)
            whitelist.add(curr_wl)

        return whitelist

    def subset_whitelist_guess(self, seq: str) -> str | None:
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

    def subset_barcode_chunks(self, seq: str, qs: np.ndarray) -> tuple:
        """Subset barcodes from sequence for given chemistry where they are supposed to be found using local alignment"""

        # Init
        barcode_set = self.load_barcode_set()
        max_corrections = 1
        msg = "SUBSET:OK"

        # Return nothing if the sequence is too short for this chemistry
        if len(seq) < BC_LENGTH:
            return None, None, "SUBSET:SEQLEN<" + str(BC_LENGTH)

        # Find BC3 (leftmost barcode) in full sequence
        bc3_coords = get_best_barcode(seq, barcode_set[2], max_corrections=max_corrections)
        if bc3_coords[0] is None:
            return None, None, "SUBSET:" + bc3_coords[1] + "3"

        # Find BC2 in sequence after BC3
        bc3_end = bc3_coords[0][1]
        search_start_bc2 = bc3_end + 1

        if search_start_bc2 + BC_CHUNK_LENGTH * 2 > len(seq):
            return None, None, "SUBSET:SEQSHORT2"

        bc2_coords = get_best_barcode(
            seq[search_start_bc2:], barcode_set[1], max_corrections=max_corrections
        )
        if bc2_coords[0] is None:
            return None, None, "SUBSET:" + bc2_coords[1] + "2"

        # Adjust BC2 coordinates to absolute positions in original sequence
        bc2_start = bc2_coords[0][0] + search_start_bc2
        bc2_end = bc2_coords[0][1] + search_start_bc2

        # Find BC1 in sequence after BC2
        search_start_bc1 = bc2_end + 1

        if search_start_bc1 + BC_CHUNK_LENGTH > len(seq):
            return None, None, "SUBSET:SEQSHORT1"

        bc1_coords = get_best_barcode(
            seq[search_start_bc1:], barcode_set[0], max_corrections=max_corrections
        )
        if bc1_coords[0] is None:
            return None, None, "SUBSET:" + bc1_coords[1] + "1"

        # Adjust BC1 coordinates to absolute positions in original sequence
        bc1_start = bc1_coords[0][0] + search_start_bc1
        bc1_end = bc1_coords[0][1] + search_start_bc1

        # Extract barcodes using absolute coordinates
        bc3 = seq[bc3_coords[0][0] : bc3_coords[0][1]]
        bc2 = seq[bc2_start:bc2_end]
        bc1 = seq[bc1_start:bc1_end]

        # Extract quality scores using absolute coordinates
        qs3 = qs[bc3_coords[0][0] : bc3_coords[0][1]]
        qs2 = qs[bc2_start:bc2_end]
        qs1 = qs[bc1_start:bc1_end]

        return [bc1, bc2, bc3], [qs1, qs2, qs3], msg
