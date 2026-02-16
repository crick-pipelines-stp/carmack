"""
Carmack Custom Sequencing 1.0 chemistry definition.

Read structure (5' to 3'):
BC3 (10bp) -> PRIMER_C (22bp) -> BC2 (10bp) -> PRIMER_A (22bp) -> BC1 (10bp) -> ...
"""

import logging
from importlib.resources import files

from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.gzip_file import GzipFile


log = logging.getLogger(__name__)


# Primer sequences
PRIMER_C = "TGTGTATAAGGACCTCGTTGCC"
PRIMER_A = "ATGGAAGCCGACGAATTAGACC"

BC_CHUNK_LEN = 10

# Barcode file paths
BC1_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc1.tsv"
)
BC2_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc2.tsv"
)
BC3_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc3.tsv"
)


class ChemistryCarmackCustomSeq10(ChemistryBase):
    """
    Chemistry definition for Carmack Custom Sequencing 1.0.

    This chemistry has a specific read structure with three barcode components
    and two primer sequences. Barcode whitelists are loaded from TSV files.
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this chemistry."""
        return "carmack_custom_seq_1_0"

    @property
    def read_structure(self) -> ReadStructure:
        """Define the layout of barcodes and primers within the read."""
        structure = [
            ReadComponent(name="PRIMER_D", is_barcode=False, length=len(PRIMER_C)),
            ReadComponent(name="BC3", is_barcode=True, length=BC_CHUNK_LEN),
            ReadComponent(
                name="PRIMER_C", is_barcode=False, length=len(PRIMER_C), sequence=PRIMER_C
            ),
            ReadComponent(name="BC2", is_barcode=True, length=BC_CHUNK_LEN),
            ReadComponent(
                name="PRIMER_A", is_barcode=False, length=len(PRIMER_A), sequence=PRIMER_A
            ),
            ReadComponent(name="BC1", is_barcode=True, length=BC_CHUNK_LEN),
        ]

        return ReadStructure(structure)

    @property
    def max_errors(self) -> MatchErrors:
        """Return the maximum allowed errors for barcode matching."""
        return MatchErrors(barcode=1, spacer=2)

    def load_barcode_whitelist(self, barcode_name: str) -> tuple[str, ...]:
        """
        Load the barcode whitelist for a specific barcode component.

        Args:
            barcode_name: The name of the barcode component ("BC1", "BC2", or "BC3")
                Must match the names defined in get_read_structure() for barcode components.

        Returns:
            Tuple of valid barcode sequences

        Raises:
            ValueError: If the barcode name is not recognized
        """
        path_map = {
            "BC1": BC1_PATH,
            "BC2": BC2_PATH,
            "BC3": BC3_PATH,
        }

        if barcode_name not in path_map:
            raise ValueError(
                f"Unknown barcode name: {barcode_name}. Valid names: {list(path_map.keys())}"
            )

        path = path_map[barcode_name]
        log.debug(f"Loading barcode whitelist for {barcode_name} from {path}")

        barcodes = []
        stream = GzipFile(str(path)).open_read_iterator(as_string=True)
        for line in stream:
            barcode = line.strip()
            if barcode:
                barcodes.append(barcode)
        stream.close()

        result = tuple(barcodes)
        log.debug(f"Loaded {len(result)} barcodes for {barcode_name}")
        return result


# Register this chemistry with the factory
ChemistryFactory.register(ChemistryCarmackCustomSeq10)
