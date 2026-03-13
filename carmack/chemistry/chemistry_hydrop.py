"""
Hydrop chemistry definition.

Read structure (5' to 3'):
BC3 (10bp) -> SPACER_1 (10bp) -> BC2 (10bp) -> SPACER_2 (10bp) -> BC1 (10bp) -> ...
"""

import logging
from functools import cached_property
from importlib.resources import files

from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.gzip_file import GzipFile


log = logging.getLogger(__name__)


# Spacer sequences
SPACER_1 = "AGGGTACTCG"
SPACER_2 = "GCAGTAGCTG"

# Barcode file paths
BC1_PATH = files("carmack.data.barcodes.hydrop").joinpath("hydrop_whitelist_bc1_96.tsv")
BC2_PATH = files("carmack.data.barcodes.hydrop").joinpath("hydrop_whitelist_bc2_96.tsv")
BC3_PATH = files("carmack.data.barcodes.hydrop").joinpath("hydrop_whitelist_bc3_96.tsv")


class ChemistryHydrop(ChemistryBase):
    """
    Chemistry definition for Hydrop.

    This chemistry has a specific read structure with three barcode components
    and two spacer sequences. Barcode whitelists are loaded from TSV files.
    """

    @cached_property
    def name(self) -> str:
        """Return the unique identifier for this chemistry."""
        return "hydrop"

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Define the layout of barcodes and spacers within the read."""
        structure = [
            ReadComponent(name="BC3", is_barcode=True, length=10),
            ReadComponent(
                name="SPACER_1", is_barcode=False, length=len(SPACER_1), sequence=SPACER_1
            ),
            ReadComponent(name="BC2", is_barcode=True, length=10),
            ReadComponent(
                name="SPACER_2", is_barcode=False, length=len(SPACER_2), sequence=SPACER_2
            ),
            ReadComponent(name="BC1", is_barcode=True, length=10),
        ]

        return ReadStructure(structure)

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return the maximum allowed errors for barcode matching."""
        return MatchErrors(barcode=2, spacer=1)

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

        bc_strip_map = {"BC1": (10, -10), "BC2": (10, -10), "BC3": (15, -10)}

        if barcode_name not in path_map:
            raise ValueError(
                f"Unknown barcode name: {barcode_name}. Valid names: {list(path_map.keys())}"
            )

        path = path_map[barcode_name]
        strip_idx = bc_strip_map[barcode_name]
        log.debug(f"Loading barcode whitelist for {barcode_name} from {path}")

        barcodes = []
        stream = GzipFile(str(path)).open_read_iterator(as_string=True)
        for line in stream:
            barcode = line.strip()[strip_idx[0] : strip_idx[1]]
            if barcode:
                barcodes.append(barcode)
        stream.close()

        result = tuple(barcodes)
        log.debug(f"Loaded {len(result)} barcodes for {barcode_name}")
        return result


# Register this chemistry with the factory
ChemistryFactory.register(ChemistryHydrop)
