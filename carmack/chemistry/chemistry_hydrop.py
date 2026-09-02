"""
Hydrop chemistry definition.

Read structure (5' to 3'):
BC3 (10bp) -> SPACER_1 (10bp) -> BC2 (10bp) -> SPACER_2 (10bp) -> BC1 (10bp) -> ...
"""

from functools import cached_property
from importlib.resources import files

from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors, WhitelistSource
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure

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
            ReadComponent(name="BC3", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(
                name="SPACER_1",
                type=ReadComponentType.OTHER,
                length=len(SPACER_1),
                sequence=SPACER_1,
            ),
            ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(
                name="SPACER_2",
                type=ReadComponentType.OTHER,
                length=len(SPACER_2),
                sequence=SPACER_2,
            ),
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
        ]

        return ReadStructure(structure)

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return the maximum allowed errors for barcode matching."""
        return MatchErrors(barcode=2, spacer=1)

    def whitelist_sources(self) -> dict[str, WhitelistSource]:
        """Return the packaged whitelist file for each barcode component.

        Returns:
            Dictionary mapping barcode component names to their whitelist
            sources. Hydrop ships each barcode flanked by padding, so every
            source declares the slice that trims it back to the barcode.
        """
        return {
            "BC1": WhitelistSource(path=BC1_PATH, line_slice=(10, -10)),
            "BC2": WhitelistSource(path=BC2_PATH, line_slice=(10, -10)),
            "BC3": WhitelistSource(path=BC3_PATH, line_slice=(15, -10)),
        }


# Register this chemistry with the factory
ChemistryFactory.register(ChemistryHydrop)
