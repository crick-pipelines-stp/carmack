"""
Base class for chemistry definitions.

Each chemistry defines the structure of barcodes within reads, including:
- Barcode positions and lengths
- Primer sequences for anchor detection
- Whitelist loading from data files
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from carmack.chemistry.read_structure import ReadStructure


@dataclass(frozen=True)
class MatchErrors:
    """
    Define maximum edit distances for barcodes and spacers.
    """

    barcode: int
    spacer: int


@dataclass
class ChemistryBase(ABC):
    """
    Abstract base class for chemistry definitions.

    Subclasses must implement methods to define barcode layouts,
    load barcode whitelists, and provide chemistry-specific parameters.
    """

    @cached_property
    @abstractmethod
    def name(self) -> str:
        """Name of the chemistry."""
        pass

    @cached_property
    @abstractmethod
    def read_structure(self) -> ReadStructure:
        """Defines the layout of barcodes and other components within reads."""
        pass

    @cached_property
    @abstractmethod
    def max_errors(self) -> MatchErrors:
        """Maximum allowed errors (substitutions/indels) for barcode matching."""
        pass

    @abstractmethod
    def load_barcode_whitelist(self, barcode_name: str) -> tuple[str, ...]:
        """
        Load the barcode whitelist for a specific barcode component.

        Args:
            barcode_name: The name of the barcode component (e.g., "BC1")

        Returns:
            Tuple of valid barcode sequences
        """
        pass

    @cached_property
    def barcode_whitelists(self) -> dict[str, tuple[str, ...]]:
        """
        Load whitelists for all barcode components defined in the read structure.

        Returns:
            Dictionary mapping barcode component names to their whitelist tuples.
        """
        whitelists = {}
        for component in self.read_structure:
            if component.is_barcode:
                try:
                    whitelists[component.name] = self.load_barcode_whitelist(component.name)
                except Exception as e:
                    raise KeyError(
                        f"Barcode layout '{component.name}' not found in whitelist: {e}"
                    ) from e

        # Check if all barcodes defined in the read structure have corresponding whitelists
        missing = [
            comp.name
            for comp in self.read_structure
            if comp.is_barcode and comp.name not in whitelists
        ]
        if missing:
            raise KeyError(f"Missing whitelists for barcode components: {missing}")

        return whitelists

    def construct_full_barcode(self, barcodes: dict[str, str]) -> str:
        """
        Construct the full barcode sequence by concatenating individual
        barcode components in the correct order. Order is determined by the read structure.
        """
        read_layout = self.read_structure
        barcode_components = [comp for comp in read_layout if comp.is_barcode]

        try:
            return "".join(barcodes[comp.name] for comp in barcode_components)
        except KeyError as e:
            raise ValueError(
                f"Missing barcode component '{e.args[0]}' for full barcode construction."
            ) from e

    def __post_init__(self):
        # Validate that all barcode components defined in the read structure have whitelists
        for component in self.read_structure:
            if component.is_barcode:
                try:
                    self.load_barcode_whitelist(component.name)
                except Exception as e:
                    raise KeyError(
                        f"Barcode layout '{component.name}' not found in whitelist: {e}"
                    ) from e

        # Validate that all spacers defined in the read structure are present in the spacers dictionary
        spacer_names = {comp.name for comp in self.read_structure if not comp.is_barcode}
        spacer_seqs = self.read_structure.get_known_sequences()
        missing_spacers = set(spacer_seqs.keys()) - spacer_names

        if not spacer_names.issuperset(spacer_seqs.keys()):
            raise KeyError(
                f"Spacers defined in the chemistry must be present in the read structure. Missing spacers: {missing_spacers}"
            )
