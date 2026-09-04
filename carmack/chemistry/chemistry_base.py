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

from carmack.chemistry.read_component import ReadComponent, ReadComponentType
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
            if component.type is ReadComponentType.BARCODE:
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
            if comp.type is ReadComponentType.BARCODE and comp.name not in whitelists
        ]
        if missing:
            raise KeyError(f"Missing whitelists for barcode components: {missing}")

        return whitelists

    def umi_component(self) -> ReadComponent | None:
        """Return the UMI component of the read structure, if one is defined.

        Returns:
            The single ``UMI`` :class:`ReadComponent`, or ``None`` for
            barcode-only chemistries that carry no UMI.
        """
        return next(
            (comp for comp in self.read_structure if comp.type is ReadComponentType.UMI),
            None,
        )

    def umi_left_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 5' of the UMI.

        The left anchor is the component preceding the UMI in the read
        structure, but only when that neighbour exists and can anchor a
        variable-length component (:attr:`ReadComponent.is_anchor`). For
        ``custom_seq_1_0`` this is the ``BC1`` barcode.

        Returns:
            The anchoring :class:`ReadComponent`, or ``None`` when there is no
            UMI or its left neighbour cannot anchor it.
        """
        umi = self.umi_component()
        if umi is None:
            return None
        previous = self.read_structure.get_previous(umi)
        if previous is not None and previous.is_anchor:
            return previous
        return None

    def umi_right_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 3' of the UMI.

        The right anchor is the component following the UMI in the read
        structure, but only when that neighbour exists and can anchor a
        variable-length component (:attr:`ReadComponent.is_anchor`). For
        ``custom_seq_1_0`` this is the ``POLYG`` homopolymer.

        Returns:
            The anchoring :class:`ReadComponent`, or ``None`` when there is no
            UMI or its right neighbour cannot anchor it.
        """
        umi = self.umi_component()
        if umi is None:
            return None
        following = self.read_structure.get_next(umi)
        if following is not None and following.is_anchor:
            return following
        return None

    def supports_umi_extraction(self) -> bool:
        """Return whether this chemistry can support UMI extraction.

        UMI extraction requires a UMI component whose left neighbour exists and
        can anchor it. Barcode-only chemistries without a UMI (e.g. hydrop)
        return ``False`` rather than raising.

        Returns:
            ``True`` when the read structure contains a UMI anchored on its 5'
            side, ``False`` otherwise.
        """
        return self.umi_left_anchor() is not None

    def construct_full_barcode(self, barcodes: dict[str, str]) -> str:
        """
        Construct the full barcode sequence by concatenating individual
        barcode components in the correct order. Order is determined by the read structure.
        """
        read_layout = self.read_structure
        barcode_components = read_layout.get_components_by_type(ReadComponentType.BARCODE)

        try:
            return "".join(barcodes[comp.name] for comp in barcode_components)
        except KeyError as e:
            raise ValueError(
                f"Missing barcode component '{e.args[0]}' for full barcode construction."
            ) from e

    def __post_init__(self):
        # Validate that all barcode components defined in the read structure have whitelists
        for component in self.read_structure:
            if component.type is ReadComponentType.BARCODE:
                try:
                    self.load_barcode_whitelist(component.name)
                except Exception as e:
                    raise KeyError(
                        f"Barcode layout '{component.name}' not found in whitelist: {e}"
                    ) from e

        # Validate that all spacers defined in the read structure are present in the spacers dictionary
        spacer_names = {
            comp.name for comp in self.read_structure if comp.type is not ReadComponentType.BARCODE
        }
        spacer_seqs = self.read_structure.get_known_sequences()
        missing_spacers = set(spacer_seqs.keys()) - spacer_names

        if not spacer_names.issuperset(spacer_seqs.keys()):
            raise KeyError(
                f"Spacers defined in the chemistry must be present in the read structure. Missing spacers: {missing_spacers}"
            )
