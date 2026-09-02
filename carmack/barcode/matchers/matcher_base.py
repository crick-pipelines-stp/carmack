import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_component import ReadComponent, ReadComponentType

log = logging.getLogger(__name__)


class UnresolvedComponentStartError(ValueError):
    """
    Raised when an operation needs a fixed start position the component does not have.

    ``ReadStructure.compute_start_positions`` can only resolve offsets while every preceding
    component has a known length. Once a variable-length component is reached, ``start`` is left
    as ``None`` for that component and for every component following it. A matcher may therefore
    hold a component with no fixed start, so helpers that assume one must say so explicitly
    rather than fail on arithmetic against ``None``.

    Subclasses ``ValueError`` because an unresolved start is a bad-value condition, which also
    keeps any existing broad ``except ValueError`` handling working unchanged.
    """


class MatcherBase(ABC):
    """
    Abstract base class for barcode matchers.

    Subclasses must implement the match method to perform barcode matching
    based on the provided whitelist and read structure.

    Attributes:
        allowed_component_types: The component types this matcher can structurally handle.
            It is a class attribute rather than a constructor argument deliberately, so that a
            caller cannot widen it by accident; a subclass that can genuinely handle another
            component type overrides it.
    """

    allowed_component_types: ClassVar[frozenset[ReadComponentType]] = frozenset(
        {ReadComponentType.BARCODE}
    )

    def __init__(
        self,
        whitelist: tuple[str, ...],
        component: ReadComponent,
        chemistry: ChemistryBase,
    ):
        self.whitelist_set = frozenset(whitelist)
        self.component = component
        self.chemistry = chemistry

        # Check the component is a type this matcher class declares it can handle
        allowed = type(self).allowed_component_types
        if self.component.type not in allowed:
            permitted = ", ".join(sorted(allowed))
            raise ValueError(
                f"{type(self).__name__} cannot handle component {self.component} of type "
                f"{self.component.type}; permitted component types: {permitted}"
            )

    @abstractmethod
    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read.

        Args:
            read: The sequencing read to match against.
            start_idx: Bounds where in the read this matcher looks, as a hint from the caller that
                everything before it has already been consumed. *What* it bounds is
                matcher-specific and is documented on each implementation: for some it is a hard
                floor on where a match may begin, for others only a floor on where the search is
                seeded, in which case a returned match is **not** guaranteed to begin at or after
                it. Defaults to 0, which imposes no bound.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results. Ambiguity can be
            represented by multiple attempts with the same candidate but different matches. Every
            ``read_idx`` is in original-read coordinates regardless of ``start_idx``, so a caller
            never has to add the offset back.

        Note:
            The two searching matchers enforce ``start_idx`` differently and are deliberately not
            unified: ``KmerMatcher`` treats it as a floor on seeding only, while
            ``AlignmentMatcher`` trims the read to it and so treats it as a floor on the match
            itself. Changing either would move existing barcode calls, so the divergence is
            recorded here rather than reconciled.
        """
        pass

    # Helper funcs
    def check_read_len(self, read: str) -> bool:
        """
        Check if the read is long enough to contain the component.

        Requires a resolved start position, since the component's extent in the read is its
        start plus its length.

        Args:
            read: The sequencing read to measure.

        Returns:
            True if the read extends to the end of the component, False otherwise.

        Raises:
            UnresolvedComponentStartError: If the component has no resolved start position.
        """
        if self.component.start is None:
            raise UnresolvedComponentStartError(
                f"Component {self.component.name} has no resolved start position, so its extent "
                "in the read cannot be computed. A component without a resolved start must be "
                "located by a matcher that does not rely on a fixed start."
            )
        end = self.component.start + self.component.length
        return len(read) >= end

    def trim_read(self, read: str, start_idx: int) -> str:
        """
        Trim the read to the expected length for the barcode component.

        This is used to ensure that the read segment being matched is of the correct length, which
        can help improve matching accuracy and reduce false positives.
        """
        if start_idx >= len(read):
            log.debug(
                f"Start index {start_idx} is beyond read length {len(read)}. Returning empty string."
            )
            return ""
        return read[start_idx:]

    def check_spacers(
        self, read: str, match_idx: tuple[int, int]
    ) -> dict[Literal["upstream", "downstream"], str | None]:
        """Check if appropriate spacer sequence(s) are present adjacent to the barcode match."""

        # Get adjacent spacer regions based on read structure
        spacer_components: dict[Literal["upstream", "downstream"], ReadComponent | None] = {
            "upstream": self.chemistry.read_structure.get_previous(self.component),
            "downstream": self.chemistry.read_structure.get_next(self.component),
        }

        def match_seq(
            loc: Literal["upstream", "downstream"], spacer_component: ReadComponent | None
        ) -> str | None:
            if (
                spacer_component
                and spacer_component.type in (ReadComponentType.PRIMER, ReadComponentType.OTHER)
                and spacer_component.sequence
            ):
                spacer_seq = None
                if loc == "upstream":
                    if match_idx[0] >= spacer_component.length:
                        spacer_seq = read[match_idx[0] - spacer_component.length : match_idx[0]]
                elif loc == "downstream":
                    if len(read) - match_idx[1] >= spacer_component.length:
                        spacer_seq = read[match_idx[1] : match_idx[1] + spacer_component.length]

                if spacer_seq is not None and spacer_seq == spacer_component.sequence:
                    return (
                        spacer_component.name
                    )  # Return spacer name if present and matches expected sequence

            return None  # No defined spacer sequence, treat as not present

        is_present = {}
        for loc, spacer_comp in spacer_components.items():
            is_present[loc] = match_seq(loc, spacer_comp)

        return is_present
