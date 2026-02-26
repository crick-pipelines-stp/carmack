import logging
from abc import ABC, abstractmethod
from typing import Literal

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_structure import ReadComponent


log = logging.getLogger(__name__)


class MatcherBase(ABC):
    """
    Abstract base class for barcode matchers.

    Subclasses must implement the match method to perform barcode matching
    based on the provided whitelist and read structure.
    """

    def __init__(
        self,
        whitelist: tuple[str, ...],
        barcode_component: ReadComponent,
        chemistry: ChemistryBase,
    ):
        self.whitelist_set = frozenset(whitelist)
        self.barcode_component = barcode_component
        self.chemistry = chemistry

        # Check if barcode_component is actually a barcode
        if not self.barcode_component.is_barcode:
            raise ValueError(
                f"barcode_component must be a barcode component (is_barcode=True), got {self.barcode_component}"
            )

    @abstractmethod
    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read.

        Args:
            read: The sequencing read to match against.
            start_idx: The index in the read to start matching from (default is 0).

        Returns:
            List of BarcodeMatchAttempt objects representing the match results. Ambiguity can be
            represented by multiple attempts with the same candidate but different matches.
        """
        pass

    # Helper funcs
    def check_read_len(self, read: str) -> bool:
        """Check if the read is long enough to contain the barcode component."""
        end = (self.barcode_component.start or 0) + self.barcode_component.length
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
            "upstream": self.chemistry.read_structure.get_previous(self.barcode_component),
            "downstream": self.chemistry.read_structure.get_next(self.barcode_component),
        }

        def match_seq(
            loc: Literal["upstream", "downstream"], spacer_component: ReadComponent | None
        ) -> str | None:
            if (
                spacer_component
                and spacer_component.is_barcode is False
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
