from abc import ABC, abstractmethod

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt
from carmack.chemistry.read_structure import ReadComponent


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
    ):
        self.whitelist_set = frozenset(whitelist)
        self.barcode_component = barcode_component

        # Check if barcode_component is actually a barcode
        if not self.barcode_component.is_barcode:
            raise ValueError(
                f"barcode_component must be a barcode component (is_barcode=True), got {self.barcode_component}"
            )

    @abstractmethod
    def match(self, read: str) -> BarcodeMatchAttempt:
        """
        Attempt to match barcodes in the given read.

        Args:
            read: The sequencing read to match against.

        Returns:
            A BarcodeMatchAttempt object describing the matching attempt.
        """
        pass

    # Helper funcs
    def check_read_len(self, read: str) -> bool:
        """Check if the read is long enough to contain the barcode component."""
        end = (self.barcode_component.start or 0) + self.barcode_component.length
        return len(read) >= end
