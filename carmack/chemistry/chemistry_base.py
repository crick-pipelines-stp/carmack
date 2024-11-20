"""
Defines base class for all chemistry classes.
"""

from abc import ABC, abstractmethod
import numpy as np

class ChemistryBase(ABC):
    """Base class for chemistry objects."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_barcode_set(self) -> list:
        """Load barcode set for chemistry."""
        pass

    @abstractmethod
    def construct_whitelist(self, barcode_set: list) -> list:
        """Construct full whitelist from barcode set"""
        pass

    @abstractmethod
    def subset_whitelist_guess(self, seq: list) -> str:
        """Make best guess sequence subset based on protocol chemistry for a whitelist match"""
        pass

    @abstractmethod
    def subset_barcode_chunks(self, seq: str, qs: np.ndarray) -> list:
        """Subset barcodes locations from sequence for given chemistry."""
        pass
