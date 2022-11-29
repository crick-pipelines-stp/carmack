"""
Defines base class for all chemistry classes.
"""

from abc import ABC, abstractmethod
 
class ChemistryBase(ABC):
    """Base class for chemistry objects."""
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def load_barcode_set(self) -> list:
        """Load barcode set for chemistry."""
        pass

    @abstractmethod
    def subset_barcodes(self, seq: str) -> list:
        """Subset barcodes from sequence for given chemistry."""
        pass
