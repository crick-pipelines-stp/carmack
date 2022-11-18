"""
Defines interface for all chemistry classes.
"""

class ChemistryBase:
    """Base class for chemistry objects."""

    def __init__(self) -> None:
        pass
        
    def load_barcode_set(self):
        """Load barcode set for chemistry."""
        raise NotImplementedError

    def subset_barcodes(self, seq):
        """Subset barcodes from sequence for given chemistry."""
        raise NotImplementedError
