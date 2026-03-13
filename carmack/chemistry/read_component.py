from dataclasses import dataclass, field


@dataclass
class ReadComponent:
    """
    Defines the layout of a single barcode component within a read.

    Attributes:
        name: Identifier for this component
        is_barcode: Indicates if this component is a barcode
        length: Length of the sequence component
        sequence: Optional known sequence for this component (e.g., primer sequence). Should be None for barcodes (since they are loaded from a whitelist).
        start: Expected start position in the read (0-indexed). This is set automatically and should not be provided manually.
    """

    name: str
    is_barcode: bool
    length: int
    sequence: str | None = None
    _start: int = field(init=False, default=0)

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError("Read component length must be positive.")

        if self.is_barcode and self.sequence is not None:
            raise ValueError("Barcode components should not be instantiated with a sequence.")

    @property
    def start(self) -> int:
        """Return the start position of this component within the read."""
        return self._start

    @start.setter
    def start(self, value: int) -> None:
        if value < 0:
            raise ValueError("Start position must be non-negative.")
        self._start = value
