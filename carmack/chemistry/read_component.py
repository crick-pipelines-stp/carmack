from dataclasses import dataclass, field
from enum import StrEnum


class ReadComponentType(StrEnum):
    BARCODE = "BARCODE"
    GGG = "GGG"
    PRIMER = "PRIMER"
    TGIDX = "TGIDX"
    UMI = "UMI"
    OTHER = "OTHER"


@dataclass(kw_only=True)
class ReadComponent:
    """
    Defines the layout of a single barcode component within a read.

    Attributes:
        name: Identifier for this component
        type: Classification of this component within the read structure
        length: Length of the sequence component
        sequence: Optional known sequence for this component (e.g., primer sequence). Should be None for barcode components (since they are loaded from a whitelist).
        start: Expected start position in the read (0-indexed). This is set automatically and should not be provided manually.
    """

    name: str
    type: ReadComponentType = ReadComponentType.OTHER
    length: int
    sequence: str | None = None
    _start: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Read component length must be positive.")

        if not isinstance(self.type, ReadComponentType):
            raise TypeError(
                f"ReadComponent type must be a ReadComponentType enum member, got {type(self.type).__name__}"
            )

        if self.type is ReadComponentType.BARCODE and self.sequence is not None:
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
