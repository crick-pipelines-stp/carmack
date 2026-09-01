from dataclasses import dataclass, field
from enum import StrEnum

from carmack.chemistry.annotation import position_key

HOMOPOLYMER_BASES = {"A", "C", "G", "T"}


class ReadComponentType(StrEnum):
    BARCODE = "BARCODE"
    HOMOPOLYMER = "HOMOPOLYMER"
    PRIMER = "PRIMER"
    TGIDX = "TGIDX"
    UMI = "UMI"
    OTHER = "OTHER"


@dataclass(kw_only=True)
class ReadComponent:
    """
    Defines the layout of a single read component within a read.

    Attributes:
        name: Identifier for this component.
        type: Classification of this component within the read structure.
        length: Expected length of the sequence component. May be ``None`` for
            components whose occupied length is only known at read time (e.g. a
            homopolymer run).
        sequence: Optional known sequence for this component (e.g., primer
            sequence). Should be None for barcode components (since they are
            loaded from a whitelist).
        length_tolerance: Symmetric jitter, in bases, allowed around ``length``
            for variable components such as UMIs. Zero means a fixed length.
        homopolymer_base: For homopolymer components, the single repeated base
            (one of A, C, G, T).
        min_run: For homopolymer components, the minimum run length that anchors
            the component.
        start: Expected start position in the read (0-indexed). This is assigned
            automatically by the read structure and should not be provided
            manually. It is ``None`` until computed, and stays ``None`` for
            components that follow a variable-length component.
    """

    name: str
    type: ReadComponentType = ReadComponentType.OTHER
    length: int | None = None
    sequence: str | None = None
    length_tolerance: int = 0
    homopolymer_base: str | None = None
    min_run: int | None = None
    start: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.type, ReadComponentType):
            raise TypeError(
                f"ReadComponent type must be a ReadComponentType enum member, got {type(self.type).__name__}"
            )

        if self.type is ReadComponentType.HOMOPOLYMER:
            if self.homopolymer_base not in HOMOPOLYMER_BASES:
                raise ValueError(
                    "Homopolymer components require a single homopolymer_base in {A, C, G, T}."
                )
            if self.min_run is None or self.min_run <= 0:
                raise ValueError("Homopolymer components require a positive min_run.")
        elif self.length is None or self.length <= 0:
            raise ValueError("Read component length must be positive.")

        if self.type is ReadComponentType.BARCODE and self.sequence is not None:
            raise ValueError("Barcode components should not be instantiated with a sequence.")

        if self.type is ReadComponentType.UMI and self.length_tolerance < 0:
            raise ValueError("UMI components require a non-negative length_tolerance.")

    @property
    def position_key(self) -> str:
        """Return the annotation header key for this component's position."""
        return position_key(self.name)

    @property
    def is_anchor(self) -> bool:
        """Return whether this component can anchor a neighbouring variable component.

        An anchor is a component whose position can be located within a read to
        bound a neighbouring variable-length component.
        """
        return self.type in {
            ReadComponentType.BARCODE,
            ReadComponentType.PRIMER,
            ReadComponentType.HOMOPOLYMER,
        }

    @property
    def is_variable_length(self) -> bool:
        """Return whether the occupied length is not a single fixed value."""
        return (
            self.length is None
            or self.length_tolerance > 0
            or self.type is ReadComponentType.HOMOPOLYMER
        )

    @property
    def min_length(self) -> int | None:
        """Return the minimum number of bases this component occupies."""
        if self.type is ReadComponentType.HOMOPOLYMER:
            return self.min_run
        if self.length is None:
            return None
        if self.length_tolerance > 0:
            return self.length - self.length_tolerance
        return self.length
