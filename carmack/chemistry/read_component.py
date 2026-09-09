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
        return self.length is None or self.type is ReadComponentType.HOMOPOLYMER

    @property
    def records_position(self) -> bool:
        """Return whether barcode extraction writes this component's position down.

        Barcode extraction is the first stage to run and writes a ``<NAME>`` and
        a ``<NAME>_POS`` tag for each barcode and for nothing else, so a barcode
        span is the only coordinate a read carries that every later stage can
        rely on. Narrower than "some stage records this": UMI extraction also
        writes a ``UMI_POS``, and that span is deliberately excluded here,
        because a stage that anchored on it would be made to depend on UMI
        extraction having run first.

        This is deliberately separate from :attr:`is_anchor`, and neither implies
        the other -- an anchor is a component that *can* be located within a
        read, while this says the position it was found at is written down early
        enough to be read back. A primer anchors a neighbour perfectly well and
        records nothing.
        """
        return self.type is ReadComponentType.BARCODE
