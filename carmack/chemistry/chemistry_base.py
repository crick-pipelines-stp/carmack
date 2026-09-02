"""
Base class for chemistry definitions.

Each chemistry defines the structure of barcodes within reads, including:
- Barcode positions and lengths
- Primer sequences for anchor detection
- Whitelist loading from data files
"""

import logging
from abc import ABC
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from importlib.resources.abc import Traversable

from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.gzip_file import GzipFile

log = logging.getLogger(__name__)


class AbstractCachedProperty(cached_property):
    """A cached property that ``ABCMeta`` still recognises as abstract.

    :class:`functools.cached_property` does not propagate
    ``__isabstractmethod__`` from the function it wraps, so an
    ``@abstractmethod`` beneath ``@cached_property`` is invisible to
    ``ABCMeta`` and leaves the owning class instantiable. Declaring a member
    with this subclass keeps the per-instance caching while restoring
    construction-time enforcement.
    """

    __isabstractmethod__ = True


@dataclass(frozen=True)
class MatchErrors:
    """Maximum edit distances allowed per component type when matching.

    Attributes:
        barcode: Max edits when matching a barcode component to its whitelist.
        spacer: Max edits when matching a spacer / primer sequence.
        tgidx: Max edits when matching a TGIDX component to its whitelist.
    """

    barcode: int
    spacer: int
    tgidx: int = 0


@dataclass(frozen=True)
class WhitelistSource:
    """Where a component's whitelist is read from, and how each line is parsed.

    Attributes:
        path: Location of the whitelist file, one entry per line.
        line_slice: Half-open ``(start, stop)`` slice applied to every stripped
            line to extract the sequence, or ``None`` to keep the whole line.
    """

    path: Traversable
    line_slice: tuple[int, int | None] | None = None


@dataclass
class ChemistryBase(ABC):
    """
    Abstract base class for chemistry definitions.

    Subclasses must implement the abstract members defining the read layout and
    matching tolerances, and may override whitelist_sources() to declare where
    whitelisted components read their sequences from.
    """

    @AbstractCachedProperty
    def name(self) -> str:
        """Name of the chemistry.

        Raises:
            NotImplementedError: Always; subclasses must override this member.
        """
        raise NotImplementedError

    @AbstractCachedProperty
    def read_structure(self) -> ReadStructure:
        """Defines the layout of barcodes and other components within reads.

        Raises:
            NotImplementedError: Always; subclasses must override this member.
        """
        raise NotImplementedError

    @AbstractCachedProperty
    def max_errors(self) -> MatchErrors:
        """Maximum allowed errors (substitutions/indels) when matching read components.

        Raises:
            NotImplementedError: Always; subclasses must override this member.
        """
        raise NotImplementedError

    def whitelist_sources(self) -> dict[str, WhitelistSource]:
        """
        Declare where the whitelist of each whitelisted component is read from.

        Chemistries override this to point their whitelisted components at
        packaged data files. The default is empty so that a chemistry with no
        whitelisted components remains valid.

        Returns:
            Dictionary mapping component names to their whitelist sources.
        """
        return {}

    def load_whitelist(self, component_name: str) -> tuple[str, ...]:
        """
        Load the whitelist declared for a single component.

        Args:
            component_name: The name of a component declared in
                whitelist_sources(). Appearing in the read structure is not
                enough; the component must have a declared source.

        Returns:
            Tuple of valid sequences in file order, with empty entries dropped.

        Raises:
            ValueError: If the component has no declared whitelist source.
        """
        sources = self.whitelist_sources()
        source = sources.get(component_name)

        if source is None:
            valid_names = list(sources.keys())
            raise ValueError(
                f"Unknown whitelist component: {component_name}. Valid names: {valid_names}"
            )

        log.debug(f"Loading whitelist for {component_name} from {source.path}")

        sequences: list[str] = []
        with closing(GzipFile(str(source.path)).open_read_iterator(as_string=True)) as stream:
            for line in stream:
                sequence = line.strip()
                if source.line_slice is not None:
                    sequence = sequence[source.line_slice[0] : source.line_slice[1]]
                if sequence:
                    sequences.append(sequence)

        result = tuple(sequences)
        log.debug(f"Loaded {len(result)} sequences for {component_name}")
        return result

    @cached_property
    def whitelists(self) -> dict[str, tuple[str, ...]]:
        """
        Load every whitelist declared by whitelist_sources().

        Each declared source is read exactly once and cached for the lifetime of
        the chemistry instance.

        Returns:
            Dictionary mapping component names to their whitelist tuples.

        Raises:
            KeyError: If a declared whitelist could not be loaded.
        """
        barcode_names = {
            comp.name for comp in self.read_structure if comp.type is ReadComponentType.BARCODE
        }

        loaded: dict[str, tuple[str, ...]] = {}
        for component_name in self.whitelist_sources():
            try:
                loaded[component_name] = self.load_whitelist(component_name)
            except Exception as e:
                if component_name in barcode_names:
                    raise KeyError(
                        f"Barcode layout '{component_name}' not found in whitelist: {e}"
                    ) from e
                raise KeyError(
                    f"Whitelist for component '{component_name}' could not be loaded: {e}"
                ) from e

        return loaded

    @cached_property
    def barcode_whitelists(self) -> dict[str, tuple[str, ...]]:
        """
        Return the whitelists of the barcode components in read-structure order.

        Every barcode component is guaranteed to be present: construction
        rejects a read structure whose barcode has no whitelist.

        Returns:
            Dictionary mapping barcode component names to their whitelist
            tuples, keyed in the order the barcodes appear in the read
            structure.
        """
        return {
            comp.name: self.whitelists[comp.name]
            for comp in self.read_structure
            if comp.type is ReadComponentType.BARCODE
        }

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return the whitelist of valid TGIDX sequences for this chemistry.

        Returns:
            A tuple of valid TGIDX sequences, or an empty tuple when the
            chemistry has no TGIDX component.
        """
        return ()

    def umi_component(self) -> ReadComponent | None:
        """Return the UMI component of the read structure, if one is defined.

        Returns:
            The single ``UMI`` :class:`ReadComponent`, or ``None`` for
            barcode-only chemistries that carry no UMI.
        """
        return next(
            (comp for comp in self.read_structure if comp.type is ReadComponentType.UMI),
            None,
        )

    def umi_left_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 5' of the UMI.

        The left anchor is the component preceding the UMI in the read
        structure, but only when that neighbour exists and can anchor a
        variable-length component (:attr:`ReadComponent.is_anchor`). For
        ``custom_seq_1_0`` this is the ``BC1`` barcode.

        Returns:
            The anchoring :class:`ReadComponent`, or ``None`` when there is no
            UMI or its left neighbour cannot anchor it.
        """
        umi = self.umi_component()
        if umi is None:
            return None
        previous = self.read_structure.get_previous(umi)
        if previous is not None and previous.is_anchor:
            return previous
        return None

    def umi_right_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 3' of the UMI.

        The right anchor is the component following the UMI in the read
        structure, but only when that neighbour exists and can anchor a
        variable-length component (:attr:`ReadComponent.is_anchor`). For
        ``custom_seq_1_0`` this is the ``POLYG`` homopolymer.

        Returns:
            The anchoring :class:`ReadComponent`, or ``None`` when there is no
            UMI or its right neighbour cannot anchor it.
        """
        umi = self.umi_component()
        if umi is None:
            return None
        following = self.read_structure.get_next(umi)
        if following is not None and following.is_anchor:
            return following
        return None

    def supports_umi_extraction(self) -> bool:
        """Return whether this chemistry can support UMI extraction.

        UMI extraction requires a UMI component whose left neighbour exists and
        can anchor it. Barcode-only chemistries without a UMI (e.g. hydrop)
        return ``False`` rather than raising.

        Returns:
            ``True`` when the read structure contains a UMI anchored on its 5'
            side, ``False`` otherwise.
        """
        return self.umi_left_anchor() is not None

    def construct_full_barcode(self, barcodes: dict[str, str]) -> str:
        """
        Construct the full barcode sequence by concatenating individual
        barcode components in the correct order. Order is determined by the read structure.
        """
        barcode_components = self.read_structure.get_components_by_type(ReadComponentType.BARCODE)

        parts = []
        for comp in barcode_components:
            value = barcodes.get(comp.name)
            if value is None:
                raise ValueError(
                    f"Missing barcode component '{comp.name}' for full barcode construction."
                )
            parts.append(value)
        return "".join(parts)

    def __post_init__(self):
        # Validate that all barcode components defined in the read structure have whitelists
        whitelists = self.whitelists
        for component in self.read_structure:
            if component.type is ReadComponentType.BARCODE and component.name not in whitelists:
                raise KeyError(
                    f"Barcode layout '{component.name}' not found in whitelist: "
                    "no whitelist source is declared for this component"
                )

        # Validate that all spacers defined in the read structure are present in the spacers dictionary
        spacer_names = {
            comp.name for comp in self.read_structure if comp.type is not ReadComponentType.BARCODE
        }
        spacer_seqs = self.read_structure.get_known_sequences()
        missing_spacers = set(spacer_seqs.keys()) - spacer_names

        if not spacer_names.issuperset(spacer_seqs.keys()):
            raise KeyError(
                f"Spacers defined in the chemistry must be present in the read structure. Missing spacers: {missing_spacers}"
            )
