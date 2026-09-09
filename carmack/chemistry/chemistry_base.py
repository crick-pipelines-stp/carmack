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

# A target index never begins with more than this many copies of the anchor base that
# precedes it in the read structure. The anchor run and the index are indistinguishable at
# their boundary, so a longer leading run would push the apparent run end into the index
# itself; this bound is what makes the index locatable at all. Validated when a whitelist
# loads, and it is a fixed cap rather than something derived from the whitelist: deriving it
# would silently widen the search window the moment a longer-leading entry was added, which
# is the drift the bound exists to prevent.
TGIDX_MAX_LEADING_ANCHOR = 2

# Largest target index whitelist that does not warn. Measured, not estimated: driving the
# real KmerMatcher over the real whitelist at the shipped error budget, the rate at which an
# index-free read is falsely assigned a target is 0.17% at one target, 1.8% at eight and
# 13.8% at ninety-six. Do not argue this threshold down without re-measuring.
TGIDX_WHITELIST_SIZE_LIMIT = 4


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
            Defaults to one so that a chemistry adding a target index without
            stating a tolerance gets sensible behaviour rather than silent
            exact-match-only. One is also the tolerance a k=4 seed-and-extend
            search can serve: an 8bp index holds two disjoint 4-mers, so a
            single error spoils at most one of them and a seed always survives,
            whereas two errors need not leave either intact.
    """

    barcode: int
    spacer: int
    tgidx: int = 1


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
        """Return the whitelist of valid target index sequences for this chemistry.

        The entries come from the source declared for the target index component
        in :meth:`whitelist_sources`.

        Returns:
            A tuple of valid target index sequences in file order, or an empty
            tuple when the chemistry declares no target index component.
        """
        component = self.tgidx_component()
        if component is None:
            return ()
        return self.whitelists[component.name]

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

    def tgidx_component(self) -> ReadComponent | None:
        """Return the target index component of the read structure, if one is defined.

        Returns:
            The single ``TGIDX`` :class:`ReadComponent`, or ``None`` for
            chemistries that carry no target index.
        """
        return next(
            (comp for comp in self.read_structure if comp.type is ReadComponentType.TGIDX),
            None,
        )

    def tgidx_anchor(self) -> ReadComponent | None:
        """Return the homopolymer run immediately 5' of the target index.

        The target index has no fixed start —
        :meth:`ReadStructure.compute_start_positions` leaves ``start`` as
        ``None`` for every component following a variable-length one — so the
        end of this run is the only reference point a read offers for locating
        it. That makes the homopolymer mandatory rather than merely useful. For
        ``custom_seq_1_0`` this is the ``POLYG`` homopolymer, whose
        :attr:`ReadComponent.homopolymer_base` supplies the anchor base.

        Returns:
            The preceding :class:`ReadComponent` when it is a homopolymer, or
            ``None`` when there is no target index or its 5' neighbour is not a
            homopolymer.
        """
        component = self.tgidx_component()
        if component is None:
            return None
        previous = self.read_structure.get_previous(component)
        if previous is not None and previous.type is ReadComponentType.HOMOPOLYMER:
            return previous
        return None

    def tgidx_right_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 3' of the target index.

        The right anchor is the component following the target index in the read
        structure, but only when that neighbour exists and can anchor a
        variable-length component (:attr:`ReadComponent.is_anchor`). For
        ``carmack_custom_seq_1_0`` this is the ``ME`` primer. Unlike
        :meth:`tgidx_anchor`, a ``None`` result here is not a construction-time
        failure: it describes a legitimate chemistry with no adapter between the
        target index and the insert, which the generic trim-boundary arithmetic
        already resolves correctly (offset zero).

        Returns:
            The anchoring :class:`ReadComponent`, or ``None`` when there is no
            target index or its right neighbour cannot anchor it.
        """
        component = self.tgidx_component()
        if component is None:
            return None
        following = self.read_structure.get_next(component)
        if following is not None and following.is_anchor:
            return following
        return None

    def supports_target_assignment(self) -> bool:
        """Return whether this chemistry can support target index assignment.

        Target assignment requires a target index component anchored on its 5'
        side by a homopolymer run. Chemistries without a target index (e.g.
        hydrop) return ``False`` rather than raising. Construction rejects a
        target index that is declared without such an anchor, so on a
        constructed chemistry this answers whether a target index is declared at
        all; the anchor test still stands because it is also consulted while
        that construction-time check is running.

        Returns:
            ``True`` when the read structure contains a target index preceded by
            a homopolymer run, ``False`` otherwise.
        """
        return self.tgidx_anchor() is not None

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

    def validate_target_index(self) -> None:
        """Reject a target index declaration that could not be matched.

        Called from :meth:`__post_init__`. Each check closes a hole that would
        otherwise surface as silently unassigned reads rather than as a
        construction failure. A whitelist longer than
        ``TGIDX_WHITELIST_SIZE_LIMIT`` only warns, because a larger panel stays
        workable once each target is confirmed by its Mosaic End sequence.

        Raises:
            KeyError: If a declared target index component has no whitelist
                source.
            ValueError: If the target index is not preceded by a homopolymer
                component to anchor it, or if a whitelist entry begins with more
                than ``TGIDX_MAX_LEADING_ANCHOR`` copies of the anchor base.
        """
        component = self.tgidx_component()
        if component is None:
            return

        whitelists = self.whitelists
        if component.name not in whitelists:
            raise KeyError(
                f"Target index layout '{component.name}' not found in whitelist: "
                "no whitelist source is declared for this component"
            )

        anchor = self.tgidx_anchor()
        if anchor is None:
            previous = self.read_structure.get_previous(component)
            neighbour = "nothing" if previous is None else f"component '{previous.name}'"
            raise ValueError(
                f"Target index component '{component.name}' must be preceded by a homopolymer "
                f"component to anchor it, but is preceded by {neighbour}. The index carries no "
                "fixed start, so the end of the homopolymer run is the only reference point "
                "that can locate it."
            )

        entries = whitelists[component.name]
        for sequence in entries:
            # lstrip with a single character removes exactly the leading run of it.
            leading = len(sequence) - len(sequence.lstrip(anchor.homopolymer_base))
            if leading > TGIDX_MAX_LEADING_ANCHOR:
                raise ValueError(
                    f"Target index '{sequence}' begins with {leading} copies of the anchor base "
                    f"'{anchor.homopolymer_base}', more than the {TGIDX_MAX_LEADING_ANCHOR} "
                    "allowed. The anchor run and the index are indistinguishable at their "
                    "boundary, so a longer leading run would leave the index unlocatable."
                )

        if len(entries) > TGIDX_WHITELIST_SIZE_LIMIT:
            log.warning(
                f"Target index whitelist for '{component.name}' holds {len(entries)} entries, "
                f"more than the {TGIDX_WHITELIST_SIZE_LIMIT} that false-assignment "
                "measurements support. Confirm each target by its Mosaic End sequence rather "
                "than by the index alone."
            )

    def __post_init__(self):
        # Validate that all barcode components defined in the read structure have whitelists
        whitelists = self.whitelists
        for component in self.read_structure:
            if component.type is ReadComponentType.BARCODE and component.name not in whitelists:
                raise KeyError(
                    f"Barcode layout '{component.name}' not found in whitelist: "
                    "no whitelist source is declared for this component"
                )

        # Validate the target index declaration, if the chemistry declares one
        self.validate_target_index()

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
