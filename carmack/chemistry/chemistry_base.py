"""
Base class for chemistry definitions.

Each chemistry defines the structure of barcodes within reads, including:
- Barcode positions and lengths
- Primer sequences for anchor detection
- Whitelist loading from data files
"""

import logging
from abc import ABC
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib.resources.abc import Traversable
from itertools import combinations

from carmack.barcode.barcode_utils import edit_distance
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


def deletion_variants(sequence: str, max_deletions: int) -> set[str]:
    """
    Return every string reachable by deleting at most ``max_deletions`` characters.

    Args:
        sequence: The sequence to delete from.
        max_deletions: Largest number of characters to delete.

    Returns:
        The deletion neighbourhood, including ``sequence`` itself.
    """
    variants = {sequence}
    frontier = {sequence}
    for _ in range(max_deletions):
        wider = {
            variant[:i] + variant[i + 1 :] for variant in frontier for i in range(len(variant))
        }
        variants |= wider
        frontier = wider
    return variants


@lru_cache(maxsize=None)
def close_whitelist_pairs(
    whitelist: tuple[str, ...], max_distance: int
) -> dict[tuple[str, str], int]:
    """
    Find every pair of whitelist entries within ``max_distance`` edits of each other.

    Comparing all pairs directly is quadratic in the whitelist and cubic in the entry length,
    which is too slow to sit on the construction path of a chemistry that is built when its
    module is imported. Instead entries are bucketed by their deletion neighbourhoods: two
    strings within ``d`` edits can always be reduced to a common string by deleting at most
    ``d`` characters from each, so any close pair collides in at least one bucket and only
    colliding pairs need their distance computing. On the shipped 96-entry whitelists this is
    around thirty times faster than the direct comparison, and it agrees with it exactly.

    Results are cached on the whitelist tuple, so a chemistry constructed repeatedly -- which
    the factory does once per registration and the tests do constantly -- pays for the scan
    once per process.

    Args:
        whitelist: The entries to compare, as a hashable tuple.
        max_distance: Largest edit distance to report a pair at.

    Returns:
        Mapping of each close pair, ordered within the pair for stability, to its edit
        distance. Empty when no pair is that close. Cached, so every caller is handed the
        same object and none of them may mutate it.
    """
    buckets: dict[str, set[str]] = defaultdict(set)
    for entry in whitelist:
        for variant in deletion_variants(entry, max_distance):
            buckets[variant].add(entry)

    distances: dict[tuple[str, str], int] = {}
    for colliding in buckets.values():
        if len(colliding) < 2:
            continue
        for first, second in combinations(sorted(colliding), 2):
            pair = (first, second)
            if pair in distances:
                continue
            distance = edit_distance(first, second, "N", False)
            if distance <= max_distance:
                distances[pair] = distance
    return distances


@lru_cache(maxsize=None)
def warn_once(message: str) -> None:
    """
    Emit a warning the first time this process is asked to, and never again.

    Chemistries are constructed when their module is imported, because the factory reads a
    chemistry's name off an instance to register it, and they are then constructed again for
    every run and every test. A construction-time warning would therefore repeat several times
    before anything has happened. Caching on the message keeps each distinct finding to one
    line per process without the caller having to track what it has already said.

    Args:
        message: The warning text, which is also the cache key.
    """
    log.warning(message)


@dataclass(frozen=True)
class WhitelistDistancePolicy:
    """
    How a chemistry answers for a whitelist that cannot satisfy its own error budget.

    The distance rule itself is fixed; what a chemistry chooses is whether it is in a position
    to act on a violation. A chemistry whose barcode set we design can retire an offending
    entry, so a violation there is a defect and construction should refuse. A chemistry
    implementing someone else's published protocol cannot change its whitelist at all, so
    refusing to construct would only make the chemistry unusable while leaving the underlying
    risk exactly where it was; reporting it loudly is the whole of what we can do.

    Attributes:
        enforce: Whether a pair within the error budget fails construction. False downgrades
            it to a warning, for a whitelist this project does not own.
        exempt_pairs: Pairs, each as a frozenset of the two entries, that are known to violate
            the bound and are accepted for now. Declaring one is a recorded decision to ship a
            whitelist with a known blind spot, which is why it is an explicit pair rather than
            a threshold that could quietly absorb the next one too.
    """

    enforce: bool = True
    exempt_pairs: frozenset[frozenset[str]] = frozenset()


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


@dataclass(frozen=True)
class AnchorOffset:
    """The recorded anchor a component's start is measured from, and that distance.

    Attributes:
        anchor: Nearest component 5' of the target that both anchors a variable
            neighbour and has its own position written to the read header, so its
            span end can be read back at read time.
        offset: Fixed bases between that anchor's end and the target component's
            start, the sum of the lengths of the components in between. Zero when
            the anchor is the target's immediate 5' neighbour.
    """

    anchor: ReadComponent
    offset: int

    @property
    def position_key(self) -> str:
        """Return the annotation header key the anchor's span is read from."""
        return self.anchor.position_key


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

    def resolve_anchor_offset(self, component: ReadComponent) -> AnchorOffset:
        """Return the recorded anchor a component's start can be measured from.

        Walks 5' from ``component``, summing the lengths of the components it
        crosses, and stops at the first that both :attr:`ReadComponent.is_anchor`
        and :attr:`ReadComponent.records_position`. That component's span is the
        only coordinate the read offers, so at read time the target component
        starts at the anchor's span end plus the returned offset. Walking from
        ``POLYG`` in ``custom_seq_1_0`` crosses the 8bp ``UMI`` and lands on
        ``BC1``, giving an offset of 8; walking from the ``UMI`` lands on ``BC1``
        directly, giving 0.

        Both conditions are required and neither implies the other.
        ``is_anchor`` says the component can be located within a read at all;
        ``records_position`` says the position it was found at is written to the
        header and so can be read back here. A primer satisfies the first and not
        the second, and treating the two as one would have this return a
        component whose position tag is never written, making every read look
        like it was missing its anchor.

        Args:
            component: The component whose start is to be located.

        Returns:
            The anchor to read the span from, and the fixed offset to add to its
            span end.

        Raises:
            ValueError: If a variable-length component is crossed before an
                anchor is reached, or if no such anchor lies 5' of ``component``.
        """
        offset = 0
        current = component
        while True:
            previous = self.read_structure.get_previous(current)
            if previous is None:
                raise ValueError(
                    f"chemistry '{self.name}' has no anchor 5' of component "
                    f"'{component.name}' whose position is recorded on the read header, "
                    "so that component's start cannot be located in a read"
                )
            if previous.is_anchor and previous.records_position:
                return AnchorOffset(anchor=previous, offset=offset)
            # Tested before the length is added, because a variable component's
            # length may be None and adding it would raise a TypeError in place
            # of the explanatory error below.
            if previous.is_variable_length:
                raise ValueError(
                    f"chemistry '{self.name}' places variable-length component "
                    f"'{previous.name}' between '{component.name}' and the nearest "
                    "recorded anchor, so the distance between them is not a fixed "
                    f"number of bases and '{component.name}' cannot be located"
                )
            offset += previous.length
            current = previous

    def umi_right_anchor(self) -> ReadComponent | None:
        """Return the anchor component immediately 3' of the UMI.

        Diagnostic only. The UMI is a fixed-length slice taken off its left
        anchor, so nothing about extracting it consults the component 3' of it.
        This supplies the base for the anchor-run tally that UMI extraction
        reports as a check that the layout is holding, which is why its absence
        degrades that report rather than failing a run.

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
        """Return whether this chemistry declares a UMI to extract.

        Answers only whether a UMI component exists. Whether its start can
        actually be located is :meth:`resolve_anchor_offset`'s question, and that
        one raises with a specific diagnostic rather than collapsing every way it
        can fail into a bare ``False``. Barcode-only chemistries without a UMI
        (e.g. hydrop) return ``False`` here rather than raising.

        Returns:
            ``True`` when the read structure contains a UMI component, ``False``
            otherwise.
        """
        return self.umi_component() is not None

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

    def whitelist_distance_policy(self) -> WhitelistDistancePolicy:
        """
        Declare how this chemistry answers for a whitelist that violates the distance bound.

        The default enforces with no exemptions, so a new chemistry gets the strict answer
        unless it says otherwise. Chemistries override this to record a decision, never to
        make the check quieter by accident.

        Returns:
            The policy ``validate_whitelist_distances`` applies to this chemistry.
        """
        return WhitelistDistancePolicy()

    def match_errors_for(self, component: ReadComponent) -> int | None:
        """
        Return the error budget this chemistry matches ``component`` to its whitelist with.

        Args:
            component: A component of this chemistry's read structure.

        Returns:
            The budget, or ``None`` for a component that is not matched against a whitelist
            and so has no whitelist distance requirement.
        """
        if component.type is ReadComponentType.BARCODE:
            return self.max_errors.barcode
        if component.type is ReadComponentType.TGIDX:
            return self.max_errors.tgidx
        return None

    def validate_whitelist_distances(self) -> None:
        """
        Reject or report whitelists that cannot satisfy the error budget they are used with.

        Correcting a read to a whitelist entry is only sound while every window inside the
        error budget has a single nearest entry. Two thresholds bound that, and they differ in
        kind rather than degree:

        * **Within ``max_errors``** -- one sequencing error inside the budget turns one valid
          entry into *the other valid entry*. It then matches exactly at its own expected
          position and is reported as a perfect match, so there is no ambiguity to observe and
          nothing downstream can detect it. The read is not lost, it is attributed to the wrong
          cell. No matching logic can fix this; only the whitelist can, which is why this tier
          fails construction for a whitelist we own.
        * **Within ``2 * max_errors``** -- a window can sit equally close to two entries. That
          is observable, and is now a terminal ambiguity verdict rather than a guess, so the
          cost is a dropped read and not a wrong barcode. Worth reporting; not worth refusing
          to run over.

        The scan escalates rather than going straight to the wider bound. Deletion
        neighbourhoods grow combinatorially in the distance, so scanning at
        ``2 * max_errors`` is markedly more expensive than at ``max_errors``, and a whitelist
        that already fails the narrow bound has nothing more to learn from the wide one.

        Raises:
            ValueError: If a whitelist holds two entries within ``max_errors`` of each other,
                the pair is not exempted, and the chemistry's policy enforces the bound.
        """
        policy = self.whitelist_distance_policy()
        whitelists = self.whitelists

        for component in self.read_structure:
            budget = self.match_errors_for(component)
            whitelist = whitelists.get(component.name)

            if budget is None or budget < 1 or whitelist is None or len(whitelist) < 2:
                continue

            within_budget = close_whitelist_pairs(whitelist, budget)
            unexempted = {
                pair: distance
                for pair, distance in within_budget.items()
                if frozenset(pair) not in policy.exempt_pairs
            }

            if unexempted:
                pair, distance = min(unexempted.items(), key=lambda item: item[1])
                message = (
                    f"Whitelist for component '{component.name}' holds {len(unexempted)} pair(s) "
                    f"of entries within its error budget of {budget}, the closest being "
                    f"'{pair[0]}' and '{pair[1]}' at edit distance {distance}. A single error "
                    "inside the budget turns one of these valid entries into the other, which "
                    "then matches exactly at its expected position and is reported as a perfect "
                    "match, so the misassignment cannot be detected downstream. Retire one entry "
                    f"of each pair so that every pair is more than {budget} edits apart."
                )
                if policy.enforce:
                    raise ValueError(message)
                warn_once(message)
            elif within_budget:
                pair, distance = min(within_budget.items(), key=lambda item: item[1])
                # Every offending pair is a recorded exemption. Say so on every run anyway: the
                # reads such a pair mis-attributes are indistinguishable from correct ones, so
                # this line is the only place the blind spot surfaces at all.
                warn_once(
                    f"Whitelist for component '{component.name}' holds {len(within_budget)} "
                    "pair(s) of entries within its error budget of "
                    f"{budget}, all of them declared exemptions, the closest being '{pair[0]}' "
                    f"and '{pair[1]}' at edit distance {distance}. A single error inside the "
                    "budget silently turns one of these valid entries into the other and is "
                    "reported as a perfect match. Reads carrying it are misattributed and "
                    "cannot be identified after the fact."
                )

            # The wider bound is informational: a window equally close to two entries is
            # detectable, and is now a terminal ambiguity verdict, so it costs a dropped read
            # rather than a wrong barcode. It is reported at debug both for that reason and
            # because the deletion neighbourhood it needs grows combinatorially in the bound --
            # scanning at twice a budget of two is an order of magnitude dearer than at the
            # budget itself, which is not worth paying on every import to say nothing new.
            if log.isEnabledFor(logging.DEBUG):
                wider = close_whitelist_pairs(whitelist, 2 * budget)
                if wider:
                    pair, distance = min(wider.items(), key=lambda item: item[1])
                    log.debug(
                        f"Whitelist for component '{component.name}' holds {len(wider)} pair(s) "
                        f"of entries within {2 * budget} edits, the closest being '{pair[0]}' "
                        f"and '{pair[1]}' at edit distance {distance}. A read window can sit "
                        "equally close to both entries of such a pair, which is unresolvable "
                        "and reported as an ambiguous match rather than corrected."
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

        # Validate that every whitelist is spread widely enough for the budget it is used with
        self.validate_whitelist_distances()

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
