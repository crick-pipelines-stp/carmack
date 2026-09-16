from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_component import ReadComponent, ReadComponentType


def best_window(
    read: str, entry: str, expected_start: int, max_errors: int, lower_bound: int = 0
) -> tuple[bool, int, int, int]:
    """
    Find the window of ``read`` that best represents ``entry`` near ``expected_start``.

    Both searching matchers locate a whitelist entry approximately -- one by a seed k-mer, the
    other by a local alignment -- and then have to report the entry's *extent* in the read.
    They share this function so they cannot disagree about that extent, because everything
    measured off a component's edge is measured off whatever they report: the adjacent spacer
    evidence the ambiguity tie-break depends on, and the UMI start, which is taken off the
    preceding barcode's span end.

    Windows are enumerated over the small band the error budget permits and ranked lowest-first
    on ``(edit distance, deviation of the window length from the entry's, distance of the
    window start from expected_start)``. The second and third terms are the point. A window one
    base short of the entry is inside the budget and reaches the *same* edit distance as the
    full-length window when the error sits at the entry's last base, so reporting the first
    window that reaches the minimum returns a 9bp span for a 10bp barcode -- which moves the
    component's 3' boundary, sends the spacer check a base off, and costs the true candidate a
    tie-break it should have won.

    Args:
        read: The sequencing read to take windows from.
        entry: The whitelist entry being measured, whose length sets the target window length.
        expected_start: Where in ``read`` the entry is thought to begin. Windows are searched
            within ``max_errors`` of it, since that is as far as the budget lets the true start
            drift from an approximate location.
        max_errors: Maximum edit distance allowed between a window and ``entry``.
        lower_bound: Earliest position in ``read`` a window may begin at. Defaults to 0, which
            bounds nothing. A caller that has already restricted where a match may begin
            passes that restriction here, so recovering a clipped base cannot quietly reach
            back past it.

    Returns:
        ``(is_valid, start, end, edit_distance)``. ``is_valid`` is False, with a span of
        ``(-1, -1)``, when no window in the band is within the budget; ``edit_distance`` is
        then the closest distance reached, for logging.
    """
    entry_len = len(entry)

    # Search within a small band around the expected start
    min_start = max(0, lower_bound, expected_start - max_errors)
    max_start = min(len(read) - 1, expected_start + max_errors)

    # Only consider window sizes where |win_len - entry_len| <= max_errors
    # This is the key pruning: window length alone must allow <= max_errors
    min_len = max(1, entry_len - max_errors)
    max_len = min(len(read), entry_len + max_errors)

    best_dist = max_errors + 1
    # Rank of the best window so far, lowest wins. None until a window inside the budget is seen.
    best_rank: tuple[int, int, int] | None = None
    best_span = (-1, -1)

    for start in range(min_start, max_start + 1):
        max_len_at_start = min(max_len, len(read) - start)
        if max_len_at_start < min_len:
            continue

        # Only iterate window lengths within feasible range
        for win_len in range(min_len, max_len_at_start + 1):
            dist = edit_distance(read[start : start + win_len], entry, "N", True)
            best_dist = min(best_dist, dist)

            if dist > max_errors:
                continue

            # Early exit: a zero-distance window cannot be improved on. Its length already
            # equals the entry's, since any length difference costs at least one edit.
            if dist == 0:
                return (True, start, start + win_len, 0)

            rank = (dist, abs(win_len - entry_len), abs(start - expected_start))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_span = (start, start + win_len)

    if best_rank is not None:
        return (True, best_span[0], best_span[1], best_rank[0])

    return (False, -1, -1, best_dist)


class UnresolvedComponentStartError(ValueError):
    """
    Raised when an operation needs a fixed start position the component does not have.

    ``ReadStructure.compute_start_positions`` can only resolve offsets while every preceding
    component has a known length. Once a variable-length component is reached, ``start`` is left
    as ``None`` for that component and for every component following it. A matcher may therefore
    hold a component with no fixed start, so helpers that assume one must say so explicitly
    rather than fail on arithmetic against ``None``.

    Subclasses ``ValueError`` because an unresolved start is a bad-value condition, which also
    keeps any existing broad ``except ValueError`` handling working unchanged.
    """


class MatcherBase(ABC):
    """
    Abstract base class for barcode matchers.

    Subclasses must implement the match method to perform barcode matching
    based on the provided whitelist and read structure.

    Attributes:
        allowed_component_types: The component types this matcher can structurally handle.
            It is a class attribute rather than a constructor argument deliberately, so that a
            caller cannot widen it by accident; a subclass that can genuinely handle another
            component type overrides it.
    """

    allowed_component_types: ClassVar[frozenset[ReadComponentType]] = frozenset(
        {ReadComponentType.BARCODE}
    )

    def __init__(
        self,
        whitelist: tuple[str, ...],
        component: ReadComponent,
        chemistry: ChemistryBase,
    ):
        self.whitelist_set = frozenset(whitelist)
        self.component = component
        self.chemistry = chemistry

        # Check the component is a type this matcher class declares it can handle
        allowed = type(self).allowed_component_types
        if self.component.type not in allowed:
            permitted = ", ".join(sorted(allowed))
            raise ValueError(
                f"{type(self).__name__} cannot handle component {self.component} of type "
                f"{self.component.type}; permitted component types: {permitted}"
            )

    @abstractmethod
    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read.

        Args:
            read: The sequencing read to match against.
            start_idx: Bounds where in the read this matcher looks, as a hint from the caller that
                everything before it has already been consumed. *What* it bounds is
                matcher-specific and is documented on each implementation: for some it is a hard
                floor on where a match may begin, for others only a floor on where the search is
                seeded, in which case a returned match is **not** guaranteed to begin at or after
                it. Defaults to 0, which imposes no bound.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results. Ambiguity can be
            represented by multiple attempts with the same candidate but different matches. Every
            ``read_idx`` is in original-read coordinates regardless of ``start_idx``, so a caller
            never has to add the offset back.

        Note:
            The two searching matchers enforce ``start_idx`` differently and are deliberately not
            unified: ``KmerMatcher`` treats it as a floor on seeding only, while
            ``AlignmentMatcher`` trims the read to it and so treats it as a floor on the match
            itself. Changing either would move existing barcode calls, so the divergence is
            recorded here rather than reconciled.
        """
        pass

    # Helper funcs
    def check_read_len(self, read: str) -> bool:
        """
        Check if the read is long enough to contain the component.

        Requires a resolved start position, since the component's extent in the read is its
        start plus its length.

        Args:
            read: The sequencing read to measure.

        Returns:
            True if the read extends to the end of the component, False otherwise.

        Raises:
            UnresolvedComponentStartError: If the component has no resolved start position.
        """
        if self.component.start is None:
            raise UnresolvedComponentStartError(
                f"Component {self.component.name} has no resolved start position, so its extent "
                "in the read cannot be computed. A component without a resolved start must be "
                "located by a matcher that does not rely on a fixed start."
            )
        end = self.component.start + self.component.length
        return len(read) >= end

    def component_window(self, read: str, max_errors: int) -> tuple[int, int]:
        """
        Return the half-open region of ``read`` this component may be located in.

        A component is bounded to its own extent widened, either side, by everything that can
        have moved it: the drift its chemistry's layout permits, plus the error budget the
        caller is searching with. Drift accounts for the read the matcher is handed being
        displaced -- an indel in any component ahead of this one shifts it -- and the budget
        accounts for the match itself being allowed to start a base or two out. Nothing else
        can put the component anywhere, so nothing else belongs in the bound.

        Bounding by the gap to the neighbouring barcode instead, as this once did, measured
        the wrong thing everywhere rather than only where it ran out of read. The distance to
        the next barcode is the width of the primer between them; it says how much room a
        component *could* occupy without colliding with its neighbour, not how far anything
        can have pushed it. It was far too generous in both directions: a two-sided window
        admitted a valid whitelist entry planted twenty bases from where its component belongs
        just as readily as the unbounded side did. And ``get_next(..., bc_only=True)`` is
        ``None`` for the last barcode of every shipped chemistry, so that side fell through to
        ``len(read)`` and the last barcode was searched across the UMI, the poly-G anchor, the
        target index, the Mosaic End and the whole cDNA insert -- a stretch in which a 96-entry
        10bp whitelist finds a chance exact match often enough to fabricate cell barcodes, and
        an exact chance match beats the component's own error-bearing window outright, so no
        tie is formed and no tie-break or spacer check is ever reached.

        The component's own length is added on the high side because the window bounds a
        *span*, not a start. A component drifted the full tolerance begins at the ceiling the
        tolerance sets and still occupies its own length behind that, so a ceiling without the
        length would cut the very read the tolerance was computed to admit.

        Args:
            read: The sequencing read being searched, used only for its length.
            max_errors: The error budget the caller matches this component with.

        Returns:
            ``(low, high)``, a half-open slice of ``read``, clamped to it. Spans the whole read
            when the layout makes no positional prediction for this component, since there is
            then nothing to measure a displacement from -- a target index, for instance, is
            located by the anchor run in front of it rather than by an offset, and that anchor
            has already bounded it before a matcher sees it.
        """
        expected_start = self.component.start
        drift = self.chemistry.drift_tolerance(self.component)
        # Having no resolved start and having no drift prediction are one fact with two
        # symptoms: both fall out of the same walk stopping at the first variable-length
        # component. Tested together so a type checker sees both narrowed at once.
        if expected_start is None or drift is None:
            return (0, len(read))

        tolerance = drift + max_errors
        low = expected_start - tolerance
        high = expected_start + self.component.length + tolerance

        return (max(0, min(low, len(read))), max(0, min(high, len(read))))

    def requires_spacer_evidence(self, read_idx: tuple[int, int], max_errors: int) -> bool:
        """
        Whether a candidate that arrived alone has to be corroborated by an adjacent spacer.

        Arriving alone is not itself evidence, so a lone candidate is not simply waved
        through. But what a lone candidate still owes depends on how well it already agrees
        with the read structure, and two different things can leave it the only one standing.

        A candidate sitting where the layout can account for it is already corroborated by its
        position, which is the prediction the read structure makes. Demanding a spacer as well
        would throw reads away for a reason unrelated to their barcode: an adjacent primer is
        22bp and has to match exactly, so a single error anywhere in it removes the evidence,
        and reads reaching a searching matcher at all are the error-laden ones.

        A candidate found beyond that is a different claim -- that the component is somewhere
        the structure cannot put it -- and needs something beyond itself to support it.
        Requiring a flanking spacer there is what closes the path that assigned a barcode read
        off a neighbouring component's sequence or off the insert.

        The threshold is the larger of the drift the layout permits and the budget the
        component is matched with, never their sum. Keyed on the sum it would be vacuous:
        ``component_window`` admits a full-length span at exactly that displacement and
        nothing past it, so the rule could never fire on a full-length candidate. Keyed on the
        drift alone it would be tighter than the budget wherever the layout permits no drift
        at all, which would refuse a first barcode found one base out with no upstream
        component able to have moved it. The larger of the two is never tighter than either,
        and it leaves a real band -- displacement in ``(drift, drift + max_errors]`` -- where
        the window admits a candidate and this rule still refuses it.

        A component the layout cannot place at all is exempt: where the structure predicts no
        position, position neither corroborates nor contradicts, so there is nothing for a
        displacement rule to measure. The exemption is load-bearing rather than merely tidy.
        A target index sits behind a homopolymer and in front of a Mosaic End shipped with
        ``verify`` false, so its spacer check returns nothing on every read by construction;
        demanding evidence would be unanswerable, and every read would be reported as carrying
        no target at all. Such a component is bounded before it reaches a matcher, by the
        anchor run that locates it, and that bound is its corroboration.

        Args:
            read_idx: The candidate's span, in original-read coordinates.
            max_errors: The error budget this candidate was matched with. Passed in rather
                than read off the matcher because this class owns no budget, one subclass has
                none at all, and a caller matching a target index passes that component's
                budget rather than a barcode's.

        Returns:
            True when the candidate must carry at least one adjacent spacer to be assigned.
        """
        drift = self.chemistry.drift_tolerance(self.component)
        expected_start = self.component.start
        # Both halves of the same fact again, so the second can only be reached with the
        # first. Named alongside it so the comparison below is narrowed to two integers
        # rather than measured against a possible None.
        if drift is None or expected_start is None:
            return False

        return abs(read_idx[0] - expected_start) > max(drift, max_errors)

    def check_spacers(
        self, read: str, match_idx: tuple[int, int]
    ) -> dict[Literal["upstream", "downstream"], str | None]:
        """Check if appropriate spacer sequence(s) are present adjacent to the barcode match."""

        # Get adjacent spacer regions based on read structure
        spacer_components: dict[Literal["upstream", "downstream"], ReadComponent | None] = {
            "upstream": self.chemistry.read_structure.get_previous(self.component),
            "downstream": self.chemistry.read_structure.get_next(self.component),
        }

        def match_seq(
            loc: Literal["upstream", "downstream"], spacer_component: ReadComponent | None
        ) -> str | None:
            if (
                spacer_component
                and spacer_component.type in (ReadComponentType.PRIMER, ReadComponentType.OTHER)
                and spacer_component.sequence
                and spacer_component.verify
            ):
                spacer_seq = None
                if loc == "upstream":
                    if match_idx[0] >= spacer_component.length:
                        spacer_seq = read[match_idx[0] - spacer_component.length : match_idx[0]]
                elif loc == "downstream":
                    if len(read) - match_idx[1] >= spacer_component.length:
                        spacer_seq = read[match_idx[1] : match_idx[1] + spacer_component.length]

                if spacer_seq is not None and spacer_seq == spacer_component.sequence:
                    return (
                        spacer_component.name
                    )  # Return spacer name if present and matches expected sequence

            return None  # No defined spacer sequence, treat as not present

        is_present = {}
        for loc, spacer_comp in spacer_components.items():
            is_present[loc] = match_seq(loc, spacer_comp)

        return is_present
