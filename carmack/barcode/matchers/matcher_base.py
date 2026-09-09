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

        A component is bounded to its own inter-barcode interval: from the expected end of the
        preceding barcode to the expected start of the following one. The interval is the
        natural allowance rather than an arbitrary margin, because what separates two barcodes
        in the read structure is exactly the room an indel upstream can push a component into,
        while a bound stopping short of the neighbour still refuses to look inside another
        barcode's window.

        That refusal is the point. Unbounded, a whitelist entry that happens to be a rotation
        of a *neighbouring* component's entry sits in that neighbour's region on every single
        read of a library using it -- and being a rotation, it is often an exact match there
        while the component's own damaged window is one edit out, so it wins on distance
        outright and no tie-break or spacer check is ever reached. The result is a cell barcode
        read off another component's sequence, with a span overlapping that component's.

        The interval is relaxed so that the component's own extent widened by the error budget
        always fits, so the bound can never cut a window that is legitimately in budget.

        Args:
            read: The sequencing read being searched, used only for its length.
            max_errors: The error budget the caller matches this component with.

        Returns:
            ``(low, high)``, a half-open slice of ``read``. Spans the whole read when the
            component has no resolved start, since there is then no expected position to
            measure an interval from -- a target index, for instance, is located by an anchor
            run rather than by an offset.
        """
        expected_start = self.component.start
        if expected_start is None:
            return (0, len(read))

        structure = self.chemistry.read_structure
        previous = structure.get_previous(self.component, bc_only=True)
        following = structure.get_next(self.component, bc_only=True)

        low = 0
        if previous is not None and previous.start is not None:
            low = previous.start + previous.length

        high = len(read)
        if following is not None and following.start is not None:
            high = following.start

        low = min(low, max(0, expected_start - max_errors))
        high = max(high, expected_start + self.component.length + max_errors)

        return (max(0, min(low, len(read))), max(0, min(high, len(read))))

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
