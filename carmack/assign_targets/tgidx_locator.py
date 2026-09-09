"""Bounded trim-window derivation for the target index.

Pure coordinate arithmetic with no I/O and no header parsing: the caller supplies the
anchor run start it has already read from the annotated header, and these functions
return the short slice of the read the target index must lie within, so matching never
scans a whole read.

The two steps are deliberately separate. :func:`locate_anchor_run` needs the anchor
base, to count the run forward. :func:`locate_tgidx_window` needs only the run end that
scan produces, because once the run's right edge is known the window is arithmetic on
lengths alone.
"""

from dataclasses import dataclass

from carmack.chemistry.chemistry_base import TGIDX_MAX_LEADING_ANCHOR
from carmack.utils import homopolymer_run_length

# Furthest forward the anchor run's start is searched for when the offset the chemistry
# predicts does not land on it. Two bases covers a read carrying one or two inserted
# bases upstream of the run; beyond that the read has diverged from the layout far
# enough that a window cut anywhere is a guess, and a zero-length run reported honestly
# is better than a plausible window scored against arbitrary sequence.
ANCHOR_RUN_MAX_SHIFT = 2


@dataclass(frozen=True)
class TrimWindow:
    """The bounded slice of a read the target index must lie within.

    Attributes:
        sequence: The window slice taken from the read. Shorter than the nominal width
            only when the read ends inside it.
        start: 0-based index in the read where ``sequence`` begins. Window coordinates
            become read coordinates by adding this, so a match spanning ``(a, b)`` in
            ``sequence`` spans ``(start + a, start + b)`` in the read.
        floor: The shortest window the matcher could still find a match in,
            ``max(k, tgidx_len - max_errors)``: verifying against the index needs
            ``tgidx_len - max_errors`` bases, and seeding needs ``k`` whichever is
            larger. Carried so a rejected window can be reported against the number it
            failed rather than as a bare boolean.
    """

    sequence: str
    start: int
    floor: int

    @property
    def is_matchable(self) -> bool:
        """Return whether the window is long enough to hand to the matcher.

        ``KmerMatcher.collect_candidates`` raises when its input is shorter than its seed
        length, so a caller tests this and counts the read as unmatched instead of calling
        the matcher and catching the error.
        """
        return len(self.sequence) >= self.floor


@dataclass(frozen=True)
class AnchorRun:
    """The anchor homopolymer run a target window is cut from.

    Attributes:
        start: 0-based index the run was found to begin at, at or 3' of the start
            the chemistry predicted.
        end: Exclusive end of the run, half-open. Equal to ``start`` when no run was
            found within the search bound.
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the observed run length, zero when no run was found."""
        return self.end - self.start


def locate_anchor_run(
    seq: str,
    expected_start: int,
    base: str,
    min_run: int,
    max_shift: int = ANCHOR_RUN_MAX_SHIFT,
) -> AnchorRun:
    """Return the anchor homopolymer run at or just 3' of a predicted start.

    The run's left edge is fuzzy and its right edge is not. The predicted start comes
    from the chemistry -- a recorded anchor's span end plus a fixed offset -- so it is
    exact only for a read that matches the layout base for base. Landing *inside* the
    run needs no correction at all, because counting the anchor base forward reaches
    the same end wherever within the run the count began; that is the property the
    window hangs off, and it is why a UMI whose final base is the anchor base costs
    nothing. Landing *before* the run does need correcting: the count stops
    immediately, and the window would be cut from the predicted coordinate with the
    index sitting outside it.

    So when the predicted base is not the anchor base the run start is searched for
    forward, never backward. A backward search would have nothing to find that the
    forward count does not already absorb. ``min_run`` copies are required at the
    shifted offset, which is what stops the search sliding onto a target index's own
    leading anchor bases: a whitelist entry may open with at most
    ``TGIDX_MAX_LEADING_ANCHOR`` of them, a bound the chemistry validates and which is
    below every real ``min_run``.

    Args:
        seq: The read sequence.
        expected_start: 0-based index the chemistry predicts the run begins at.
        base: The anchor base the run repeats.
        min_run: Copies of ``base`` that must be present at a shifted offset for it to
            be taken as the run start.
        max_shift: Furthest forward the run start is searched for.

    Returns:
        The run found, or a zero-length run at ``expected_start`` when none was found
        within the bound.
    """
    length = homopolymer_run_length(seq, expected_start, base)
    if length:
        return AnchorRun(start=expected_start, end=expected_start + length)

    run = base * min_run
    for shift in range(1, max_shift + 1):
        start = expected_start + shift
        if seq[start : start + min_run] == run:
            return AnchorRun(start=start, end=start + homopolymer_run_length(seq, start, base))

    return AnchorRun(start=expected_start, end=expected_start)


def locate_tgidx_window(
    seq: str,
    run_end: int,
    tgidx_len: int,
    max_errors: int,
    max_leading_anchor: int = TGIDX_MAX_LEADING_ANCHOR,
    k: int = 4,
) -> TrimWindow:
    """Return the bounded window the target index must lie within.

    The window spans ``[run_end - max_leading_anchor - max_errors, run_end + tgidx_len +
    max_errors)``, clamped to the read. The left margin absorbs the run end overshooting
    the index start: an index beginning with up to ``max_leading_anchor`` anchor bases has
    those bases counted into the run, and one further base of slack covers an error in a
    leading base. The right margin holds an index that begins a base late. For the current
    chemistry the window is a constant twelve bases exposing at most three anchor bases,
    so an index beginning with the anchor base has nowhere to slide along the run.

    The anchor base is not needed here. Once the run's end is known the window is
    arithmetic on lengths alone; see :func:`locate_anchor_run` for the scan that
    produces it.

    Args:
        seq: The read sequence.
        run_end: Exclusive end of the anchor homopolymer run, in read coordinates.
        tgidx_len: Length of the target index component.
        max_errors: Edit budget the index is matched at, which widens both margins.
        max_leading_anchor: Most anchor bases a whitelisted index may begin with, and so
            the furthest the run end can overshoot the index start. Defaults to the bound
            the chemistry validates its whitelist against.
        k: Seed length the window must be able to hold. Defaults to 4, tracking
            ``KmerMatcher``'s own default ``k``; it is a parameter rather than an import
            so this module does not depend on the matcher package.

    Returns:
        The window slice, its start in read coordinates, and the floor its length is
        judged against. A window below that floor reports ``is_matchable`` false and must
        not be handed to the matcher.
    """
    start = min(max(0, run_end - max_leading_anchor - max_errors), len(seq))
    stop = min(len(seq), run_end + tgidx_len + max_errors)
    return TrimWindow(sequence=seq[start:stop], start=start, floor=max(k, tgidx_len - max_errors))
