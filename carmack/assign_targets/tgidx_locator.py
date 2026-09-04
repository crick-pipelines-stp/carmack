"""Bounded trim-window derivation for the target index.

Pure coordinate arithmetic with no I/O and no header parsing: the caller supplies the
anchor run start it has already read from the annotated header, and these functions
return the short slice of the read the target index must lie within, so matching never
scans a whole read.

The two steps are deliberately separate. :func:`homopolymer_run_end` needs the anchor
base, to count the run forward. :func:`locate_tgidx_window` needs only the run end that
scan produces, because once the run's right edge is known the window is arithmetic on
lengths alone.
"""

from dataclasses import dataclass

from carmack.chemistry.chemistry_base import TGIDX_MAX_LEADING_ANCHOR
from carmack.utils import homopolymer_run_length


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


def homopolymer_run_end(seq: str, run_start: int, base: str) -> int:
    """Return the exclusive end of the maximal homopolymer run covering ``run_start``.

    The run's left edge is fuzzy and its right edge is not. A UMI whose final base is the
    anchor base makes the recorded run start latch one base early, and both readings are
    self-consistent, so the start cannot be trusted to a single position. Counting the
    anchor base forward lands on the same index wherever within the run the count began,
    so the end is exact, and it is the point the window hangs off.

    Args:
        seq: The read sequence.
        run_start: 0-based index at or after which the run begins, as recorded by the end
            of the UMI span.
        base: The anchor base the run repeats.

    Returns:
        The 0-based index of the first position after the run, half-open. Equal to
        ``run_start`` when no anchor base sits there, and ``len(seq)`` when the run
        continues to the end of the read.
    """
    return run_start + homopolymer_run_length(seq, run_start, base)


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
    arithmetic on lengths alone; see :func:`homopolymer_run_end` for the scan that
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
