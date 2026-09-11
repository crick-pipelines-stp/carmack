"""Tests for the target-index locator.

The locator turns a read plus an observed anchor homopolymer run into the slice of
that read where the target index must lie. Two facts about the read shape the design
and therefore these tests.

First, the anchor run's left edge is ambiguous but its right edge is exact. Upstream
anchoring scans offsets upward and returns the first position at which the minimum run
of anchor bases begins, so a UMI whose final bases are the anchor base makes the
recorded run start latch early. Counting the anchor base forward from any position
inside the run lands on the same index, so the run END is the stable reference point
and the window hangs off it.

Second, the window is constant-size rather than growing with slippage. Its left margin
absorbs an index that itself opens with up to ``max_leading_anchor`` anchor bases,
which get counted into the run, plus one base of error slack. Its right margin holds
an index that begins a base late.

``TrimWindow.is_matchable`` exists so a caller can skip the matcher rather than call it
and catch the error it raises on input shorter than its seed length.
"""

from inspect import signature

import pytest
from assertpy import assert_that

from carmack.assign_targets.tgidx_locator import (
    ANCHOR_RUN_MAX_BRIDGED,
    ANCHOR_RUN_MAX_SHIFT,
    TrimWindow,
    locate_anchor_run,
    locate_tgidx_window,
    run_end_bridging_interruptions,
)
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.chemistry_base import TGIDX_MAX_LEADING_ANCHOR
from carmack.chemistry.chemistry_factory import ChemistryFactory

CHEMISTRY = "carmack_custom_seq_1_0"

# The shipped layout the locator is tuned for: an 8bp index matched at one error.
TGIDX_LEN = 8
MAX_ERRORS = 1

# Consequences of that layout, spelled out so a change to either constant is visible
# here rather than only inside the locator. The window spans max_leading_anchor +
# max_errors bases before the run end and tgidx_len + max_errors after it.
WINDOW_WIDTH = 12
WINDOW_LEAD = 3
WINDOW_FLOOR = 7

ANCHOR_BASE = "G"

# Copies of the anchor base a shifted offset must present before it is taken as the run
# start. Tracks the shipped chemistry's POLYG min_run.
MIN_RUN = 3

# Filler either side of the anchor run, deliberately free of the anchor base so that
# every run in a synthetic read is exactly the run the test placed there.
PREFIX = "ACTACTACTACT"
TAIL = "CTCTTATACACATCTCCTC"

# The shipped whitelist entry, which opens with no anchor bases, and the two synthetic
# variants that open with one and with two. Leading anchor bases are indistinguishable
# from the run they follow, so they are counted into it and pull the run end rightward.
INDEX_NO_LEADING_ANCHOR = "TATAGCCT"
INDEX_ONE_LEADING_ANCHOR = "GTATAGCC"
INDEX_TWO_LEADING_ANCHOR = "GGTATAGC"

# A minority of real reads carry one non-anchor base between the run end and the index, so
# the index begins a base later than the run end predicts. The window's right margin is
# what holds it.
INTERVENING_BASE = "A"

# UMIs whose final bases are the anchor base. These are what make the recorded run
# start ambiguous: each trailing anchor base is a position at which upstream anchoring
# could equally well decide the run begins.
UMI_ONE_TRAILING_ANCHOR = "ACTACTAG"
UMI_TWO_TRAILING_ANCHOR = "ACTACTGG"
UMI_LEN = 8


def build_read(run_len: int, index: str = INDEX_NO_LEADING_ANCHOR, tail: str = TAIL) -> str:
    """Assemble a read carrying one anchor run followed by an index and a tail.

    Args:
        run_len: Number of anchor bases placed after the prefix.
        index: Target index sequence placed immediately after the run.
        tail: Anchor-free sequence closing the read.

    Returns:
        The assembled read sequence, whose anchor run begins at ``len(PREFIX)``.
    """
    return PREFIX + ANCHOR_BASE * run_len + index + tail


def build_umi_read(umi: str, run_len: int) -> str:
    """Assemble a read whose UMI abuts the anchor run, blurring the run's left edge.

    Args:
        umi: UMI sequence placed at ``len(PREFIX)``, ending in one or more anchor bases.
        run_len: Number of anchor bases placed after the UMI.

    Returns:
        The assembled read sequence.
    """
    return PREFIX + umi + ANCHOR_BASE * run_len + INDEX_NO_LEADING_ANCHOR + TAIL


def window_after_run(seq: str, run_start: int, **kwargs) -> TrimWindow:
    """Locate the index window for the anchor run beginning at ``run_start``.

    Mirrors how a caller drives the two functions together: resolve the run end first,
    then hang the window off it.

    Args:
        seq: The read sequence.
        run_start: Any position inside the anchor run.
        **kwargs: Overrides forwarded to ``locate_tgidx_window``.

    Returns:
        The located window.
    """
    run_end = locate_anchor_run(seq, run_start, ANCHOR_BASE, MIN_RUN).end
    kwargs.setdefault("tgidx_len", TGIDX_LEN)
    kwargs.setdefault("max_errors", MAX_ERRORS)
    return locate_tgidx_window(seq, run_end, **kwargs)


class TestLocateAnchorRun:
    """Tests for locating an anchor homopolymer run against a predicted start."""

    @pytest.mark.parametrize("run_len", [3, 4, 5, 6, 7, 8])
    def test_run_end_is_start_plus_run_length(self, run_len: int) -> None:
        """Test that the exclusive end sits one past the last anchor base of the run."""
        seq = build_read(run_len)
        run_start = len(PREFIX)

        result = locate_anchor_run(seq, run_start, ANCHOR_BASE, MIN_RUN).end

        assert_that(result).is_equal_to(run_start + run_len)

    @pytest.mark.parametrize("run_len", [3, 4, 5, 6, 7, 8])
    def test_run_end_identical_across_both_starts_when_umi_ends_in_one_anchor_base(
        self, run_len: int
    ) -> None:
        """Test that a run end is unchanged when a trailing UMI anchor base latches it early.

        The UMI's last base is the anchor base, so upstream anchoring finds its minimum
        run of anchor bases one position early and records the run as starting there.
        Both readings describe the same run, and the run end must not care which was
        taken.
        """
        seq = build_umi_read(UMI_ONE_TRAILING_ANCHOR, run_len)
        umi_start = len(PREFIX)
        honest_start = umi_start + UMI_LEN
        latched_start = umi_start + UMI_LEN - 1

        honest_end = locate_anchor_run(seq, honest_start, ANCHOR_BASE, MIN_RUN).end
        latched_end = locate_anchor_run(seq, latched_start, ANCHOR_BASE, MIN_RUN).end

        assert_that(honest_end).is_equal_to(latched_end)
        assert_that(honest_end).is_equal_to(umi_start + UMI_LEN + run_len)

    @pytest.mark.parametrize("run_len", [3, 4, 5, 6, 7, 8])
    def test_run_end_identical_across_three_starts_when_umi_ends_in_two_anchor_bases(
        self, run_len: int
    ) -> None:
        """Test that a run end is unchanged across every self-consistent start reading.

        Two trailing UMI anchor bases give three positions from which the run can be
        read as beginning, all of them internally consistent. One end must serve all
        three.
        """
        seq = build_umi_read(UMI_TWO_TRAILING_ANCHOR, run_len)
        umi_start = len(PREFIX)
        starts = [umi_start + UMI_LEN - 2, umi_start + UMI_LEN - 1, umi_start + UMI_LEN]

        ends = {locate_anchor_run(seq, start, ANCHOR_BASE, MIN_RUN).end for start in starts}

        assert_that(ends).is_length(1)
        assert_that(ends.pop()).is_equal_to(umi_start + UMI_LEN + run_len)

    def test_run_end_clamps_to_read_end_when_run_reaches_it(self) -> None:
        """Test that a run running off the end of the read ends at the read end."""
        seq = PREFIX + ANCHOR_BASE * 5

        result = locate_anchor_run(seq, len(PREFIX), ANCHOR_BASE, MIN_RUN).end

        assert_that(result).is_equal_to(len(seq))

    def test_run_end_handles_run_longer_than_the_index(self) -> None:
        """Test that an unusually long run is measured in full, with no length cap.

        Real reads carry runs of up to eighteen anchor bases, so nothing here may
        assume the run is bounded by the index length or by any other constant.
        """
        run_len = 18
        seq = build_read(run_len)

        result = locate_anchor_run(seq, len(PREFIX), ANCHOR_BASE, MIN_RUN).end

        assert_that(result).is_equal_to(len(PREFIX) + run_len)


class TestLocateAnchorRunForwardSearch:
    """Tests for the bounded forward search when the predicted start misses the run.

    The predicted start is chemistry arithmetic over a recorded anchor span, so it is
    exact only for a read that matches the layout base for base. Landing inside the run
    is absorbed for free by counting forward; landing before it is what this search is
    for. Only forward, because a backward search would find nothing the forward count
    does not already reach.
    """

    @pytest.mark.parametrize("shift", [1, 2])
    def test_run_starting_late_is_found_and_measured_in_full(self, shift: int) -> None:
        """A run pushed later by an upstream insertion is still located exactly."""
        run_len = 6
        seq = PREFIX + "C" * shift + ANCHOR_BASE * run_len + INDEX_NO_LEADING_ANCHOR + TAIL
        predicted = len(PREFIX)

        run = locate_anchor_run(seq, predicted, ANCHOR_BASE, MIN_RUN)

        assert_that(run.start).is_equal_to(predicted + shift)
        assert_that(run.end).is_equal_to(predicted + shift + run_len)
        assert_that(run.length).is_equal_to(run_len)

    def test_run_beyond_the_shift_bound_reports_no_run(self) -> None:
        """Past the bound the read has diverged too far to guess a window from."""
        seq = PREFIX + "C" * (ANCHOR_RUN_MAX_SHIFT + 1) + ANCHOR_BASE * 6 + TAIL
        predicted = len(PREFIX)

        run = locate_anchor_run(seq, predicted, ANCHOR_BASE, MIN_RUN)

        assert_that(run.start).is_equal_to(predicted)
        assert_that(run.length).is_equal_to(0)

    def test_fewer_than_min_run_copies_at_the_shift_is_not_taken(self) -> None:
        """A short smear of anchor bases is not the run and must not be latched onto.

        This is what keeps the search off a target index's own leading anchor bases:
        a whitelist entry may open with at most ``TGIDX_MAX_LEADING_ANCHOR`` of them,
        which is below every real ``min_run``.
        """
        seq = PREFIX + "C" + ANCHOR_BASE * (MIN_RUN - 1) + "C" + TAIL
        predicted = len(PREFIX)

        run = locate_anchor_run(seq, predicted, ANCHOR_BASE, MIN_RUN)

        assert_that(run.length).is_equal_to(0)

    def test_predicted_start_inside_the_run_needs_no_search(self) -> None:
        """A start already inside the run is taken as-is, not shifted forward."""
        run_len = 6
        seq = build_read(run_len)
        inside = len(PREFIX) + 2

        run = locate_anchor_run(seq, inside, ANCHOR_BASE, MIN_RUN)

        assert_that(run.start).is_equal_to(inside)
        assert_that(run.end).is_equal_to(len(PREFIX) + run_len)

    def test_predicted_start_past_the_read_end_reports_no_run(self) -> None:
        """A truncated read yields a zero-length run rather than raising."""
        seq = PREFIX + ANCHOR_BASE * 4
        predicted = len(seq) + 5

        run = locate_anchor_run(seq, predicted, ANCHOR_BASE, MIN_RUN)

        assert_that(run.length).is_equal_to(0)


class TestLocateTgidxWindow:
    """Tests for the constant-size window hung off the anchor run end."""

    @pytest.mark.parametrize("run_len", [3, 4, 5, 6, 7, 8])
    def test_window_is_constant_size_regardless_of_run_length(self, run_len: int) -> None:
        """Test that slippage moves the window but never stretches it.

        The run end absorbs the slippage, so a longer run shifts the window rightward
        by exactly as much and leaves its width alone.
        """
        seq = build_read(run_len)
        run_end = len(PREFIX) + run_len

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.sequence).is_length(WINDOW_WIDTH)
        assert_that(window.start).is_equal_to(run_end - WINDOW_LEAD)
        assert_that(window.is_matchable).is_true()

    @pytest.mark.parametrize(
        "index,expected_offset",
        [
            (INDEX_NO_LEADING_ANCHOR, 3),
            (INDEX_ONE_LEADING_ANCHOR, 2),
            (INDEX_TWO_LEADING_ANCHOR, 1),
        ],
    )
    def test_window_contains_whole_index_for_every_allowed_leading_anchor_count(
        self, index: str, expected_offset: int
    ) -> None:
        """Test that the window covers the index whatever its leading anchor bases do.

        An index opening with anchor bases has them counted into the run, which pushes
        the run end into the index itself and pulls the index back towards the window's
        left edge. The left margin exists to absorb exactly that, so the index start
        stays inside the window and the whole index still fits.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + index + TAIL
        true_start = len(PREFIX) + 3

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.start).is_less_than_or_equal_to(true_start)
        assert_that(true_start + TGIDX_LEN).is_less_than_or_equal_to(
            window.start + len(window.sequence)
        )
        assert_that(window.sequence[true_start - window.start :][:TGIDX_LEN]).is_equal_to(index)
        assert_that(true_start - window.start).is_equal_to(expected_offset)

    def test_window_contains_whole_index_when_it_begins_one_base_after_the_run(self) -> None:
        """Test that the window still covers an index beginning a base after the run ends.

        A non-anchor base sitting between the run end and the index pushes the index one
        place further right than the run end predicts. The right margin exists for exactly
        this: the index lands flush against the window's far edge, with nothing to spare,
        so dropping the error slack from the right bound would clip its final base.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + INTERVENING_BASE + INDEX_NO_LEADING_ANCHOR + TAIL
        true_start = len(PREFIX) + 3 + len(INTERVENING_BASE)

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.start).is_less_than_or_equal_to(true_start)
        assert_that(true_start + TGIDX_LEN).is_less_than_or_equal_to(
            window.start + len(window.sequence)
        )
        assert_that(window.sequence[true_start - window.start :][:TGIDX_LEN]).is_equal_to(
            INDEX_NO_LEADING_ANCHOR
        )
        assert_that(true_start - window.start).is_equal_to(4)
        assert_that(true_start + TGIDX_LEN).is_equal_to(window.start + len(window.sequence))

    @pytest.mark.parametrize(
        "index",
        [INDEX_NO_LEADING_ANCHOR, INDEX_ONE_LEADING_ANCHOR, INDEX_TWO_LEADING_ANCHOR],
    )
    def test_read_coordinates_recovered_by_adding_window_start_to_window_offset(
        self, index: str
    ) -> None:
        """Test that a hit's read coordinate is the window start plus its window offset.

        The window is a plain slice with no other transformation applied, so a caller
        converts back to read coordinates by addition alone.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + index + TAIL
        true_start = len(PREFIX) + 3

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.start + window.sequence.index(index)).is_equal_to(true_start)

    def test_window_clamps_to_read_end_when_run_reaches_it(self) -> None:
        """Test that a read ending inside the anchor run yields a short, unmatchable window.

        There is no index left in the read to find, so the window is whatever remains
        and the caller is told not to bother matching it.
        """
        seq = PREFIX + ANCHOR_BASE * 5

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.start + len(window.sequence)).is_equal_to(len(seq))
        assert_that(window.is_matchable).is_false()

    def test_window_clamps_to_read_end_when_read_stops_just_after_the_run(self) -> None:
        """Test that a truncated read shortens the window's right side only.

        The left margin is already behind the run end and so survives truncation; only
        the right margin is cut back to the read end.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + "TATA"
        run_end = len(PREFIX) + 3

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.start + len(window.sequence)).is_equal_to(len(seq))
        assert_that(window.start).is_equal_to(run_end - WINDOW_LEAD)

    @pytest.mark.parametrize(
        "tail,expected_length,expected_matchable",
        [("TAT", 6, False), ("TATA", 7, True)],
    )
    def test_matchability_turns_over_at_the_floor(
        self, tail: str, expected_length: int, expected_matchable: bool
    ) -> None:
        """Test that the floor is the exact boundary between skipping and matching."""
        seq = PREFIX + ANCHOR_BASE * 3 + tail

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.sequence).is_length(expected_length)
        assert_that(window.floor).is_equal_to(WINDOW_FLOOR)
        assert_that(window.is_matchable).is_equal_to(expected_matchable)

    def test_floor_falls_back_to_the_seed_length_for_a_short_index(self) -> None:
        """Test that the floor never drops below the matcher's seed length.

        A four-base index at two errors could verify against a two-base window, but the
        matcher cannot even seed on one, so the seed length is the binding term.
        """
        seq = build_read(3)

        window = window_after_run(seq, len(PREFIX), tgidx_len=4, max_errors=2)

        assert_that(window.floor).is_equal_to(4)

    def test_window_start_clamps_at_zero_for_a_run_near_the_read_start(self) -> None:
        """Test that a run end inside the left margin clamps the start rather than wrapping.

        Subtracting the margin from a small run end goes negative, and a negative slice
        bound would silently take the window from the far end of the read.
        """
        seq = ANCHOR_BASE * 2 + INDEX_NO_LEADING_ANCHOR + TAIL

        window = window_after_run(seq, 0)

        assert_that(window.start).is_equal_to(0)
        assert_that(seq.startswith(window.sequence)).is_true()

    def test_default_leading_anchor_allowance_tracks_the_chemistry_constant(self) -> None:
        """Test that the default matches the bound the chemistry validates whitelists against.

        The whitelist loader rejects any entry opening with more leading anchor bases
        than this, so a locator defaulting to anything else would size its left margin
        against a bound nobody enforces.
        """
        seq = build_read(3)

        default_window = window_after_run(seq, len(PREFIX))
        explicit_window = window_after_run(
            seq, len(PREFIX), max_leading_anchor=TGIDX_MAX_LEADING_ANCHOR
        )

        assert_that(default_window).is_equal_to(explicit_window)


class TestMatcherContract:
    """Tests that the locator's floor lines up with the matcher it feeds."""

    @pytest.fixture
    def matcher(self) -> KmerMatcher:
        """Provide a target-index matcher built exactly as the assigner will build one.

        Returns:
            A KmerMatcher over the shipped target index whitelist, taking its seed
            length from its own default rather than from the test.
        """
        chemistry = ChemistryFactory.get_chemistry(CHEMISTRY)
        return KmerMatcher(
            whitelist=chemistry.tgidx_whitelist(),
            component=chemistry.tgidx_component(),
            chemistry=chemistry,
            max_errors=MAX_ERRORS,
        )

    def test_matcher_seed_length_matches_the_locator_default(self, matcher: KmerMatcher) -> None:
        """Test that the matcher's default seed length is the one the floor is built from.

        The floor only protects the matcher while the two agree, so this fails loudly
        if either default ever moves away from the other.
        """
        locator_default = signature(locate_tgidx_window).parameters["k"].default

        assert_that(matcher.k).is_equal_to(4)
        assert_that(locator_default).is_equal_to(matcher.k)

    def test_matcher_rejects_a_window_below_the_floor(self, matcher: KmerMatcher) -> None:
        """Test that a sub-floor window is exactly what the matcher refuses to accept.

        This is what ``is_matchable`` buys the caller: the window it flags is the window
        that would have raised.
        """
        seq = PREFIX + ANCHOR_BASE * 5

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.is_matchable).is_false()
        with pytest.raises(ValueError, match="Read segment too short for k-mer matching"):
            matcher.collect_candidates(window.sequence)

    def test_matcher_accepts_a_window_at_the_floor(self, matcher: KmerMatcher) -> None:
        """Test that a window exactly at the floor is accepted by the matcher.

        The floor is not merely safe but tight: one base shorter would raise, so the
        caller loses nothing by trusting it.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + "TATA"

        window = window_after_run(seq, len(PREFIX))

        assert_that(window.sequence).is_length(WINDOW_FLOOR)
        assert_that(window.is_matchable).is_true()
        assert_that(matcher.collect_candidates(window.sequence)).is_instance_of(list)

    def test_matcher_recovers_the_index_from_a_well_formed_window(
        self, matcher: KmerMatcher
    ) -> None:
        """Test that a normal window does not merely survive the matcher but yields the index.

        The floor tests prove a window is safe to hand over; this proves one is worth
        handing over. Candidates are ``(entry, start, end, edit_distance)`` with their
        coordinates relative to the window, so the read coordinate of a hit is the window
        start plus the candidate start.
        """
        seq = PREFIX + ANCHOR_BASE * 3 + INDEX_NO_LEADING_ANCHOR + TAIL
        true_start = len(PREFIX) + 3

        window = window_after_run(seq, len(PREFIX))
        candidates = matcher.collect_candidates(window.sequence)

        assert_that([candidate[0] for candidate in candidates]).contains(INDEX_NO_LEADING_ANCHOR)
        assert_that([window.start + candidate[1] for candidate in candidates]).contains(true_start)


class TestAnchorRunBridgesInterruptions:
    """Tests for reading an anchor tract that carries a base which is not the anchor.

    The construct's anchor is a real homopolymer tract, and in this library a tract
    frequently carries one base that is not the anchor base at high quality. Counting
    forward stops there, reporting a run that ends several bases early; every window cut
    from that end then sits too far left and hides the target index behind its right edge.
    A tract is told apart from sequence that has genuinely moved on by what follows the
    interruption, so the run resumes only when at least ``min_run`` anchor bases do.
    """

    def test_single_interruption_is_bridged_when_the_run_resumes(self) -> None:
        """Test that one non-anchor base inside a tract does not end the run."""
        seq = "GG" + "A" + "GGG" + "TATAGCCT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(6)
        assert_that(result.length).is_equal_to(6)

    def test_run_that_has_genuinely_ended_is_not_extended(self) -> None:
        """Test that a tract followed by ordinary sequence stops where it stops.

        This is the case the resumption requirement exists to protect: one anchor base
        appears in arbitrary sequence often, so bridging on its presence alone would walk
        the run, and the window cut from it, into the insert.
        """
        seq = "GGG" + "T" + "ACGTACGTACGT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(3)

    def test_interruption_is_not_bridged_when_too_few_anchor_bases_resume(self) -> None:
        """Test that a resumption shorter than min_run does not license a bridge."""
        seq = "GG" + "A" + "GG" + "TATAGCCT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(2)

    def test_at_most_one_interruption_is_bridged(self) -> None:
        """Test that a second interruption ends the run even when it too resumes.

        A tract with one blemish is still a tract. Two is a different piece of sequence,
        and bridging repeatedly would let the run cross into the index itself.
        """
        seq = "GG" + "A" + "GGG" + "A" + "GGG" + "TATAGCCT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(6)
        assert_that(ANCHOR_RUN_MAX_BRIDGED).is_equal_to(1)

    def test_bridging_is_safe_at_the_end_of_the_read(self) -> None:
        """Test that a run reaching the read's end is reported rather than read past it."""
        seq = "ACGT" + "GGG"

        result = locate_anchor_run(seq, 4, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(len(seq))

    def test_an_interruption_beyond_the_read_end_is_not_bridged(self) -> None:
        """Test that a tract ending one base short of the read does not read off the end."""
        seq = "GGG" + "A"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(3)

    @pytest.mark.parametrize("run_len", [3, 4, 5, 6, 7, 8])
    def test_an_uninterrupted_run_is_unchanged(self, run_len: int) -> None:
        """Test that the common case is untouched, so no window moves for it."""
        seq = build_read(run_len)
        run_start = len(PREFIX)

        result = locate_anchor_run(seq, run_start, ANCHOR_BASE, MIN_RUN)

        assert_that(result.end).is_equal_to(run_start + run_len)

    def test_bridging_finds_the_same_end_as_an_uninterrupted_tract_of_equal_length(self) -> None:
        """Test that a blemished tract and a clean one of the same extent agree.

        The point of bridging is that the index is at a fixed distance behind the tract,
        so the two reads must hand the window the same coordinate.
        """
        clean = "GGGGGG" + "TATAGCCT"
        blemished = "GGAGGG" + "TATAGCCT"

        clean_end = locate_anchor_run(clean, 0, ANCHOR_BASE, MIN_RUN).end
        blemished_end = locate_anchor_run(blemished, 0, ANCHOR_BASE, MIN_RUN).end

        assert_that(blemished_end).is_equal_to(clean_end)

    def test_helper_reports_the_run_end_directly(self) -> None:
        """Test the bridging scan on its own, since the insert locator calls it too."""
        seq = "GGAGGGTATAGCCT"

        result = run_end_bridging_interruptions(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result).is_equal_to(6)

    def test_bridging_can_be_disabled(self) -> None:
        """Test that the bridge count is a parameter, so the old scan remains expressible."""
        seq = "GGAGGGTATAGCCT"

        result = run_end_bridging_interruptions(seq, 0, ANCHOR_BASE, MIN_RUN, max_bridged=0)

        assert_that(result).is_equal_to(2)

    def test_a_shifted_run_start_also_bridges(self) -> None:
        """Test that a run found by the forward search is read with the same rule.

        The forward search and the on-target count report the same kind of thing, so a
        tract found one base late must not be measured by a different scan.
        """
        seq = "T" + "GGG" + "A" + "GGG" + "TATAGCCT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.start).is_equal_to(1)
        assert_that(result.end).is_equal_to(8)

    def test_forward_search_still_needs_min_run_before_any_interruption(self) -> None:
        """Test that the forward search is not itself made tolerant by this change.

        Finding where a run begins and reading how far it extends are separate questions.
        The search still demands min_run anchor bases at the shifted offset, so a tract
        that both starts late and is interrupted before min_run bases have accumulated is
        reported as absent rather than guessed at.
        """
        seq = "T" + "GG" + "A" + "GGG" + "TATAGCCT"

        result = locate_anchor_run(seq, 0, ANCHOR_BASE, MIN_RUN)

        assert_that(result.length).is_equal_to(0)
