"""
End-to-end tests for the barcode ambiguity contract.

The contract is that a read whose barcode window sits equally close to two or more whitelist
entries is not a read we know the barcode of, and must fail rather than be assigned. It is not
the property of any one class: each matcher decides privately that it cannot separate two
candidates, ``BarcodeMatchHistory`` has to record that as something other than "found nothing",
and ``HybridExtractor`` has to stop escalating on it. Tests split along class boundaries can
each pass while the contract as a whole does not hold, which is what happened, so the contract
is tested here as one thing.

The misassignment sweep at the bottom is the metric this work is judged by. It is the only test
in the suite that measures the thing the ambiguity contract exists to prevent -- a read being
given a *wrong* barcode rather than no barcode -- and a wrong cell barcode is otherwise
invisible: it is a valid whitelist entry, indistinguishable in every output from a correct one,
and reported as a successfully corrected match.

Every read here is assembled at the full 150 bases a run produces, cDNA tail included. That is
not decoration. On the 94bp read this file used to build, a whitelist entry planted where the
component cannot be has nowhere to sit, and the last barcode's search window runs off the end
of the read rather than being bounded by anything -- so a component called from sequence it has
no business in was invisible to every test in the file. The tail is a fixed sequence rather
than random filler precisely so that what a test plants in it is the only thing in it a matcher
can react to, which ``TestReadFixture`` below asserts rather than assumes.
"""

import random

import pytest
from assertpy import assert_that

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    BarcodeMatchHistory,
    MatchMethod,
)
from carmack.barcode.hybrid_extractor import HybridExtractor
from carmack.barcode.matchers.alignment_matcher import AlignmentMatcher
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (
    BC_CHUNK_LEN,
    PRIMER_A,
    PRIMER_C,
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.read_component import ReadComponentType

# Ceiling on wrong calls per sweep cell, as a fraction of the reads in it. Every read in a cell
# carries exactly one error in one component, so a wrong call means the pipeline reported a
# different whitelist entry from the one the read was built with. The ceilings are not aspirations
# -- they are set just above what the tree measures, so that a regression that starts producing
# wrong barcodes fails here rather than being discovered in a library.
WRONG_CALL_CEILING = 0.01

# Reads per cell in the always-run smoke tier. The full tier below uses the same n as the
# investigation that produced these numbers.
SMOKE_READS_PER_CELL = 60
SWEEP_READS_PER_CELL = 1200

# The length of an assembled read: the barcodes, primers, UMI, poly-G anchor and target index
# come to 94 bases, and the cDNA tail below carries it to what a run actually produces.
READ_LENGTH = 150

# The cDNA every assembled read ends in. It has to be inert in two separate ways, both of them
# asserted by ``TestReadFixture`` rather than taken on trust: it carries no ``GGG`` run, so it
# cannot be mistaken for the poly-G anchor, and no window in it is within the barcode budget of
# any whitelist entry, so a decoy planted in it is the only thing there a matcher can react to.
# Random filler would satisfy neither reliably -- a 96-entry 10bp whitelist matches random
# sequence at roughly 9e-05 per window, which over a tail this long is not a rare event.
DEFAULT_CDNA_TAIL = "TTTCCTCATGCAATTCAAAACCATGTCCGTAATGTAGGCGAAATAGTAAACCATTT"

# Bases inserted into a primer to displace everything 3' of it. An insertion is the only lesion
# that moves a component without damaging it, which is what makes it the test for whether a
# positional bound has been drawn too tightly.
PRIMER_INSERTION_BASES = "TCATGACT"

# PRIMER_A carrying one substitution at its midpoint. It is still 22 bases, so nothing 3' of it
# moves, but ``check_spacers`` compares the flanking region to the primer base for base, so it
# no longer counts as evidence. Reads that need a displaced component to stand on its own,
# with no corroboration available anywhere, are built with this in place of the real primer.
PRIMER_A_MISMATCHED = PRIMER_A[:11] + "A" + PRIMER_A[12:]

# The barcode triplet the positional tests are built from. Each damaged form carries one
# substitution away from its own entry, and is four or more edits from every other entry in its
# whitelist, so nothing but a deliberately planted sequence can outscore the component's own
# window. The decoys are valid entries of the component they are planted against, which is what
# makes them dangerous: called, they are indistinguishable in every output from a real call.
TRUE_BC3 = "TGACCGTACT"
TRUE_BC2 = "TTAGTTGGAC"
TRUE_BC1 = "TGTAGCAAGT"
DAMAGED_BC2 = "TTAGCTGGAC"
DAMAGED_BC1 = "TGTAACAAGT"
DECOY_BC2 = "AATAGCGTGG"
DECOY_BC1 = "GTCAACTAAC"


def build_matchers(
    chemistry: ChemistryCarmackCustomSeq10, fast: bool
) -> dict[MatchMethod, dict[str, MatcherBase]]:
    """
    Build the matcher stack the barcode extractor would build for this chemistry.

    Args:
        chemistry: The chemistry supplying components, whitelists and error budget.
        fast: When True, omit the alignment matcher, as ``--fast`` does.

    Returns:
        Mapping of match method to per-component matcher, in escalation order.
    """
    fixed: dict[str, MatcherBase] = {}
    kmer: dict[str, MatcherBase] = {}
    alignment: dict[str, MatcherBase] = {}

    for component in chemistry.read_structure:
        if component.type is not ReadComponentType.BARCODE:
            continue
        shared = {
            "component": component,
            "whitelist": chemistry.barcode_whitelists[component.name],
            "chemistry": chemistry,
        }
        fixed[component.name] = FixedPositionMatcher(**shared)
        kmer[component.name] = KmerMatcher(**shared, k=4, max_errors=chemistry.max_errors.barcode)
        if not fast:
            alignment[component.name] = AlignmentMatcher(
                **shared, max_errors=chemistry.max_errors.barcode
            )

    matchers: dict[MatchMethod, dict[str, MatcherBase]] = {
        MatchMethod.EXACTMATCH: fixed,
        MatchMethod.KMERMATCH: kmer,
    }
    if not fast:
        matchers[MatchMethod.ALIGNMATCH] = alignment
    return matchers


def build_read(
    bc3: str, bc2: str, bc1: str, umi: str = "ACGTACGT", tail: str = DEFAULT_CDNA_TAIL
) -> str:
    """
    Assemble a ``carmack_custom_seq_1_0`` read from its components.

    Args:
        bc3: BC3 sequence as it appears in the read, errors included.
        bc2: BC2 sequence as it appears in the read.
        bc1: BC1 sequence as it appears in the read.
        umi: UMI sequence to place after BC1.
        tail: The cDNA the read ends in, which is what carries it to full length. It defaults
            to a sequence held inert by ``TestReadFixture``; pass a shorter one only to build a
            read that is deliberately truncated.

    Returns:
        The assembled read: barcodes, primers, UMI, poly-G anchor, target index and cDNA.
    """
    return bc3 + PRIMER_C + bc2 + PRIMER_A + bc1 + umi + "GGGG" + "TATAGCCT" + tail


def plant(read: str, at: int, sequence: str) -> str:
    """
    Overwrite part of a read in place, leaving its length unchanged.

    Substituting rather than inserting is the point: the read keeps its layout, so whatever is
    planted is the only difference between this read and the one it was built from, and no
    component downstream of the planted sequence moves.

    Args:
        read: The assembled read to plant into.
        at: Index in ``read`` the sequence is written at.
        sequence: The bases to write, which may be a whole whitelist entry or a single base.

    Returns:
        The read with ``sequence`` written over the bases at ``at``.
    """
    return read[:at] + sequence + read[at + len(sequence) :]


def insert_into_primer(read: str, primer: str, bases: str) -> str:
    """
    Insert bases into the middle of a primer, displacing everything 3' of it.

    The insertion is placed inside the primer rather than at its edge so that the primer is
    genuinely broken as a spacer: ``check_spacers`` compares the whole 22 bases adjacent to a
    span against the primer, and a run of bases inserted anywhere within it fails that
    comparison. The read is trimmed back to its original length from the 3' end, the way a
    fixed-length run reports an upstream insertion.

    Args:
        read: The assembled read to lengthen.
        primer: The primer sequence to insert into, which must occur in ``read``.
        bases: The bases to insert.

    Returns:
        The read with the insertion applied, at its original length.
    """
    at = read.index(primer) + len(primer) // 2
    return (read[:at] + bases + read[at:])[: len(read)]


@pytest.fixture
def chemistry() -> ChemistryCarmackCustomSeq10:
    """
    Provide the custom_seq chemistry, whose barcode budget of one is where ambiguity bites.

    Returns:
        A freshly constructed ChemistryCarmackCustomSeq10.
    """
    return ChemistryCarmackCustomSeq10()


class TestReadFixture:
    """Tests that the assembled read is the thing the positional tests below assume it is."""

    def test_build_read_assembles_a_full_length_read(self) -> None:
        """Test that an assembled read is as long as one a run produces.

        Read length is not a detail here, it is what makes the rest of the file able to see
        anything. A component's search window is bounded partly by the read, so on a read that
        stops at the target index the last barcode's window degenerates to "everything left",
        and there is no cDNA for a planted entry to be found in. Both of those hid the defect
        this file exists to catch, and both are properties of the fixture rather than of the
        code under test, so they are asserted here.
        """
        read = build_read("TGACCGTACT", "TTAGTTGGAC", "TGTAGCAAGT")

        assert_that(read).is_length(READ_LENGTH)
        assert_that(DEFAULT_CDNA_TAIL).is_length(56)

    def test_default_tail_carries_no_polyg_run(self) -> None:
        """Test that the cDNA tail cannot be mistaken for the poly-G anchor.

        The UMI is cut at a ``GGG`` run rather than by any length check, so a tail carrying one
        would let a read whose barcodes were called from the wrong place still produce a
        plausible UMI -- exactly the compounding failure the tail is here to expose, arriving
        from the fixture instead of from the code.
        """
        assert_that(DEFAULT_CDNA_TAIL).does_not_contain("GGG")

    def test_default_tail_holds_no_window_near_any_whitelist_entry(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that nothing in the cDNA tail is within the barcode budget of a whitelist entry.

        Without this the decoy tests below would pass or fail for a reason unrelated to what
        they plant. Each of them asserts that a particular entry written into the tail is not
        called; if the tail already held a chance lookalike of some *other* entry, a matcher
        could call that instead and the assertion would still hold, or the lookalike could tie
        with the planted decoy and change the verdict outright. The tail has to contribute
        nothing for the plant to be the only variable.

        Every window a matcher could verify is checked, not just the ten-base ones: an indel
        inside the budget makes a nine- or eleven-base window a candidate span too. The
        distance is measured with the same helper and the same arguments the matchers use, so
        that this bound is the bound they will apply and not a stricter or looser cousin of it.
        """
        budget = chemistry.max_errors.barcode
        lengths = range(BC_CHUNK_LEN - budget, BC_CHUNK_LEN + budget + 1)

        nearest = min(
            (edit_distance(DEFAULT_CDNA_TAIL[start : start + length], entry, "N", True), entry)
            for name in ("BC1", "BC2", "BC3")
            for entry in chemistry.barcode_whitelists[name]
            for length in lengths
            for start in range(len(DEFAULT_CDNA_TAIL) - length + 1)
        )

        assert_that(nearest[0]).described_as(
            f"closest whitelist entry to any window of the default tail is {nearest[1]} at "
            f"edit distance {nearest[0]}, against a barcode budget of {budget}"
        ).is_greater_than(budget)


class TestAmbiguityIsADistinctState:
    """Tests that ambiguity is recorded as something other than "found nothing"."""

    def test_a_fresh_history_is_neither_successful_nor_ambiguous(self) -> None:
        """Test that a history starts in neither terminal state."""
        history = BarcodeMatchHistory(bc_name="BC1")

        assert_that(history.success).is_false()
        assert_that(history.is_ambiguous).is_false()
        assert_that(history.is_terminal).is_false()
        assert_that(history.ambiguous_at).is_none()

    def test_a_matchless_attempt_is_not_ambiguous(self) -> None:
        """Test that finding nothing is not recorded as ambiguity.

        This is the distinction the whole contract rests on. A component with no candidate
        anywhere has not been resolved, but neither has anything been decided about it, so a
        later matcher is free to try.
        """
        history = BarcodeMatchHistory(bc_name="BC1")
        history.record_attempt(BarcodeMatchAttempt(method=MatchMethod.KMERMATCH))

        assert_that(history.is_ambiguous).is_false()
        assert_that(history.is_terminal).is_false()

    def test_ambiguity_latches_and_is_terminal(self) -> None:
        """Test that an ambiguity verdict is recorded, attributed and final."""
        history = BarcodeMatchHistory(bc_name="BC1")
        for candidate in ("AACCAACTTA", "CACCAACCTA"):
            history.record_attempt(
                BarcodeMatchAttempt(method=MatchMethod.KMERMATCH, candidate=candidate),
                ambiguous=True,
            )

        assert_that(history.success).is_false()
        assert_that(history.is_ambiguous).is_true()
        assert_that(history.is_terminal).is_true()
        assert_that(history.ambiguous_at).is_equal_to(MatchMethod.KMERMATCH)

    def test_ambiguity_keeps_the_method_that_declared_it(self) -> None:
        """Test that a later matcher does not overwrite which stage declared ambiguity."""
        history = BarcodeMatchHistory(bc_name="BC1")
        history.record_attempt(BarcodeMatchAttempt(method=MatchMethod.KMERMATCH), ambiguous=True)
        history.record_attempt(BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH), ambiguous=True)

        assert_that(history.ambiguous_at).is_equal_to(MatchMethod.KMERMATCH)

    def test_success_is_terminal_too(self) -> None:
        """Test that a called barcode also stops escalation."""
        history = BarcodeMatchHistory(bc_name="BC1")
        history.record_attempt(
            BarcodeMatchAttempt(method=MatchMethod.KMERMATCH, match="AACCAACTTA"),
            success=True,
        )

        assert_that(history.is_terminal).is_true()
        assert_that(history.is_ambiguous).is_false()

    def test_status_string_distinguishes_ambiguous_from_no_candidate(self) -> None:
        """Test that the annotation says which of the two failure kinds happened.

        Both used to render ``NOMATCH``, so nothing downstream, and no human reading an
        annotated read name, could tell a read the pipeline lost from a read it declined to
        guess at -- nor could a run report how often it did each.
        """
        nothing_found = BarcodeMatchHistory(bc_name="BC1")
        nothing_found.record_attempt(BarcodeMatchAttempt(method=MatchMethod.KMERMATCH))

        unresolvable = BarcodeMatchHistory(bc_name="BC1")
        for candidate in ("AACCAACTTA", "CACCAACCTA"):
            unresolvable.record_attempt(
                BarcodeMatchAttempt(method=MatchMethod.KMERMATCH, candidate=candidate),
                ambiguous=True,
            )

        assert_that(nothing_found.to_status_string()).is_equal_to("BC1:NOMATCH")
        assert_that(unresolvable.to_status_string()).is_equal_to("BC1:KMERMATCH-AMBIG")
        assert_that(unresolvable.to_status_string()).is_not_equal_to(
            nothing_found.to_status_string()
        )


class TestAmbiguityStopsEscalation:
    """Tests that a later matcher is not run on a component an earlier one could not resolve."""

    def test_alignment_matcher_is_not_invoked_after_an_ambiguity_verdict(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that the alignment stage never sees a component the k-mer stage gave up on.

        Escalation used to be skipped only on success, and an ambiguity verdict is stored as a
        failure, so the alignment matcher ran on precisely the reads the k-mer matcher had
        declared unresolvable -- and because it ranked candidates by a different criterion it
        separated them and assigned one.
        """
        matchers = build_matchers(chemistry, fast=False)
        seen: list[str] = []

        class RecordingAlignmentMatcher(AlignmentMatcher):
            """An alignment matcher that records every component it is asked about."""

            def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
                """Record the component name, then behave normally."""
                seen.append(self.component.name)
                return super().match(read, start_idx)

        for name, matcher in list(matchers[MatchMethod.ALIGNMATCH].items()):
            matchers[MatchMethod.ALIGNMATCH][name] = RecordingAlignmentMatcher(
                whitelist=chemistry.barcode_whitelists[name],
                component=matcher.component,
                chemistry=chemistry,
                max_errors=chemistry.max_errors.barcode,
            )

        # BC1 window one substitution from both CACCAACCTA and AACCAACTTA.
        read = build_read("TGACCGTACT", "TTAGTTGGAC", "CACCAACTTA")
        extractor = HybridExtractor(chemistry=chemistry, matchers=matchers)
        result = extractor.process_read("read", read, "I" * len(read))

        bc1 = next(bc for bc in result.bc_results if bc.bc_name == "BC1")
        assert_that(bc1.is_ambiguous).is_true()
        assert_that(seen).does_not_contain("BC1")

    def test_a_component_with_no_candidate_still_escalates(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that "found nothing" does not stop escalation.

        Making ambiguity terminal must not quietly make failure terminal: the alignment stage
        exists to rescue reads the earlier stages could not place at all.
        """
        matchers = build_matchers(chemistry, fast=False)
        seen: list[str] = []

        class RecordingAlignmentMatcher(AlignmentMatcher):
            """An alignment matcher that records every component it is asked about."""

            def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
                """Record the component name, then behave normally."""
                seen.append(self.component.name)
                return super().match(read, start_idx)

        for name, matcher in list(matchers[MatchMethod.ALIGNMATCH].items()):
            matchers[MatchMethod.ALIGNMATCH][name] = RecordingAlignmentMatcher(
                whitelist=chemistry.barcode_whitelists[name],
                component=matcher.component,
                chemistry=chemistry,
                max_errors=chemistry.max_errors.barcode,
            )

        # A BC1 region of filler that seeds no whitelist k-mer at all.
        read = build_read("TGACCGTACT", "TTAGTTGGAC", "TTTTTTTTTT")
        extractor = HybridExtractor(chemistry=chemistry, matchers=matchers)
        extractor.process_read("read", read, "I" * len(read))

        assert_that(seen).contains("BC1")


class TestAmbiguityRegressions:
    """Reproductions of specific reads that used to be given a barcode they had not earned."""

    def test_equidistant_bc1_window_fails_in_both_modes(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that a window equidistant from two whitelist entries fails, fast mode or not.

        The narrowest case the investigation found, and it came out of a random sweep rather
        than being constructed. The read's BC1 window ``CACCAACTTA`` is one substitution from
        ``CACCAACCTA`` (the entry it was built from, mismatch interior) and one substitution
        from ``AACCAACTTA`` (mismatch terminal). The k-mer matcher flagged it ambiguous and the
        alignment matcher then assigned the wrong one, because a terminal mismatch is clipped by
        local alignment and outscores an interior one. Under ``--fast`` the same read reported
        no match, so the two modes were not the same analysis.
        """
        read = build_read("TGACCGTACT", "TTAGTTGGAC", "CACCAACTTA")
        verdicts = {}

        for fast in (False, True):
            extractor = HybridExtractor(
                chemistry=chemistry, matchers=build_matchers(chemistry, fast=fast)
            )
            result = extractor.process_read("read", read, "I" * len(read))
            bc1 = next(bc for bc in result.bc_results if bc.bc_name == "BC1")

            assert_that(result.success).described_as(f"fast={fast}").is_false()
            assert_that(result.full_barcode).described_as(f"fast={fast}").is_none()
            for attempt in bc1.attempts:
                assert_that(attempt.match).described_as(f"fast={fast}").is_none()
            verdicts[fast] = bc1.is_ambiguous

        assert_that(verdicts[False]).described_as("full mode verdict").is_true()
        assert_that(verdicts[False]).described_as("both modes agree").is_equal_to(verdicts[True])

    def test_terminal_substitution_in_bc3_resolves_to_its_own_entry(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that a BC3 error at its last base no longer pulls a barcode out of the BC2 region.

        Four defects compounded on this read. The reported BC3 span truncated to 9bp, which
        moved the component's 3' boundary so the spacer check looked a base early and the true
        candidate lost its only evidence; it then tied with ``TAGTTGGACT`` -- itself a BC3
        whitelist entry -- verifying at (33, 43), inside the BC2 region; neither validated, so
        the k-mer matcher failed; and the alignment matcher, searching the whole read and
        ranking by score, preferred the spurious entry and assigned it with no spacer check at
        all. The read was reported as a successful corrected match.
        """
        true_bc3 = "ACTAGCTCTC"
        read = build_read(true_bc3[:-1] + "T", "TTAGTTGGAC", "TGTAGCAAGT")

        for fast in (False, True):
            extractor = HybridExtractor(
                chemistry=chemistry, matchers=build_matchers(chemistry, fast=fast)
            )
            result = extractor.process_read("read", read, "I" * len(read))
            bc3 = next(bc for bc in result.bc_results if bc.bc_name == "BC3")

            assert_that(bc3.success).described_as(f"fast={fast}").is_true()
            assert_that(bc3.attempts[-1].match).described_as(f"fast={fast}").is_equal_to(true_bc3)
            assert_that(bc3.attempts[-1].read_idx).described_as(f"fast={fast}").is_equal_to(
                (0, 10)
            )

    def test_a_barcode_is_never_called_from_another_components_region(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that a decoy entry planted in a neighbour's window is not called.

        ``TAGTTGGACT`` is a BC3 whitelist entry and a one-base rotation of the BC2 entry
        ``TTAGTTGGAC``, so any library using that BC2 carries a BC3 decoy in its BC2 region on
        every single read. On a real 1M-read library this produced a spurious barcode that was
        the second most abundant in the run.
        """
        read = build_read("TTTTTTTTTT", "TTAGTTGGAC", "TGTAGCAAGT")

        extractor = HybridExtractor(
            chemistry=chemistry, matchers=build_matchers(chemistry, fast=False)
        )
        result = extractor.process_read("read", read, "I" * len(read))
        bc3 = next(bc for bc in result.bc_results if bc.bc_name == "BC3")

        for attempt in bc3.attempts:
            assert_that(attempt.match).is_not_equal_to("TAGTTGGACT")
        assert_that(bc3.success).is_false()


def extract(
    chemistry: ChemistryCarmackCustomSeq10, read: str, fast: bool, bc_name: str
) -> BarcodeMatchHistory:
    """
    Run one read through the whole matcher stack and return one component's history.

    Args:
        chemistry: The chemistry to extract with.
        read: The assembled read.
        fast: Whether to omit the alignment matcher, as ``--fast`` does.
        bc_name: The component whose history is wanted.

    Returns:
        The match history recorded for ``bc_name``.
    """
    extractor = HybridExtractor(chemistry=chemistry, matchers=build_matchers(chemistry, fast=fast))
    result = extractor.process_read("read", read, "I" * len(read))
    return next(bc for bc in result.bc_results if bc.bc_name == bc_name)


def nominal_span(
    chemistry: ChemistryCarmackCustomSeq10, bc_name: str, shift: int = 0
) -> tuple[int, int]:
    """
    Return where the read structure says a component sits, optionally displaced.

    Args:
        chemistry: The chemistry whose read structure predicts the position.
        bc_name: The component to locate.
        shift: Bases the component has been pushed 3' by an upstream insertion.

    Returns:
        The component's half-open span in a read built to that layout.
    """
    start = chemistry.read_structure.get_component_by_name(bc_name).start + shift
    return (start, start + BC_CHUNK_LEN)


class TestPlantedDecoysAreNeverCalled:
    """Tests that a valid whitelist entry lying where its component cannot be is not called.

    This is the failure mode that produces a wrong cell barcode rather than a lost read, and
    it is the reason the read structure has to bound the search. The construction is the same
    in both tests: damage the component's own window by one substitution so it can no longer
    win on edit distance, then write a different valid entry of the *same* component somewhere
    the layout cannot put it. The decoy then scores zero, the true window scores one, zero wins
    outright, and no tie is formed -- so neither the tie-break nor the spacer check ever runs.
    Only a bound on where the component may be, and a rule that a lone candidate away from its
    predicted position has to show something for itself, can refuse it.
    """

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("offset", [14, 20, 30, 49, 60, 73, 76])
    def test_planted_bc1_entry_in_the_cdna_is_never_called(
        self, chemistry: ChemistryCarmackCustomSeq10, offset: int, fast: bool
    ) -> None:
        """Test that a BC1 entry planted downstream of BC1 is not called, at any offset.

        BC1 is the last barcode in the structure, so nothing follows it to bound its search and
        it was searched across the UMI, the poly-G anchor, the target index and the whole cDNA.
        The offsets walk a decoy out through every one of those regions in turn, ending with one
        that reaches the last base of the read: the bound has to hold along the entire span, not
        merely somewhere past the insert boundary.
        """
        read = plant(
            build_read(TRUE_BC3, TRUE_BC2, DAMAGED_BC1),
            nominal_span(chemistry, "BC1")[0] + offset,
            DECOY_BC1,
        )
        assert_that(read).is_length(READ_LENGTH)

        bc1 = extract(chemistry, read, fast, "BC1")

        for attempt in bc1.attempts:
            assert_that(attempt.match).described_as(
                f"decoy at +{offset}, fast={fast}, {attempt.method} span {attempt.read_idx}"
            ).is_not_equal_to(DECOY_BC1)
        assert_that(bc1.attempts[-1].match).described_as(
            f"decoy at +{offset}, fast={fast}"
        ).is_equal_to(TRUE_BC1)
        assert_that(bc1.attempts[-1].read_idx).described_as(
            f"decoy at +{offset}, fast={fast}"
        ).is_equal_to(nominal_span(chemistry, "BC1"))

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("offset", [-20, -12, 14, 18, 22])
    def test_planted_bc2_entry_inside_the_old_window_is_never_called(
        self, chemistry: ChemistryCarmackCustomSeq10, offset: int, fast: bool
    ) -> None:
        """Test that a BC2 entry planted either side of BC2 is not called.

        BC2 has a barcode on both sides of it, so the gap between its neighbours bounds it from
        both directions, and every offset here sits inside that gap. It fails identically to
        BC1 all the same, which is what says the missing right-hand neighbour was never the
        root cause: the distance to the next barcode is simply not a measure of how far this
        component can have moved. What bounds a component is the error budget of everything
        upstream of it, and that is a much smaller number than the primer separating it from
        its neighbour.
        """
        read = plant(
            build_read(TRUE_BC3, DAMAGED_BC2, TRUE_BC1),
            nominal_span(chemistry, "BC2")[0] + offset,
            DECOY_BC2,
        )
        assert_that(read).is_length(READ_LENGTH)

        bc2 = extract(chemistry, read, fast, "BC2")

        for attempt in bc2.attempts:
            assert_that(attempt.match).described_as(
                f"decoy at {offset:+d}, fast={fast}, {attempt.method} span {attempt.read_idx}"
            ).is_not_equal_to(DECOY_BC2)
        assert_that(bc2.attempts[-1].match).described_as(
            f"decoy at {offset:+d}, fast={fast}"
        ).is_equal_to(TRUE_BC2)
        assert_that(bc2.attempts[-1].read_idx).described_as(
            f"decoy at {offset:+d}, fast={fast}"
        ).is_equal_to(nominal_span(chemistry, "BC2"))


class TestUpstreamDisplacement:
    """Tests on how far a component may be pushed 3' by an upstream insertion and still be called.

    These are the other half of the contract, and they are the half that keeps it honest. A
    bound tight enough to refuse a planted decoy is also tight enough to refuse a real read
    whose barcodes have genuinely moved, and reads with an indel in a primer are a real and
    ordinary population. What decides the boundary is the cumulative error budget of everything
    upstream of the component -- one base per barcode and two per primer for this chemistry --
    so BC1, with two primers and two barcodes ahead of it, may travel six bases on its own
    while BC2, with one of each, may travel three.

    Where the insertion goes is not a free choice, and it is the whole subtlety of these tests.
    An insertion only displaces what is 3' of it, and it only removes spacer evidence from the
    span it is adjacent to. So an insertion in PRIMER_C displaces BC2 and BC1 both, but leaves
    PRIMER_A intact and hard against BC1, which then has corroboration whatever it does; and an
    insertion in PRIMER_A removes BC1's corroboration but leaves BC2 exactly where it started.
    Each test therefore puts the insertion in the primer that actually exercises the component
    it is about, and says so.
    """

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("insertion", [1, 2, 3, 4, 5, 6])
    def test_upstream_insertion_still_calls_bc1_at_its_displaced_span(
        self, chemistry: ChemistryCarmackCustomSeq10, insertion: int, fast: bool
    ) -> None:
        """Test that BC1 is called at its displaced span with no spacer evidence at all.

        The insertion goes into PRIMER_A, the primer immediately upstream of BC1, because that
        is the only placement that leaves BC1 with nothing to lean on. Its downstream neighbour
        is the UMI, which carries no sequence and so can never be a spacer, and an insertion
        anywhere in PRIMER_A means the 22 bases in front of the span no longer equal the primer.
        BC1 is then a lone candidate, away from where the structure predicts it, with no
        adjacent spacer -- and it must still be called, because the read is a perfectly good
        read that happens to have an indel in a primer. Putting the insertion in PRIMER_C
        instead would prove nothing: PRIMER_A would move along with BC1, stay hard against it,
        and supply the corroboration the test is trying to withhold.
        """
        read = insert_into_primer(
            build_read(TRUE_BC3, TRUE_BC2, TRUE_BC1), PRIMER_A, PRIMER_INSERTION_BASES[:insertion]
        )
        assert_that(read).is_length(READ_LENGTH)

        bc1 = extract(chemistry, read, fast, "BC1")
        attempt = bc1.attempts[-1]
        described = f"insertion of {insertion} into PRIMER_A, fast={fast}"

        assert_that(bc1.success).described_as(described).is_true()
        assert_that(attempt.match).described_as(described).is_equal_to(TRUE_BC1)
        assert_that(attempt.read_idx).described_as(described).is_equal_to(
            nominal_span(chemistry, "BC1", insertion)
        )
        assert_that(attempt.spacer_upstream).described_as(described).is_none()
        assert_that(attempt.spacer_downstream).described_as(described).is_none()

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("insertion", [1, 2, 3])
    def test_upstream_insertion_still_calls_bc2_at_its_displaced_span(
        self, chemistry: ChemistryCarmackCustomSeq10, insertion: int, fast: bool
    ) -> None:
        """Test that BC2 is called at its displaced span with no spacer evidence at all.

        The insertion goes into PRIMER_C, the only primer upstream of BC2 and so the only one
        that can displace it. Three bases is where this stops being a test of displacement
        alone: BC2's cumulative budget is three, and beyond that a call needs corroboration
        rather than just room, which the next two tests take up. PRIMER_C is broken by the
        insertion so BC2's upstream evidence is gone, and its downstream PRIMER_A is not
        consulted at all within this range, because a candidate at or inside its budget is
        corroborated by its position and nothing further is asked of it. Both spacer fields
        staying empty is the observable form of that: evidence recorded here would mean the
        matcher had gone looking for it on a read where the answer was already settled, which
        would move every annotated read name and every spacer statistic in the pipeline.
        """
        read = insert_into_primer(
            build_read(TRUE_BC3, TRUE_BC2, TRUE_BC1), PRIMER_C, PRIMER_INSERTION_BASES[:insertion]
        )
        assert_that(read).is_length(READ_LENGTH)

        bc2 = extract(chemistry, read, fast, "BC2")
        attempt = bc2.attempts[-1]
        described = f"insertion of {insertion} into PRIMER_C, fast={fast}"

        assert_that(bc2.success).described_as(described).is_true()
        assert_that(attempt.match).described_as(described).is_equal_to(TRUE_BC2)
        assert_that(attempt.read_idx).described_as(described).is_equal_to(
            nominal_span(chemistry, "BC2", insertion)
        )
        assert_that(attempt.spacer_upstream).described_as(described).is_none()
        assert_that(attempt.spacer_downstream).described_as(described).is_none()

    @pytest.mark.parametrize("fast", [False, True])
    def test_a_displaced_bc2_past_its_budget_is_called_on_spacer_evidence(
        self, chemistry: ChemistryCarmackCustomSeq10, fast: bool
    ) -> None:
        """Test that a component past its budget is still called when a spacer corroborates it.

        Four bases is one past what BC2's cumulative budget predicts, so position alone no
        longer speaks for the candidate and it has to show something. On this read it can: an
        insertion in PRIMER_C moves BC2 and PRIMER_A together, so the primer is still exactly
        the 22 bases immediately 3' of the span and says so. The rule is not a hard ceiling on
        displacement, it is a demand for evidence proportional to how surprising the position
        is, and this is the case that distinguishes the two -- the very same displacement is
        refused by the next test, on a read where that evidence has been taken away.
        """
        read = insert_into_primer(
            build_read(TRUE_BC3, TRUE_BC2, TRUE_BC1), PRIMER_C, PRIMER_INSERTION_BASES[:4]
        )
        assert_that(read).is_length(READ_LENGTH)

        bc2 = extract(chemistry, read, fast, "BC2")
        attempt = bc2.attempts[-1]
        described = f"insertion of 4 into PRIMER_C, fast={fast}"

        assert_that(bc2.success).described_as(described).is_true()
        assert_that(attempt.match).described_as(described).is_equal_to(TRUE_BC2)
        assert_that(attempt.read_idx).described_as(described).is_equal_to(
            nominal_span(chemistry, "BC2", 4)
        )
        assert_that(attempt.spacer_downstream).described_as(described).is_equal_to("PRIMER_A")

    @pytest.mark.parametrize("fast", [False, True])
    @pytest.mark.parametrize("insertion", [4, 5, 6])
    def test_displacement_beyond_the_cumulative_budget_is_refused(
        self, chemistry: ChemistryCarmackCustomSeq10, insertion: int, fast: bool
    ) -> None:
        """Test that an uncorroborated component past its budget is refused rather than called.

        This is the design working, not a regression, and it is the price of the decoy tests
        above: BC2 found more than the three bases from its predicted start that the structure
        upstream of it can account for is making a claim the structure does not support, and if
        nothing else supports it either then a refused read is the honest answer. A wrong cell
        barcode is invisible downstream, a dropped read is not.

        The read is built with PRIMER_A substituted so the displaced BC2 has no evidence
        anywhere -- PRIMER_C is broken by the insertion, PRIMER_A no longer matches base for
        base -- which is what isolates the budget from the corroboration. BC1 on the same read
        is displaced by exactly as much and has exactly as little evidence, and is called
        throughout, because six is what its own cumulative budget allows: the refusal is BC2's
        budget being spent, not a bound drawn tight across the whole structure.
        """
        read = insert_into_primer(
            plant(
                build_read(TRUE_BC3, TRUE_BC2, TRUE_BC1),
                chemistry.read_structure.get_component_by_name("PRIMER_A").start,
                PRIMER_A_MISMATCHED,
            ),
            PRIMER_C,
            PRIMER_INSERTION_BASES[:insertion],
        )
        assert_that(read).is_length(READ_LENGTH)
        described = f"insertion of {insertion} into PRIMER_C, fast={fast}"

        bc2 = extract(chemistry, read, fast, "BC2")
        assert_that(bc2.success).described_as(described).is_false()
        for attempt in bc2.attempts:
            assert_that(attempt.match).described_as(
                f"{described}, {attempt.method} span {attempt.read_idx}"
            ).is_none()

        bc1 = extract(chemistry, read, fast, "BC1")
        assert_that(bc1.attempts[-1].match).described_as(described).is_equal_to(TRUE_BC1)
        assert_that(bc1.attempts[-1].read_idx).described_as(described).is_equal_to(
            nominal_span(chemistry, "BC1", insertion)
        )


class SweepOutcome:
    """Tally of one sweep cell: how many reads were called right, wrong, or not at all."""

    def __init__(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.ambiguous = 0
        self.nomatch = 0
        self.wrong_examples: list[tuple[str, str, str]] = []
        # What each read was called, in generation order: the assembled barcode, or None for a
        # read that was not called. Tallies alone cannot say the two modes agree -- one mode
        # calling read A right and read B wrong while the other does the reverse leaves every
        # count identical -- and it is agreement on the individual read that decides which
        # cell a molecule lands in.
        self.calls: list[str | None] = []

    @property
    def total(self) -> int:
        """Return the number of reads tallied."""
        return self.correct + self.wrong + self.ambiguous + self.nomatch


def run_sweep(
    chemistry: ChemistryCarmackCustomSeq10,
    component_name: str,
    error: str,
    fast: bool,
    reads: int,
    seed: int = 1234,
) -> SweepOutcome:
    """
    Introduce one error into one component of random reads and tally what comes back.

    Args:
        chemistry: The chemistry to extract with.
        component_name: The component the error is introduced into.
        error: One of ``sub``, ``del`` or ``ins``.
        fast: Whether to omit the alignment matcher.
        reads: Number of reads to generate.
        seed: Seed for the read generator, so a failure is reproducible.

    Returns:
        The tally for this cell.
    """
    whitelists = chemistry.barcode_whitelists
    entries = {name: sorted(whitelists[name]) for name in ("BC1", "BC2", "BC3")}
    extractor = HybridExtractor(chemistry=chemistry, matchers=build_matchers(chemistry, fast=fast))
    rng = random.Random(seed)
    outcome = SweepOutcome()

    for _ in range(reads):
        truth = {name: rng.choice(entries[name]) for name in ("BC1", "BC2", "BC3")}
        target = truth[component_name]
        position = rng.randrange(len(target))
        if error == "sub":
            replacement = rng.choice([b for b in "ACGT" if b != target[position]])
            mutated = target[:position] + replacement + target[position + 1 :]
        elif error == "del":
            mutated = target[:position] + target[position + 1 :]
        else:
            mutated = target[:position] + rng.choice("ACGT") + target[position:]

        observed = dict(truth)
        observed[component_name] = mutated
        umi = "".join(rng.choice("ACGT") for _ in range(8))
        read = build_read(observed["BC3"], observed["BC2"], observed["BC1"], umi)

        result = extractor.process_read("read", read, "I" * len(read))
        outcome.calls.append(result.full_barcode if result.success else None)
        if result.success:
            called = {bc.bc_name: bc.attempts[-1].match for bc in result.bc_results}
            if called == truth:
                outcome.correct += 1
            else:
                outcome.wrong += 1
                if len(outcome.wrong_examples) < 3:
                    outcome.wrong_examples.append(
                        (component_name, target, str(called[component_name]))
                    )
        elif any(bc.is_ambiguous for bc in result.bc_results):
            outcome.ambiguous += 1
        else:
            outcome.nomatch += 1

    return outcome


class TestMisassignmentSweep:
    """The metric: how often a read carrying one error is given the wrong barcode."""

    @pytest.mark.parametrize("component_name", ["BC3", "BC2", "BC1"])
    def test_a_single_substitution_is_never_misassigned(
        self, chemistry: ChemistryCarmackCustomSeq10, component_name: str
    ) -> None:
        """Test that one substitution in one component does not produce a wrong barcode.

        Substitutions are the case the alignment stage used to handle worst: it rescued almost
        nothing and over half of what it did rescue was wrong, so on this error class it was a
        net converter of honest failures into confident wrong answers.

        Residual wrong calls here would come from the whitelists rather than the matchers --
        two entries within the error budget of each other admit a window that is one edit from
        both -- so the ceiling is a fraction rather than zero, and the chemistry warns about
        those pairs at construction.
        """
        outcome = run_sweep(
            chemistry, component_name, "sub", fast=False, reads=SMOKE_READS_PER_CELL
        )

        assert_that(outcome.total).is_equal_to(SMOKE_READS_PER_CELL)
        assert_that(outcome.wrong / outcome.total).described_as(
            f"wrong calls, examples: {outcome.wrong_examples}"
        ).is_less_than_or_equal_to(WRONG_CALL_CEILING)

    @pytest.mark.parametrize("component_name", ["BC3", "BC2", "BC1"])
    def test_both_modes_agree_on_a_single_substitution(
        self, chemistry: ChemistryCarmackCustomSeq10, component_name: str
    ) -> None:
        """Test that omitting the alignment stage does not change the verdict on a read.

        ``--fast`` and the full path have to be the same analysis. They were not: the fast path
        correctly reported no match on reads the full path assigned a barcode to, which meant
        the mode a run happened to use changed which cells its reads landed in.

        Agreement is asserted read by read and not only in aggregate, because the aggregate
        form is satisfiable by two modes that disagree on individual reads in offsetting
        directions, and it is the individual read that ends up in a cell. It is agreement on
        what was *called* rather than on the shape of a failure: a read the full path finds
        two equally close candidates for is recorded as unresolvable, and the fast path, which
        never runs the stage that says so, records the same read as unmatched. Both decline to
        guess, which is the property that matters; only the annotation differs.
        """
        full = run_sweep(chemistry, component_name, "sub", fast=False, reads=SMOKE_READS_PER_CELL)
        fast = run_sweep(chemistry, component_name, "sub", fast=True, reads=SMOKE_READS_PER_CELL)

        assert_that(full.wrong).is_equal_to(fast.wrong)
        assert_that(full.correct).is_equal_to(fast.correct)
        disagreements = [
            (index, full_call, fast_call)
            for index, (full_call, fast_call) in enumerate(zip(full.calls, fast.calls))
            if full_call != fast_call
        ]
        assert_that(disagreements).described_as(
            f"{component_name}: reads the two modes called differently, as "
            f"(read, full, fast): {disagreements[:5]}"
        ).is_empty()

    @pytest.mark.only_run_with_direct_target
    @pytest.mark.parametrize("component_name", ["BC3", "BC2", "BC1"])
    @pytest.mark.parametrize("error", ["sub", "del", "ins"])
    @pytest.mark.parametrize("fast", [False, True])
    def test_misassignment_sweep(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        component_name: str,
        error: str,
        fast: bool,
    ) -> None:
        """Test the full sweep: every component, every error class, both modes.

        This is the sweep the investigation ran, at the same size. It is slow -- around half an
        hour for the eighteen cells, which is most of the direct-target tier's runtime -- and the
        size is why: the ceiling is a rate, so it only means anything at an n where the measured
        rates (up to 0.5% on a cell) sit comfortably under it. Lowering the reads per cell to
        shorten the run would put the worst cell level with its own ceiling and make this flaky,
        which is worse than slow for a test whose whole job is to be believed. Run it with::

            python -m pytest tests/test_barcode_ambiguity.py -k misassignment_sweep
        """
        outcome = run_sweep(
            chemistry, component_name, error, fast=fast, reads=SWEEP_READS_PER_CELL
        )

        assert_that(outcome.total).is_equal_to(SWEEP_READS_PER_CELL)
        assert_that(outcome.wrong / outcome.total).described_as(
            f"{component_name} {error} fast={fast}: {outcome.wrong} wrong of "
            f"{outcome.total}, examples: {outcome.wrong_examples}"
        ).is_less_than_or_equal_to(WRONG_CALL_CEILING)
