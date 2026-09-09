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
"""

import random

import pytest
from assertpy import assert_that

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


def build_read(bc3: str, bc2: str, bc1: str, umi: str = "ACGTACGT") -> str:
    """
    Assemble a ``carmack_custom_seq_1_0`` read from its components.

    Args:
        bc3: BC3 sequence as it appears in the read, errors included.
        bc2: BC2 sequence as it appears in the read.
        bc1: BC1 sequence as it appears in the read.
        umi: UMI sequence to place after BC1.

    Returns:
        The assembled read, ending in the poly-G run and target index.
    """
    return bc3 + PRIMER_C + bc2 + PRIMER_A + bc1 + umi + "GGGG" + "TATAGCCT"


@pytest.fixture
def chemistry() -> ChemistryCarmackCustomSeq10:
    """
    Provide the custom_seq chemistry, whose barcode budget of one is where ambiguity bites.

    Returns:
        A freshly constructed ChemistryCarmackCustomSeq10.
    """
    return ChemistryCarmackCustomSeq10()


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


class SweepOutcome:
    """Tally of one sweep cell: how many reads were called right, wrong, or not at all."""

    def __init__(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.ambiguous = 0
        self.nomatch = 0
        self.wrong_examples: list[tuple[str, str, str]] = []

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
        """
        full = run_sweep(chemistry, component_name, "sub", fast=False, reads=SMOKE_READS_PER_CELL)
        fast = run_sweep(chemistry, component_name, "sub", fast=True, reads=SMOKE_READS_PER_CELL)

        assert_that(full.wrong).is_equal_to(fast.wrong)
        assert_that(full.correct).is_equal_to(fast.correct)

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
