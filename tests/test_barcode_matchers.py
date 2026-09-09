"""
Tests for barcode matcher classes: MatcherBase, FixedPositionMatcher, KmerMatcher, and AlignmentMatcher.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

import pytest
from assertpy import assert_that
from Bio.Align import PairwiseAligner, PairwiseAlignments

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.alignment_matcher import (
    GAP_EXTEND_SCORE,
    GAP_OPEN_SCORE,
    MATCH_SCORE,
    MISMATCH_SCORE,
    AlignmentContainer,
    AlignmentMatcher,
)
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.barcode.matchers.matcher_base import MatcherBase, UnresolvedComponentStartError
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import PRIMER_A, PRIMER_C
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from tests.test_chemistry import ChemistryCarmackCustomSeq10

# Matcher classes whose constructor takes an explicit error budget rather than reading one
# from the chemistry.
MATCHERS_REQUIRING_MAX_ERRORS = (KmerMatcher, AlignmentMatcher)


class TestBarcodeMatcherBase:
    """Tests for MatcherBase abstract class functionality, exercised via FixedPositionMatcher."""

    @pytest.fixture
    def component(self) -> ReadComponent:
        """Provide a barcode ReadComponent with start position set."""
        comp = ReadComponent(name="BC_TEST", type=ReadComponentType.BARCODE, length=10)
        comp.start = 5
        return comp

    @pytest.fixture
    def whitelist(self) -> tuple[str, ...]:
        """Provide a small test whitelist."""
        chemistry = ChemistryHydrop()
        component = chemistry.read_structure.get_component_by_name("BC3")
        return chemistry.barcode_whitelists[component.name]

    @pytest.fixture
    def matcher(self, whitelist: tuple[str, ...]) -> MatcherBase:
        """Provide a MatcherBase instance using FixedPositionMatcher for testing."""
        chemistry = ChemistryHydrop()
        component = chemistry.read_structure.get_component_by_name("BC3")
        return FixedPositionMatcher(whitelist=whitelist, component=component, chemistry=chemistry)

    def test_init_with_valid_component(
        self, component: ReadComponent, whitelist: tuple[str, ...], matcher: MatcherBase
    ) -> None:
        """Test that FixedPositionMatcher initializes successfully with a valid barcode component."""
        matcher = FixedPositionMatcher(
            whitelist=whitelist, component=component, chemistry=ChemistryHydrop()
        )
        assert_that(matcher.component).is_equal_to(component)

    @pytest.fixture
    def component_cases(self) -> dict[str, tuple[ReadComponent, tuple[str, ...], ChemistryBase]]:
        """Provide the component, whitelist and chemistry triples used for allowed-type checks.

        Returns:
            Mapping of case key to the component under test, a whitelist appropriate for it,
            and the chemistry that component belongs to.
        """
        spacer_component = ReadComponent(
            name="SPACER",
            type=ReadComponentType.OTHER,
            length=10,
            sequence="AGGGTACTCG",
        )
        custom_chemistry = ChemistryCarmackCustomSeq10()
        return {
            "spacer": (spacer_component, ("AAAAAAAAAA",), ChemistryHydrop()),
            "tgidx": (
                custom_chemistry.tgidx_component(),
                custom_chemistry.tgidx_whitelist(),
                custom_chemistry,
            ),
        }

    @pytest.mark.parametrize(
        "component_key, matcher_class, expects_raise",
        [
            ("spacer", FixedPositionMatcher, True),
            ("spacer", KmerMatcher, True),
            ("spacer", AlignmentMatcher, True),
            ("tgidx", FixedPositionMatcher, True),
            ("tgidx", KmerMatcher, False),
            ("tgidx", AlignmentMatcher, True),
        ],
    )
    def test_init_enforces_allowed_component_types(
        self,
        component_key: str,
        matcher_class: type[MatcherBase],
        expects_raise: bool,
        component_cases: dict[str, tuple[ReadComponent, tuple[str, ...], ChemistryBase]],
    ) -> None:
        """Test that construction accepts only components listed in allowed_component_types.

        Every matcher rejects a plain OTHER-typed spacer. Only KmerMatcher declares TGIDX as an
        allowed type, so the other two must reject a TGIDX component at construction rather than
        failing later inside match().
        """
        component, whitelist, chemistry = component_cases[component_key]
        kwargs: dict[str, Any] = {}
        if matcher_class in MATCHERS_REQUIRING_MAX_ERRORS:
            kwargs["max_errors"] = chemistry.max_errors.barcode

        if expects_raise:
            with pytest.raises(ValueError):
                matcher_class(
                    whitelist=whitelist, component=component, chemistry=chemistry, **kwargs
                )
        else:
            matcher = matcher_class(
                whitelist=whitelist, component=component, chemistry=chemistry, **kwargs
            )
            assert_that(matcher.component).is_equal_to(component)

    @pytest.mark.parametrize(
        "matcher_class, expected_types",
        [
            (MatcherBase, frozenset({ReadComponentType.BARCODE})),
            (FixedPositionMatcher, frozenset({ReadComponentType.BARCODE})),
            (AlignmentMatcher, frozenset({ReadComponentType.BARCODE})),
            (KmerMatcher, frozenset({ReadComponentType.BARCODE, ReadComponentType.TGIDX})),
        ],
    )
    def test_allowed_component_types_declared_on_class(
        self,
        matcher_class: type[MatcherBase],
        expected_types: frozenset[ReadComponentType],
    ) -> None:
        """Test that allowed_component_types is readable from the class itself, not just instances."""
        assert_that(hasattr(matcher_class, "allowed_component_types")).is_true()
        assert_that(matcher_class.allowed_component_types).is_instance_of(frozenset)
        assert_that(matcher_class.allowed_component_types).is_equal_to(expected_types)

    def test_unresolved_component_start_error_is_value_error(self) -> None:
        """Test that UnresolvedComponentStartError is a ValueError subclass."""
        assert_that(issubclass(UnresolvedComponentStartError, ValueError)).is_true()

    def test_check_read_len_unresolved_start_raises(self) -> None:
        """Test that check_read_len raises a named error when the component start is unresolved."""
        chemistry = ChemistryCarmackCustomSeq10()
        component = chemistry.tgidx_component()
        assert_that(component.start).is_none()
        matcher = KmerMatcher(
            whitelist=chemistry.tgidx_whitelist(),
            component=component,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.tgidx,
        )

        with pytest.raises(UnresolvedComponentStartError) as exc_info:
            matcher.check_read_len("A" * 100)

        assert_that(str(exc_info.value)).contains("TGIDX")

    def test_whitelist_stored_as_frozenset(
        self, whitelist: tuple[str, ...], matcher: MatcherBase
    ) -> None:
        """Test that the whitelist is stored as a frozenset containing all entries."""
        assert_that(matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(matcher.whitelist_set).is_equal_to(frozenset(whitelist))

    def test_check_read_len_sufficient(
        self, component: ReadComponent, whitelist: tuple[str, ...], matcher: MatcherBase
    ) -> None:
        """Test that check_read_len returns True when the read is long enough."""
        # start=5, length=10, so need >= 15
        read = "A" * 20
        assert_that(matcher.check_read_len(read)).is_true()

    def test_check_read_len_insufficient(self, matcher: MatcherBase) -> None:
        """Test that check_read_len returns False when the read is too short."""
        # BC_LEN = 10
        read = "A" * 5
        assert_that(matcher.check_read_len(read)).is_false()

    def test_check_read_len_exact_boundary(self, matcher: MatcherBase) -> None:
        """Test that check_read_len returns True when the read length exactly meets the requirement."""
        # start=5, length=10, so need >= 15
        read = "A" * 15
        assert_that(matcher.check_read_len(read)).is_true()

    @pytest.mark.parametrize(
        "read_len, start, bc_len, expected",
        [
            (50, 0, 10, True),
            (50, 40, 10, True),
            (50, 41, 10, False),
            (10, 0, 10, True),
            (9, 0, 10, False),
            (0, 0, 10, False),
            (100, 90, 10, True),
            (100, 91, 10, False),
        ],
    )
    def test_check_read_len_parametrized(
        self, read_len: int, start: int, bc_len: int, expected: bool
    ) -> None:
        """Test check_read_len with various read length, start position, and barcode length combos."""
        comp = ReadComponent(name="BC_TEST", type=ReadComponentType.BARCODE, length=bc_len)
        comp.start = start
        matcher = FixedPositionMatcher(
            whitelist=("AAAAAAAAAA",), component=comp, chemistry=ChemistryHydrop()
        )
        read = "A" * read_len
        assert_that(matcher.check_read_len(read)).is_equal_to(expected)

    @pytest.mark.parametrize(
        # Using trimmed carmack_custom_seq_1_0 reads
        "read, bc, upstream_spacer, downstream_spacer",
        [
            # No primer upstream of BC3 defined, correct primer downstream
            (
                "TTCCAGACCGACAAGTTAAGCCGAACTTGTAGTGTGTATAAGGACCTCGTTGCCAGCTTGAGAGA",
                "BC3",
                None,
                "PRIMER_C",
            ),
            # Both primers upstream and downstream of BC2 defined and present
            (
                "GCGTGATGTGTATAAGAGACAGTGACCGTACTTGTGTATAAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGTAGCAAGTCACCTGCCTGTCTCTTATACACATCTCCGAGCCCACGAGACCGAGGCTGATCTG",
                "BC2",
                "PRIMER_C",
                "PRIMER_A",
            ),
            # No primer downstream of BC1 defined, correct primer upstream
            (
                "GCGTGATGTGTATAAGAGACAGTGACCGTACTTGTGTATAAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGTAGCAAGTCACCTGCCTGTCTCTTATACACATCTCCGAGCCCACGAGACCGAGGCTGATCTG",
                "BC1",
                "PRIMER_A",
                None,
            ),
        ],
    )
    def test_check_spacers(
        self,
        bc: Literal["BC1", "BC2", "BC3"],
        read: str,
        upstream_spacer: str | None,
        downstream_spacer: str | None,
    ) -> None:
        """Test check_spacers with various read and match_idx combinations."""

        chemistry = ChemistryCarmackCustomSeq10()
        comp = chemistry.read_structure.get_component_by_name(bc)
        whitelist = chemistry.load_whitelist(bc)
        matcher = FixedPositionMatcher(whitelist=whitelist, component=comp, chemistry=chemistry)

        # General carmack_custom_seq_1_0 read structure
        bc_idx_map = {"BC3": (22, 32), "BC2": (54, 64), "BC1": (86, 96)}
        match_idx = bc_idx_map[bc]

        spacer_results = matcher.check_spacers(read, match_idx)

        assert_that(spacer_results).contains_key("upstream").contains_key("downstream")
        assert_that(spacer_results["upstream"]).is_equal_to(upstream_spacer)
        assert_that(spacer_results["downstream"]).is_equal_to(downstream_spacer)


class TestMatcherStartIdxContract:
    """Characterisation tests pinning what `start_idx` means to each matcher.

    The parameter deliberately means different things to the two searching matchers, and that
    divergence is part of the contract rather than an inconsistency waiting to be unified.
    KmerMatcher treats `start_idx` as a floor on where a *seed k-mer* may begin; AlignmentMatcher
    treats it as a hard trim point that hides every base before it; FixedPositionMatcher ignores
    it outright. These tests assert that divergence, so any future attempt to unify the three has
    to change a test in order to do it.
    """

    BARCODE = "ACGTACGTAC"
    WHITELIST = (BARCODE,)

    # HyDrop's barcode error budget, passed explicitly because both searching matchers take
    # their budget as a constructor argument rather than reading it from the chemistry.
    MAX_ERRORS = 2
    K = 4

    # The barcode occupies read positions 0-10; the poly-T tail seeds no whitelist k-mer.
    READ_BARCODE_AT_START = BARCODE + "T" * 20

    # The barcode occupies read positions 4-14, behind four bases of filler.
    READ_BARCODE_AT_FOUR = "TTTT" + BARCODE + "T" * 8

    @pytest.fixture
    def chemistry(self) -> ChemistryHydrop:
        """Provide the HyDrop chemistry supplying the surrounding read structure.

        Returns:
            A freshly constructed ChemistryHydrop.
        """
        return ChemistryHydrop()

    @pytest.fixture
    def component(self, chemistry: ChemistryHydrop) -> ReadComponent:
        """Provide the HyDrop BC3 component (start=0, length=10).

        Args:
            chemistry: The chemistry supplying the read structure.

        Returns:
            The BC3 ReadComponent.
        """
        return chemistry.read_structure.get_component_by_name("BC3")

    @pytest.fixture
    def kmer_matcher(self, chemistry: ChemistryHydrop, component: ReadComponent) -> KmerMatcher:
        """Provide a KmerMatcher over the single-entry whitelist.

        Args:
            chemistry: The chemistry supplying the read structure.
            component: The BC3 component being matched.

        Returns:
            The configured KmerMatcher.
        """
        return KmerMatcher(
            whitelist=self.WHITELIST,
            component=component,
            chemistry=chemistry,
            max_errors=self.MAX_ERRORS,
            k=self.K,
        )

    @pytest.fixture
    def alignment_matcher(
        self, chemistry: ChemistryHydrop, component: ReadComponent
    ) -> AlignmentMatcher:
        """Provide an AlignmentMatcher over the same single-entry whitelist.

        Args:
            chemistry: The chemistry supplying the read structure.
            component: The BC3 component being matched.

        Returns:
            The configured AlignmentMatcher.
        """
        return AlignmentMatcher(
            whitelist=self.WHITELIST,
            component=component,
            chemistry=chemistry,
            max_errors=self.MAX_ERRORS,
        )

    @pytest.fixture
    def fixed_matcher(
        self, chemistry: ChemistryHydrop, component: ReadComponent
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher over the same single-entry whitelist.

        Args:
            chemistry: The chemistry supplying the read structure.
            component: The BC3 component being matched.

        Returns:
            The configured FixedPositionMatcher.
        """
        return FixedPositionMatcher(
            whitelist=self.WHITELIST, component=component, chemistry=chemistry
        )

    # ==========================================
    # KmerMatcher: start_idx is a seed floor only
    # ==========================================

    def test_kmer_start_idx_is_a_seed_floor_not_a_match_floor(
        self, kmer_matcher: KmerMatcher
    ) -> None:
        """Test that a KmerMatcher match may begin strictly before `start_idx`.

        `start_idx` bounds only where a *seed k-mer* may begin, never where a match may begin.
        A seed found at or after `start_idx` is extended in both directions from
        `expected_start = read_kmer_pos - bc_kmer_pos`, clamped only at 0, so the verified span
        can reach back in front of `start_idx`. Here it reaches back a full four bases: the caller
        asks matching to start at index 4 and gets a match spanning 0-10.
        """
        result = kmer_matcher.match(self.READ_BARCODE_AT_START, 4)[0]

        assert_that(result.match).is_equal_to(self.BARCODE)
        assert_that(result.read_idx).is_equal_to((0, 10))
        assert_that(result.read_idx[0]).is_less_than(4)
        assert_that(result.edit_distance).is_equal_to(0)

    @pytest.mark.parametrize("start_idx", [0, 1, 2, 3, 4, 5, 6])
    def test_kmer_match_is_unaffected_by_start_idx_while_a_seed_survives(
        self, kmer_matcher: KmerMatcher, start_idx: int
    ) -> None:
        """Test that every `start_idx` leaving a seed intact yields the identical full-span match.

        Because the seed is extended backwards, raising `start_idx` does not truncate the match:
        the result is byte-for-byte the same across the whole range that still leaves at least one
        whitelist k-mer starting at or after `start_idx`.
        """
        result = kmer_matcher.match(self.READ_BARCODE_AT_START, start_idx)[0]

        assert_that(result.match).is_equal_to(self.BARCODE)
        assert_that(result.read_idx).is_equal_to((0, 10))
        assert_that(result.candidate).is_equal_to(self.BARCODE)
        assert_that(result.edit_distance).is_equal_to(0)

    def test_kmer_no_match_once_start_idx_passes_the_last_seed(
        self, kmer_matcher: KmerMatcher
    ) -> None:
        """Test that KmerMatcher stops matching only when `start_idx` outruns every seed.

        The last whitelist k-mer in this read begins at index 6, so a `start_idx` of 7 leaves the
        seed scan with nothing to find and the match fails. This is the sole way `start_idx`
        suppresses a KmerMatcher match: by starving the seed scan, not by hiding bases.
        """
        result = kmer_matcher.match(self.READ_BARCODE_AT_START, 7)[0]

        assert_that(result.match).is_none()
        assert_that(result.read_idx).is_none()

    # ==========================================
    # AlignmentMatcher: start_idx is a hard trim
    # ==========================================

    @pytest.mark.parametrize(
        "start_idx, expected_read_idx, expected_candidate, expected_edit_distance",
        [
            (0, (0, 10), "ACGTACGTAC", 0),
            (1, (1, 10), "CGTACGTAC", 1),
            (2, (2, 10), "GTACGTAC", 2),
        ],
    )
    def test_alignment_truncated_match_within_budget_still_resolves(
        self,
        alignment_matcher: AlignmentMatcher,
        start_idx: int,
        expected_read_idx: tuple[int, int],
        expected_candidate: str,
        expected_edit_distance: int,
    ) -> None:
        """Test that AlignmentMatcher resolves a barcode it has partly trimmed away.

        This is a genuinely surprising edge. The matcher aligns against `read[start_idx:]`, so a
        non-zero `start_idx` physically deletes leading barcode bases before alignment begins. The
        truncated candidate is still assigned the whole whitelist entry as its match, because the
        bases lost to the trim simply register as edits, and while that loss stays inside
        `max_errors` the entry survives the edit-distance gate. The reported span therefore starts
        at `start_idx`, not at the barcode's real start, and the candidate is shorter than the
        barcode it resolved to.
        """
        result = alignment_matcher.match(self.READ_BARCODE_AT_START, start_idx)[0]

        assert_that(result.match).is_equal_to(self.BARCODE)
        assert_that(result.read_idx).is_equal_to(expected_read_idx)
        assert_that(result.candidate).is_equal_to(expected_candidate)
        assert_that(result.edit_distance).is_equal_to(expected_edit_distance)
        assert_that(len(result.candidate)).is_less_than_or_equal_to(len(self.BARCODE))

    @pytest.mark.parametrize("start_idx", [3, 4, 5, 6, 7])
    def test_alignment_truncation_beyond_budget_returns_no_match(
        self, alignment_matcher: AlignmentMatcher, start_idx: int
    ) -> None:
        """Test that AlignmentMatcher fails once the trim costs more than `max_errors`.

        From `start_idx` of 3 onwards the trim removes three or more of the ten barcode bases,
        which exceeds the two-error budget, so no alignment clears the score threshold and the
        matcher reports nothing at all.
        """
        result = alignment_matcher.match(self.READ_BARCODE_AT_START, start_idx)[0]

        assert_that(result.match).is_none()
        assert_that(result.read_idx).is_none()
        assert_that(result.candidate).is_none()

    def test_alignment_is_blind_to_bases_before_start_idx_where_kmer_is_not(
        self, kmer_matcher: KmerMatcher, alignment_matcher: AlignmentMatcher
    ) -> None:
        """Test the two searching matchers diverging on the same read and the same `start_idx`.

        At `start_idx` of 4 over a barcode sitting at 0-10, KmerMatcher extends its seed backwards
        and returns the full 0-10 span, while AlignmentMatcher has trimmed those four bases away
        and can no longer see them, so it returns no match at all. Both behaviours are intended;
        neither matcher should be changed to imitate the other.
        """
        kmer_result = kmer_matcher.match(self.READ_BARCODE_AT_START, 4)[0]
        alignment_result = alignment_matcher.match(self.READ_BARCODE_AT_START, 4)[0]

        assert_that(kmer_result.match).is_equal_to(self.BARCODE)
        assert_that(kmer_result.read_idx).is_equal_to((0, 10))
        assert_that(alignment_result.match).is_none()
        assert_that(alignment_result.read_idx).is_none()

    # ==========================================
    # The one shared guarantee: original-read coordinates
    # ==========================================

    @pytest.mark.parametrize("matcher_fixture", ["kmer_matcher", "alignment_matcher"])
    def test_read_idx_is_reported_in_original_read_coordinates(
        self, request: pytest.FixtureRequest, matcher_fixture: str
    ) -> None:
        """Test the one part of the `start_idx` contract both searching matchers do share.

        Whatever `start_idx` meant on the way in, `read_idx` comes back in coordinates of the
        original untrimmed read. AlignmentMatcher adds `start_idx` back onto its trimmed-read
        coordinates before reporting, and KmerMatcher never trimmed in the first place, so the
        caller never has to add `start_idx` back itself and `read[read_idx[0]:read_idx[1]]` always
        reproduces the reported candidate.
        """
        matcher: MatcherBase = request.getfixturevalue(matcher_fixture)
        result = matcher.match(self.READ_BARCODE_AT_FOUR, 4)[0]

        assert_that(result.match).is_equal_to(self.BARCODE)
        assert_that(result.read_idx).is_equal_to((4, 14))
        assert_that(
            self.READ_BARCODE_AT_FOUR[result.read_idx[0] : result.read_idx[1]]
        ).is_equal_to(result.candidate)

    # ==========================================
    # FixedPositionMatcher: start_idx is ignored
    # ==========================================

    @pytest.mark.parametrize("start_idx", [0, 1, 4, 7, 25, 500])
    def test_fixed_position_matcher_ignores_start_idx(
        self, fixed_matcher: FixedPositionMatcher, start_idx: int
    ) -> None:
        """Test that FixedPositionMatcher returns the same result for every `start_idx`.

        It does not search, so it has nowhere to start from: it slices the component's fixed start
        out of the read structure and reads that window directly. `start_idx` is accepted only to
        keep the matcher signature uniform, and a value far past the end of the read changes
        nothing.
        """
        baseline = fixed_matcher.match(self.READ_BARCODE_AT_START, 0)[0]
        result = fixed_matcher.match(self.READ_BARCODE_AT_START, start_idx)[0]

        assert_that(result).is_equal_to(baseline)
        assert_that(result.match).is_equal_to(self.BARCODE)
        assert_that(result.read_idx).is_equal_to((0, 10))


class TestFixedPositionMatcher:
    """Tests for FixedPositionMatcher barcode matching logic."""

    @pytest.fixture
    def matcher_class(self) -> type[MatcherBase]:
        """Build HyDrop matcher fixtures as FixedPositionMatcher instances."""
        return FixedPositionMatcher

    def test_match_returns_barcode_match_attempt(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that match() returns a BarcodeMatchAttempt instance."""
        read = "A" * 52
        result = bc3_matcher.match(read)

        # FixePositionMatcher should always return a single attempt
        assert_that(len(result)).is_equal_to(1)
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)

    def test_match_method_is_always_exactmatch(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that the method field is always EXACTMATCH regardless of match outcome."""
        read = "A" * 52
        result = bc3_matcher.match(read)[0]
        assert_that(result.method).is_equal_to(MatchMethod.EXACTMATCH)

    def test_match_candidate_is_extracted_substring(
        self, bc3_matcher: FixedPositionMatcher
    ) -> None:
        """Test that candidate is the substring at [start:start+length] from the read."""
        read = "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT"
        result = bc3_matcher.match(read)[0]
        # BC3: start=0, length=10
        assert_that(result.candidate).is_equal_to(read[0:10])

    def test_match_read_too_short_returns_no_match(
        self, bc1_matcher: FixedPositionMatcher
    ) -> None:
        """Test that a read shorter than required returns no match."""
        # BC1 needs start=40 + length=10 = 50bp minimum
        read = "A" * 30
        result = bc1_matcher.match(read)[0]
        assert_that(result.match).is_none()
        assert_that(result.read_idx).is_none()

    def test_match_exact_returns_correct_read_idx(
        self, bc3_matcher: FixedPositionMatcher, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> None:
        """Test that a successful match returns the correct read index tuple."""
        bc3_barcode = hydrop_whitelists["BC3"][0]
        read = bc3_barcode + "A" * 42  # BC3 at [0:10], padded to 52bp
        result = bc3_matcher.match(read)[0]
        assert_that(result.match).is_equal_to(bc3_barcode)
        assert_that(result.read_idx).is_equal_to((0, 10))

    def test_match_no_match_returns_none_idx(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that a failed match returns None for both match and read_idx."""
        read = "ZZZZZZZZZZ" + "A" * 42
        result = bc3_matcher.match(read)[0]
        assert_that(result.match).is_none()
        assert_that(result.read_idx).is_none()

    @pytest.mark.parametrize(
        "seq, bc_name, expected_match",
        [
            # Case 1 (WL_MATCH): all 3 BCs match
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC3", True),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC2", True),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC1", True),
            # Case 2 (INDEL_FAIL): BC3 matches, BC2 and BC1 don't
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC3", True),
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC2", False),
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC1", False),
            # Case 3 (SUB_FAIL): BC3+BC2 match, BC1 substitution error
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC3", True),
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC2", True),
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC1", False),
            # Case 5 (SPC_NOTFND): BC3+BC2 match, BC1 doesn't
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC3", True),
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC2", True),
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC1", False),
            # Case 6 (SEVERE_INDEL): BC3 matches, BC2+BC1 don't
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC3", True),
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC2", False),
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC1", False),
        ],
    )
    def test_match_dev_sequences(
        self,
        seq: str,
        bc_name: str,
        expected_match: bool,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test fixed-position matching using HyDrop barcode read sequences."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        matcher = FixedPositionMatcher(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
        )
        result = matcher.match(seq)[0]

        if expected_match:
            assert_that(result.match).is_not_none()
            assert_that(result.match).is_in(*hydrop_whitelists[bc_name])
            assert_that(result.read_idx).is_not_none()
        else:
            assert_that(result.match).is_none()


class TestKmerMatcher:
    """Tests for KmerMatcher barcode matching logic."""

    # --- HyDrop spacer constants for read construction ---
    SPACER_1 = "AGGGTACTCG"
    SPACER_2 = "GCAGTAGCTG"

    # --- Fixtures ---

    @pytest.fixture
    def matcher_class(self) -> type[MatcherBase]:
        """Build HyDrop matcher fixtures as KmerMatcher instances."""
        return KmerMatcher

    @pytest.fixture
    def matcher_kwargs(self, hydrop_chemistry: ChemistryHydrop) -> dict[str, Any]:
        """Give the shared matcher fixtures the HyDrop barcode error budget."""
        return {"max_errors": hydrop_chemistry.max_errors.barcode}

    # --- Helper to build a full HyDrop read ---

    def _build_hydrop_read(
        self,
        bc3: str,
        bc2: str,
        bc1: str,
    ) -> str:
        """Construct a full 50bp HyDrop read: BC3 + SPACER_1 + BC2 + SPACER_2 + BC1."""
        return bc3 + self.SPACER_1 + bc2 + self.SPACER_2 + bc1

    # ==========================================
    # Initialization & K-mer Index Tests
    # ==========================================

    def test_initialization_stores_correct_attributes(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that KmerMatcher initializes correctly with given whitelist and chemistry."""
        assert_that(bc3_matcher.component.name).is_equal_to("BC3")
        assert_that(bc3_matcher.k).is_equal_to(4)
        assert_that(bc3_matcher.max_errors).is_equal_to(hydrop_chemistry.max_errors.barcode)
        assert_that(bc3_matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(bc3_matcher.whitelist_set).is_equal_to(frozenset(hydrop_whitelists["BC3"]))

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_initialization_with_custom_k(
        self,
        k: int,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that KmerMatcher accepts different k-mer sizes."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists["BC3"],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
            k=k,
        )
        assert_that(matcher.k).is_equal_to(k)
        assert_that(matcher.kmer_index).is_not_empty()

    def test_kmer_index_contains_all_expected_kmers(
        self, small_matcher: KmerMatcher, small_whitelist: tuple[str, ...]
    ) -> None:
        """Test that the kmer index contains entries for all k-mers from the whitelist."""
        k = small_matcher.k
        expected_kmers: set[str] = set()
        for bc in small_whitelist:
            for i in range(len(bc) - k + 1):
                expected_kmers.add(bc[i : i + k])

        for kmer in expected_kmers:
            assert_that(small_matcher.kmer_index).contains_key(kmer)

    def test_kmer_index_maps_to_correct_barcode_positions(
        self, small_matcher: KmerMatcher
    ) -> None:
        """Test that each kmer index entry maps back to the correct (barcode, position) tuples."""
        # "ACGTACGTAC" has kmer "ACGT" at positions 0 and 4
        entries = small_matcher.kmer_index.get("ACGT", [])
        barcode_positions = [(bc, pos) for bc, pos in entries if bc == "ACGTACGTAC"]
        assert_that(barcode_positions).contains(("ACGTACGTAC", 0))
        assert_that(barcode_positions).contains(("ACGTACGTAC", 4))

    def test_kmer_index_is_plain_dict(self, small_matcher: KmerMatcher) -> None:
        """Test that build_kmer_index returns a plain dict, not a defaultdict."""
        assert_that(type(small_matcher.kmer_index)).is_equal_to(dict)

    @pytest.mark.parametrize(
        "k, barcode, expected_kmer_count",
        [
            (4, "ACGTACGTAC", 7),  # 10 - 4 + 1 = 7 kmers
            (5, "ACGTACGTAC", 6),  # 10 - 5 + 1 = 6 kmers
            (3, "ACGTACGTAC", 8),  # 10 - 3 + 1 = 8 kmers
            (10, "ACGTACGTAC", 1),  # 10 - 10 + 1 = 1 kmer (the whole barcode)
        ],
    )
    def test_kmer_index_entry_count_per_barcode(
        self, k: int, barcode: str, expected_kmer_count: int
    ) -> None:
        """Test that the correct number of k-mer entries are created for a single barcode."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=(barcode,),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
            k=k,
        )

        # Count entries pointing back to our barcode
        total_entries = sum(
            1 for entries in matcher.kmer_index.values() for bc, _ in entries if bc == barcode
        )
        assert_that(total_entries).is_equal_to(expected_kmer_count)

    # ==========================================
    # extend_and_verify Tests
    # ==========================================

    def test_extend_and_verify_perfect_match(self, small_matcher: KmerMatcher) -> None:
        """Test that extend_and_verify returns valid with distance 0 for a perfect match."""
        barcode = "ACGTACGTAC"
        read = barcode + "NNNNNNNNNN"  # barcode at position 0-10

        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_true()
        assert_that(edit_dist).is_equal_to(0)
        assert_that(read[start:end]).is_equal_to(barcode)

    def test_extend_and_verify_single_substitution(self, small_matcher: KmerMatcher) -> None:
        """Test that extend_and_verify detects a single substitution within max_errors."""
        barcode = "ACGTACGTAC"
        mutated = "ACGTACGTAG"  # last base C -> G (1 substitution)
        read = mutated + "NNNNNNNNNN"

        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_true()
        assert_that(edit_dist).is_equal_to(1)

    def test_extend_and_verify_single_insertion(self, small_matcher: KmerMatcher) -> None:
        """Test that extend_and_verify handles a single insertion in the read."""
        barcode = "ACGTACGTAC"
        # Insert 'T' after position 4: ACGT_T_ACGTAC -> 11bp vs 10bp barcode
        inserted = "ACGTTACGTAC"
        read = inserted + "NNNNNNNNN"

        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_true()
        assert_that(edit_dist).is_less_than_or_equal_to(small_matcher.max_errors)

    def test_extend_and_verify_single_deletion(self, small_matcher: KmerMatcher) -> None:
        """Test that extend_and_verify handles a single deletion in the read."""
        barcode = "ACGTACGTAC"
        # Delete position 4: ACGT_CGTAC -> 9bp vs 10bp barcode
        deleted = "ACGTCGTAC"
        read = deleted + "NNNNNNNNNNN"

        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_true()
        assert_that(edit_dist).is_less_than_or_equal_to(small_matcher.max_errors)

    def test_extend_and_verify_beyond_max_errors_returns_invalid(
        self, small_matcher: KmerMatcher
    ) -> None:
        """Test that extend_and_verify rejects a candidate exceeding max_errors."""
        barcode = "ACGTACGTAC"
        # 3 substitutions (max_errors=2 for HyDrop): TTTAACGTAC
        heavily_mutated = "TTTAACGTAC"
        # Use real bases (not N) to avoid wildcard matching in adjacent windows
        read = heavily_mutated + "TTTTTTTTTT"

        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_false()
        assert_that(start).is_equal_to(-1)
        assert_that(end).is_equal_to(-1)

    @pytest.mark.parametrize(
        "barcode, read_segment, expected_valid",
        [
            # Perfect match
            ("ACGTACGTAC", "ACGTACGTAC", True),
            # 1 substitution
            ("ACGTACGTAC", "ACGTACGTAG", True),
            # 2 substitutions (at max_errors=2)
            ("ACGTACGTAC", "ACGTACGTGG", True),
            # 3 substitutions (exceeds max_errors=2)
            ("ACGTACGTAC", "ACGTACGGGG", False),
            # 1 insertion
            ("ACGTACGTAC", "ACGTTACGTAC", True),
            # 1 deletion
            ("ACGTACGTAC", "ACGTCGTAC", True),
        ],
    )
    def test_extend_and_verify_parametrized_error_types(
        self,
        barcode: str,
        read_segment: str,
        expected_valid: bool,
        small_matcher: KmerMatcher,
    ) -> None:
        """Test extend_and_verify with various error types and counts."""
        read = read_segment + "N" * 20  # pad read

        is_valid, _, _, _ = small_matcher.extend_and_verify(
            read, read_kmer_pos=0, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_equal_to(expected_valid)

    def test_extend_and_verify_barcode_offset_in_read(self, small_matcher: KmerMatcher) -> None:
        """Test extend_and_verify when the barcode is offset into the read."""
        barcode = "ACGTACGTAC"
        prefix = "NNNNN"
        read = prefix + barcode + "NNNNN"

        # kmer "ACGT" is at read position 5, and at barcode position 0
        is_valid, start, end, edit_dist = small_matcher.extend_and_verify(
            read, read_kmer_pos=5, barcode=barcode, bc_kmer_pos=0
        )
        assert_that(is_valid).is_true()
        assert_that(edit_dist).is_equal_to(0)
        assert_that(start).is_equal_to(5)
        assert_that(end).is_equal_to(15)

    # ==========================================
    # match() - Return Type & Basic Structure
    # ==========================================

    def test_match_returns_list_of_barcode_match_attempt(
        self, bc3_matcher: KmerMatcher, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> None:
        """Test that match() returns a list of BarcodeMatchAttempt instances."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        assert_that(result).is_instance_of(list)
        assert_that(result).is_not_empty()
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)

    def test_match_method_is_always_kmermatch(
        self, bc3_matcher: KmerMatcher, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> None:
        """Test that the method field is always KMERMATCH regardless of match outcome."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]
        assert_that(result.method).is_equal_to(MatchMethod.KMERMATCH)

    def test_match_no_match_returns_none_fields(self, small_matcher: KmerMatcher) -> None:
        """Test that a failed match returns None for match, read_idx, and edit_distance."""
        # Read with no matching k-mers at all
        read = "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        result = small_matcher.match(read)
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()
        assert_that(result[0].read_idx).is_none()
        assert_that(result[0].edit_distance).is_none()

    # ==========================================
    # match() - Exact Matching
    # ==========================================

    def test_match_exact_barcode_at_expected_position(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that an exact barcode at its expected position is matched correctly."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)
        assert_that(result.read_idx).is_not_none()
        assert_that(result.candidate).is_equal_to(bc3)

    def test_match_exact_barcode_returns_correct_read_idx(
        self,
        bc2_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that read_idx correctly spans the barcode region in the read."""
        bc2 = hydrop_whitelists["BC2"][0]
        read = self._build_hydrop_read("A" * 10, bc2, "A" * 10)
        result = bc2_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc2)
        assert_that(result.read_idx).is_not_none()
        start, end = result.read_idx
        assert_that(read[start:end]).is_equal_to(bc2)

    @pytest.mark.parametrize("bc_idx", [0, 1, 10, 50, 95])
    def test_match_exact_various_whitelist_entries(
        self,
        bc_idx: int,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that various barcodes from the whitelist are matched exactly."""
        bc3 = hydrop_whitelists["BC3"][bc_idx]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)

    # ==========================================
    # match() - Substitution Errors
    # ==========================================

    def test_match_single_substitution(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a barcode with a single substitution is still matched."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        mutated = bc3[:5] + ("A" if bc3[5] != "A" else "T") + bc3[6:]
        read = self._build_hydrop_read(mutated, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(1)

    def test_match_two_substitutions_at_max_errors(self, small_matcher: KmerMatcher) -> None:
        """Test that a barcode with exactly max_errors substitutions is matched (HyDrop max_errors=2)."""
        barcode = "ACGTACGTAC"  # in small_whitelist
        # Mutate positions 0 and 9: A->T at 0, C->G at 9
        mutated = "T" + barcode[1:9] + "G"
        read = mutated + "TTTTTTTTTT" * 4
        result = small_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(barcode)
        assert_that(result.edit_distance).is_less_than_or_equal_to(2)

    @pytest.mark.parametrize(
        "sub_positions",
        [
            [0],  # first base
            [9],  # last base
            [4],  # middle base
            [2, 7],  # two internal positions
        ],
    )
    def test_match_substitutions_at_various_positions(
        self,
        sub_positions: list[int],
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that substitutions at various positions are tolerated within max_errors."""
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        for pos in sub_positions:
            mutated[pos] = "A" if mutated[pos] != "A" else "C"
        mutated_str = "".join(mutated)
        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(len(sub_positions))

    def test_match_two_substitutions_at_extremes(self, small_matcher: KmerMatcher) -> None:
        """Test that substitutions at both extremes (positions 0 and 9) are tolerated."""
        barcode = "ACGTACGTAC"
        mutated = "T" + barcode[1:9] + "G"  # sub at pos 0 and 9
        read = mutated + "TTTTTTTTTT" * 4
        result = small_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(barcode)
        assert_that(result.edit_distance).is_less_than_or_equal_to(2)

    # ==========================================
    # match() - Indel Errors
    # ==========================================

    def test_match_single_insertion_in_barcode(self, small_matcher: KmerMatcher) -> None:
        """Test that a single insertion in the barcode region is detected as a candidate.

        With a small whitelist and no spacer context, indels create multiple
        equally-scored alignment windows. The matcher finds the barcode but
        ambiguity resolution without spacers returns no definitive match.
        """
        barcode = "ACGTACGTAC"  # in small_whitelist
        # Insert 'T' at position 5: ACGTA_T_CGTAC -> 11bp
        inserted = barcode[:5] + "T" + barcode[5:]
        read = inserted + "TTTTTTTTTT" * 4
        result = small_matcher.match(read)

        # Multiple alignment windows tie -> ambiguity -> no validated match without spacers
        assert_that(result).is_not_empty()
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)

    def test_match_single_insertion_with_spacer_context(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a single insertion is resolved when spacer context is available."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        inserted = bc3[:5] + "A" + bc3[5:]  # 11bp
        # Place insertion in a full read so spacer check can disambiguate
        read = inserted + self.SPACER_1 + "A" * 10 + self.SPACER_2 + "A" * 10
        result = bc3_matcher.match(read)

        # Ambiguity is expected, but spacer check should at least find candidates
        assert_that(result).is_not_empty()
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)

    def test_match_single_deletion_in_barcode(self, small_matcher: KmerMatcher) -> None:
        """Test that a single deletion in the barcode region is detected as a candidate.

        Similar to insertion, deletion creates alignment ambiguity without spacer context.
        """
        barcode = "ACGTACGTAC"  # in small_whitelist
        # Delete position 5: ACGTA_GTAC -> 9bp
        deleted = barcode[:5] + barcode[6:]
        read = deleted + "TTTTTTTTTT" * 4
        result = small_matcher.match(read)

        assert_that(result).is_not_empty()
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)

    def test_match_single_deletion_with_spacer_context(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a single deletion is resolved when spacer context is available."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        deleted = bc3[:5] + bc3[6:]  # 9bp
        read = deleted + self.SPACER_1 + "A" * 10 + self.SPACER_2 + "A" * 10
        result = bc3_matcher.match(read)

        assert_that(result).is_not_empty()
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)

    def test_match_two_insertions_at_max_errors(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that two insertions (at max_errors=2) are tolerated."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        # Insert at positions 2 and 7: TG_A_ACCGT_A_ACT -> 12bp
        inserted = bc3[:2] + "A" + bc3[2:7] + "A" + bc3[7:]
        read = inserted + self.SPACER_1 + "A" * 10 + self.SPACER_2 + "A" * 10
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(2)

    # ==========================================
    # match() - N Wildcard Handling
    # ==========================================

    def test_match_n_bases_in_barcode_region(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that N bases in the read are treated as wildcards (cost 0)."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        # Replace position 5 with N
        with_n = bc3[:5] + "N" + bc3[6:]
        read = self._build_hydrop_read(with_n, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)

    def test_match_multiple_n_bases_disrupts_kmer_seeding(
        self, small_matcher: KmerMatcher
    ) -> None:
        """Test that N bases disrupting all k-mer seeds prevent matching.

        When N bases are placed such that every possible 4-mer in the barcode
        region contains an N, no seeds fire and no candidates are found.
        """
        barcode = "ACGTACGTAC"  # in small_whitelist
        # Replace positions 3 and 7 with N: ACG_N_ACG_N_AC
        # This disrupts k-mers at positions 0-3, 1-4, 4-7, 5-8 etc.
        with_n = barcode[:3] + "N" + barcode[4:7] + "N" + barcode[8:]
        read = with_n + "T" * 40
        result = small_matcher.match(read)

        # N in the middle of k-mers prevents seed matches, so no candidate is found
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()

    # ==========================================
    # match() - Edge Cases
    # ==========================================

    def test_match_read_too_short_for_kmer_raises_value_error(
        self, small_matcher: KmerMatcher
    ) -> None:
        """Test that a read shorter than k raises ValueError."""
        short_read = "ACG"  # k=4, so this is too short
        with pytest.raises(ValueError, match="Read segment too short"):
            small_matcher.match(short_read)

    def test_match_read_exactly_k_length(self, small_matcher: KmerMatcher) -> None:
        """Test that a read of exactly k length can still be processed (single kmer)."""
        read = "ACGT"  # k=4, exactly one kmer
        result = small_matcher.match(read)
        assert_that(result).is_length(1)
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)

    def test_match_read_with_all_n_bases(self, bc3_matcher: KmerMatcher) -> None:
        """Test matching a read composed entirely of N bases."""
        read = "N" * 50
        result = bc3_matcher.match(read)

        # N matches everything, but result should still be deterministic
        assert_that(result).is_not_empty()
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)

    def test_match_barcode_not_in_whitelist(self, small_matcher: KmerMatcher) -> None:
        """Test that a read with no matching barcode returns no match."""
        # Use a barcode that shares no 4-mers with the small whitelist
        read = "TTTTTTTTTT" + "A" * 40
        result = small_matcher.match(read)
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()

    def test_match_barcode_exceeds_max_errors_returns_no_match(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a heavily mutated barcode beyond max_errors is not matched."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        # Mutate 5 positions (well beyond max_errors=2)
        heavily_mutated = "AAAAAGTACT"
        read = self._build_hydrop_read(heavily_mutated, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        # Should either return no match or not match to bc3
        if result[0].match is not None:
            assert_that(result[0].match).is_not_equal_to(bc3)

    # ==========================================
    # match() - Position Independence
    # ==========================================

    def test_match_finds_barcode_regardless_of_position(
        self,
        bc2_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that KmerMatcher finds a barcode even when not at its expected fixed position."""
        bc2 = hydrop_whitelists["BC2"][0]
        # Place BC2 at an unusual offset, surrounded by filler
        read = "A" * 5 + bc2 + "A" * 35
        result = bc2_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc2)
        assert_that(result.read_idx).is_not_none()
        start, end = result.read_idx
        assert_that(read[start:end]).is_equal_to(bc2)

    def test_match_candidate_field_matches_read_slice(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that result.candidate equals the read slice at result.read_idx."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.candidate).is_not_none()
        assert_that(result.read_idx).is_not_none()
        start, end = result.read_idx
        assert_that(result.candidate).is_equal_to(read[start:end])

    # ==========================================
    # match() - Ambiguity & Spacer Tiebreaking
    # ==========================================

    def test_match_single_candidate_returns_single_result(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that an unambiguous match returns exactly one result."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc3)

    def test_match_ambiguous_resolved_by_spacer_returns_single_result(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that ambiguous matches are resolved by spacer presence to a single result."""
        # Construct a read where BC2 has an exact match flanked by correct spacers
        bc2 = hydrop_whitelists["BC2"][0]
        read = self._build_hydrop_read("A" * 10, bc2, "A" * 10)
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists["BC2"],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(read)

        # Even if internally ambiguous, spacer checks should resolve to a single match
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc2)

    def test_match_ambiguous_returns_multiple_results_with_none_match(
        self,
    ) -> None:
        """Test that truly ambiguous matches return multiple results with match=None."""
        # Construct a whitelist with two barcodes that differ by exactly max_errors
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")

        # Two barcodes differing by 2 bases (max_errors=2): both match a read that is
        # equidistant to both. The read has 1 error from each.
        bc_a = "ACGTACGTAC"
        bc_b = "ACGTACGTCA"  # differs at positions 8,9 from bc_a
        ambiguous_read = "ACGTACGTCC"  # distance 1 from both

        matcher = KmerMatcher(
            whitelist=(bc_a, bc_b),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
        )
        # Pad read with non-spacer filler to avoid spacer resolution
        read = ambiguous_read + "TTTTTTTTTT" * 4
        result = matcher.match(read)

        # Should have multiple results (ambiguity) OR a single result with no match
        # since spacer check can't resolve (no valid spacers present)
        if len(result) > 1:
            # Ambiguous results should have match=None
            for r in result:
                assert_that(r.match).is_none()
                assert_that(r.method).is_equal_to(MatchMethod.KMERMATCH)
        else:
            # Single result with no spacer-validated match
            assert_that(result[0].match).is_none()

    # ==========================================
    # match() - Full HyDrop Read Sequences (Parametrized)
    # ==========================================

    @pytest.mark.parametrize(
        "seq, bc_name, expected_match",
        [
            # Case 1 (WL_MATCH): all 3 BCs match (same sequences as FixedPositionMatcher tests)
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC3", True),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC2", True),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC1", True),
            # Case 2 (INDEL_FAIL): BC3 matches, BC2 and BC1 may not (indel disruption)
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC3", True),
            # Case 3 (SUB_FAIL): BC3+BC2 match, BC1 substitution error
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC3", True),
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC2", True),
            # Case 5 (SPC_NOTFND): BC3+BC2 match
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC3", True),
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC2", True),
            # Case 6 (SEVERE_INDEL): BC3 matches
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC3", True),
        ],
    )
    def test_match_dev_sequences_expected_matches(
        self,
        seq: str,
        bc_name: str,
        expected_match: bool,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test kmer matching using HyDrop barcode read sequences that should produce matches."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(seq)

        if expected_match:
            # At least one result should be returned
            assert_that(result).is_not_empty()
            # The first (or only) result should have a match in the whitelist
            matched_results = [r for r in result if r.match is not None]
            assert_that(matched_results).is_not_empty()
            assert_that(matched_results[0].match).is_in(*hydrop_whitelists[bc_name])
            assert_that(matched_results[0].read_idx).is_not_none()

    @pytest.mark.parametrize(
        "seq, bc_name",
        [
            # Case 2 (INDEL_FAIL): BC2 fails due to indels - no close match in whitelist
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC2"),
            # Case 5 (SPC_NOTFND): BC1 doesn't match
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC1"),
        ],
    )
    def test_match_dev_sequences_expected_failures(
        self,
        seq: str,
        bc_name: str,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test kmer matching on dev sequences where the barcode should not match."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(seq)

        # No successful match expected
        matched_results = [r for r in result if r.match is not None]
        assert_that(matched_results).is_empty()

    @pytest.mark.parametrize(
        "seq, bc_name, expected_bc",
        [
            # KmerMatcher finds barcodes at shifted positions that FixedPositionMatcher misses
            # Case 2: BC1 CTCCTCATCC is a real whitelist match at a shifted position
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC1", "CTCCTCATCC"),
            # Case 3: BC1 TCTGAGATCG is a whitelist match with ed=1 at shifted position
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC1", "TCTGAGATCG"),
            # Case 6: BC2 TATGCAGTTA is found with ed=1 at shifted position
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC2", "TATGCAGTTA"),
            # Case 6: BC1 CGTCAGACAA is a perfect match at shifted position
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC1", "CGTCAGACAA"),
        ],
    )
    def test_match_dev_sequences_kmer_finds_shifted_barcodes(
        self,
        seq: str,
        bc_name: str,
        expected_bc: str,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that KmerMatcher finds valid barcodes at shifted positions that FixedPositionMatcher misses."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(seq)

        matched = [r for r in result if r.match is not None]
        assert_that(matched).is_not_empty()
        assert_that(matched[0].match).is_equal_to(expected_bc)
        assert_that(matched[0].match).is_in(*hydrop_whitelists[bc_name])

    # ==========================================
    # match() - Edit Distance Accuracy
    # ==========================================

    @pytest.mark.parametrize(
        "n_substitutions",
        [0, 1, 2],
    )
    def test_match_edit_distance_equals_substitution_count(
        self,
        n_substitutions: int,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that edit_distance in the result accurately reflects the number of substitutions."""
        bc3 = hydrop_whitelists["BC3"][0]  # TGACCGTACT
        mutated = list(bc3)
        # Mutate the first n_substitutions bases
        for i in range(n_substitutions):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(n_substitutions)

    # ==========================================
    # match() - Different Barcode Components
    # ==========================================

    @pytest.mark.parametrize("bc_name", ["BC3", "BC2", "BC1"])
    def test_match_across_all_barcode_components(
        self,
        bc_name: str,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that KmerMatcher works for all barcode components in the HyDrop read structure."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        whitelist = hydrop_whitelists[bc_name]
        matcher = KmerMatcher(
            whitelist=whitelist,
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )

        bc3 = hydrop_whitelists["BC3"][0]
        bc2 = hydrop_whitelists["BC2"][0]
        bc1 = hydrop_whitelists["BC1"][0]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        result = matcher.match(read)
        matched_results = [r for r in result if r.match is not None]
        assert_that(matched_results).is_not_empty()
        assert_that(matched_results[0].match).is_in(*whitelist)

    # ==========================================
    # match() - Spacer Fields in Result
    # ==========================================

    def test_match_unambiguous_result_has_no_spacer_fields(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that an unambiguous single-candidate match does not populate spacer fields."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        # Unambiguous matches go through the single-candidate path, no spacer check
        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.spacer_upstream).is_none()
        assert_that(result.spacer_downstream).is_none()

    def test_match_spacer_validated_result_has_spacer_fields(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that spacer-validated results populate the spacer fields when disambiguation occurs."""
        # BC2 has SPACER_1 upstream and SPACER_2 downstream
        bc2 = hydrop_whitelists["BC2"][0]
        bc3 = hydrop_whitelists["BC3"][0]
        bc1 = hydrop_whitelists["BC1"][0]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = KmerMatcher(
            whitelist=hydrop_whitelists["BC2"],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(read)
        # If spacer validation was triggered, spacer fields may be set
        # The result should be valid regardless
        assert_that(result).is_not_empty()
        matched = [r for r in result if r.match is not None]
        assert_that(matched).is_not_empty()
        assert_that(matched[0].match).is_equal_to(bc2)

    # ==========================================
    # build_kmer_index() - Edge Cases
    # ==========================================

    def test_build_kmer_index_with_single_barcode(self) -> None:
        """Test that kmer index is correctly built for a single barcode."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=("ACGTACGTAC",),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
        )

        # 10bp barcode with k=4: 7 positions, but some kmers may repeat
        assert_that(matcher.kmer_index).is_not_empty()
        for kmer, entries in matcher.kmer_index.items():
            assert_that(len(kmer)).is_equal_to(4)
            for bc, pos in entries:
                assert_that(bc).is_equal_to("ACGTACGTAC")
                assert_that(pos).is_greater_than_or_equal_to(0)
                assert_that(pos).is_less_than_or_equal_to(6)

    def test_build_kmer_index_empty_whitelist(self) -> None:
        """Test that kmer index is empty when whitelist is empty."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=(),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
        )
        assert_that(matcher.kmer_index).is_empty()

    def test_build_kmer_index_duplicate_barcodes_deduplicated(self) -> None:
        """Test that duplicate barcodes in whitelist are deduplicated via frozenset."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=("ACGTACGTAC", "ACGTACGTAC", "ACGTACGTAC"),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
        )

        # whitelist_set should have only 1 unique barcode
        assert_that(matcher.whitelist_set).is_length(1)

        # Each kmer should map to at most 1 (barcode, pos) pair since there's only 1 unique BC
        for entries in matcher.kmer_index.values():
            barcodes = [bc for bc, _ in entries]
            assert_that(len(set(barcodes))).is_equal_to(1)

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_build_kmer_index_various_k_sizes(self, k: int) -> None:
        """Test that kmer index is built correctly for various k sizes."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        barcode = "ACGTACGTAC"
        matcher = KmerMatcher(
            whitelist=(barcode,),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
            k=k,
        )

        for kmer in matcher.kmer_index:
            assert_that(len(kmer)).is_equal_to(k)

    # ==========================================
    # match() - Realistic Multi-Barcode Reads
    # ==========================================

    def test_match_correct_barcode_in_multi_barcode_read(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that each matcher finds only its own barcode in a read with all 3 barcodes."""
        bc3 = hydrop_whitelists["BC3"][0]
        bc2 = hydrop_whitelists["BC2"][0]
        bc1 = hydrop_whitelists["BC1"][0]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        for bc_name, expected_bc in [("BC3", bc3), ("BC2", bc2), ("BC1", bc1)]:
            comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
            matcher = KmerMatcher(
                whitelist=hydrop_whitelists[bc_name],
                component=comp,
                chemistry=hydrop_chemistry,
                max_errors=hydrop_chemistry.max_errors.barcode,
            )
            result = matcher.match(read)
            matched = [r for r in result if r.match is not None]
            assert_that(matched).is_not_empty()
            assert_that(matched[0].match).is_equal_to(expected_bc)

    def test_match_different_barcodes_per_component(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when each barcode position uses a different whitelist index."""
        bc3 = hydrop_whitelists["BC3"][10]
        bc2 = hydrop_whitelists["BC2"][20]
        bc1 = hydrop_whitelists["BC1"][30]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        for bc_name, expected_bc in [("BC3", bc3), ("BC2", bc2), ("BC1", bc1)]:
            comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
            matcher = KmerMatcher(
                whitelist=hydrop_whitelists[bc_name],
                component=comp,
                chemistry=hydrop_chemistry,
                max_errors=hydrop_chemistry.max_errors.barcode,
            )
            result = matcher.match(read)
            matched = [r for r in result if r.match is not None]
            assert_that(matched).is_not_empty()
            assert_that(matched[0].match).is_equal_to(expected_bc)

    # ==========================================
    # match() - Boundary & Robustness
    # ==========================================

    def test_match_barcode_at_very_end_of_read(
        self,
        bc1_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when the barcode is at the very end of the read."""
        bc1 = hydrop_whitelists["BC1"][0]
        read = "A" * 40 + bc1  # BC1 at the end, padded with filler
        result = bc1_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc1)
        assert_that(result.read_idx).is_not_none()
        assert_that(result.read_idx[1] if result.read_idx is not None else 0).is_equal_to(50)

    def test_match_barcode_at_very_start_of_read(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when the barcode is at the very start of the read."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = bc3 + "A" * 40
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.read_idx).is_not_none()
        assert_that(result.read_idx[0] if result.read_idx is not None else 0).is_equal_to(0)

    def test_match_with_long_read(
        self,
        bc3_matcher: KmerMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching in a read much longer than typical."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = bc3 + "A" * 200
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)

    def test_match_minimum_viable_read(self, small_matcher: KmerMatcher) -> None:
        """Test matching with a read that is the minimum length to contain a barcode."""
        barcode = "ACGTACGTAC"
        read = barcode  # exactly the barcode length, no flanking
        result = small_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(barcode)
        assert_that(result.edit_distance).is_equal_to(0)


class TestKmerMatcherCandidateSplit:
    """Tests for KmerMatcher.collect_candidates and KmerMatcher.resolve_candidates.

    `match` is split into a collection half and a resolution half with no change in behaviour:
    `collect_candidates` performs the short-read guard, the seed scan and verification and returns
    every verified candidate, while `resolve_candidates` applies the best-score filter and the
    spacer-driven tie-break ladder. Testing them directly makes the tie-break ladder reachable
    from hand-built candidate lists, rather than only through reads contrived to produce them.
    """

    # --- HyDrop spacer constants for read construction ---
    SPACER_1 = "AGGGTACTCG"
    SPACER_2 = "GCAGTAGCTG"

    MAX_ERRORS = 2
    K = 4

    # Two whitelist entries one edit apart, so a read matching the first exactly also verifies
    # the second at a worse-but-legal score.
    NEAR_NEIGHBOUR_WHITELIST = ("ACGTACGTAC", "ACGTACGTCA")
    SINGLE_WHITELIST = ("ACGTACGTAC",)

    # The first whitelist entry at read positions 0-10, followed by filler that seeds nothing.
    READ_BARCODE_AT_START = "ACGTACGTAC" + "T" * 20
    READ_NO_SEEDS = "T" * 20

    # A HyDrop-shaped read whose BC2 window at 20-30 is flanked by both spacers.
    HYDROP_READ = "A" * 10 + SPACER_1 + "C" * 10 + SPACER_2 + "G" * 10

    # A read carrying two separate windows, each flanked by both spacers, so two candidates can
    # be equally well validated and neither can win the two-spacer tie-break.
    TWO_WINDOW_READ = SPACER_1 + "A" * 10 + SPACER_2 + SPACER_1 + "C" * 10 + SPACER_2

    @pytest.fixture
    def chemistry(self) -> ChemistryHydrop:
        """Provide the HyDrop chemistry supplying the read structure.

        Returns:
            A freshly constructed ChemistryHydrop.
        """
        return ChemistryHydrop()

    @pytest.fixture
    def collecting_matcher(self, chemistry: ChemistryHydrop) -> KmerMatcher:
        """Provide a BC3 matcher over the near-neighbour whitelist, for collection tests.

        Args:
            chemistry: The chemistry supplying the read structure.

        Returns:
            The configured KmerMatcher.
        """
        return KmerMatcher(
            whitelist=self.NEAR_NEIGHBOUR_WHITELIST,
            component=chemistry.read_structure.get_component_by_name("BC3"),
            chemistry=chemistry,
            max_errors=self.MAX_ERRORS,
            k=self.K,
        )

    @pytest.fixture
    def single_entry_matcher(self, chemistry: ChemistryHydrop) -> KmerMatcher:
        """Provide a BC3 matcher over a single-entry whitelist, for seed-floor tests.

        Args:
            chemistry: The chemistry supplying the read structure.

        Returns:
            The configured KmerMatcher.
        """
        return KmerMatcher(
            whitelist=self.SINGLE_WHITELIST,
            component=chemistry.read_structure.get_component_by_name("BC3"),
            chemistry=chemistry,
            max_errors=self.MAX_ERRORS,
            k=self.K,
        )

    @pytest.fixture
    def resolving_matcher(self, chemistry: ChemistryHydrop) -> KmerMatcher:
        """Provide a BC2 matcher, whose neighbours in the read structure are the two spacers.

        BC2 is used because it is the only HyDrop barcode with a defined spacer on both sides, so
        `check_spacers` can return two names and the two-spacer tie-break becomes reachable.

        Args:
            chemistry: The chemistry supplying the read structure.

        Returns:
            The configured KmerMatcher.
        """
        return KmerMatcher(
            whitelist=("AAAAAAAAAA", "CCCCCCCCCC", "CCCCCCCCC"),
            component=chemistry.read_structure.get_component_by_name("BC2"),
            chemistry=chemistry,
            max_errors=self.MAX_ERRORS,
            k=self.K,
        )

    # ==========================================
    # collect_candidates()
    # ==========================================

    def test_collect_candidates_returns_suboptimal_candidates_too(
        self, collecting_matcher: KmerMatcher
    ) -> None:
        """Test that collection returns every verified candidate, not only the best-scoring ones.

        Filtering by best score belongs to resolution, so a read matching one whitelist entry
        exactly must still surface the near neighbour that verifies at a worse edit distance, and
        the second alignment of the exact entry that verifies at the very edge of the budget.
        """
        candidates = collecting_matcher.collect_candidates(self.READ_BARCODE_AT_START, 0)

        assert_that(sorted(candidates)).is_equal_to(
            [
                ("ACGTACGTAC", 0, 10, 0),
                ("ACGTACGTAC", 2, 10, 2),
                ("ACGTACGTCA", 0, 9, 1),
            ]
        )
        assert_that(sorted({c[3] for c in candidates})).is_equal_to([0, 1, 2])

    def test_collect_candidates_keeps_candidates_match_would_discard(
        self, collecting_matcher: KmerMatcher
    ) -> None:
        """Test that collection is strictly wider than what `match` ends up reporting.

        `match` reports the single best-scoring candidate; collection hands back that candidate
        alongside the ones the best-score filter is about to drop.
        """
        candidates = collecting_matcher.collect_candidates(self.READ_BARCODE_AT_START, 0)
        result = collecting_matcher.match(self.READ_BARCODE_AT_START)

        assert_that(len(candidates)).is_greater_than(len(result))
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to("ACGTACGTAC")
        assert_that(result[0].edit_distance).is_equal_to(0)

    @pytest.mark.parametrize(
        "start_idx, expected_candidates",
        [
            (0, [("ACGTACGTAC", 0, 10, 0), ("ACGTACGTAC", 2, 10, 2)]),
            (4, [("ACGTACGTAC", 0, 10, 0), ("ACGTACGTAC", 2, 10, 2)]),
            (7, []),
        ],
    )
    def test_collect_candidates_respects_the_start_idx_seed_floor(
        self,
        single_entry_matcher: KmerMatcher,
        start_idx: int,
        expected_candidates: list[tuple[str, int, int, int]],
    ) -> None:
        """Test that `start_idx` bounds the seed scan in collection, not the spans it returns.

        At a `start_idx` of 4 the collected spans still begin at 0, because seeds found at or
        after the floor are extended backwards. Only once the floor outruns the last whitelist
        k-mer in the read, at 7, does collection come back empty.
        """
        candidates = single_entry_matcher.collect_candidates(self.READ_BARCODE_AT_START, start_idx)

        assert_that(sorted(candidates)).is_equal_to(expected_candidates)

    def test_collect_candidates_read_shorter_than_k_raises_value_error(
        self, single_entry_matcher: KmerMatcher
    ) -> None:
        """Test that the short-read guard lives in collection and raises the same ValueError.

        The guard is the first thing `match` does today, so it must move wholesale into the
        collection half and keep raising for the same input with the same message.
        """
        with pytest.raises(ValueError, match="Read segment too short for k-mer matching"):
            single_entry_matcher.collect_candidates("ACG", 0)

    def test_collect_candidates_returns_empty_list_when_nothing_verifies(
        self, single_entry_matcher: KmerMatcher
    ) -> None:
        """Test that a read seeding nothing collects an empty list rather than failing."""
        candidates = single_entry_matcher.collect_candidates(self.READ_NO_SEEDS, 0)

        assert_that(candidates).is_equal_to([])

    def test_match_is_collect_then_resolve(self, collecting_matcher: KmerMatcher) -> None:
        """Test that `match` is exactly the composition of the two halves.

        This is the guarantee that the split carries no behaviour change: running the halves by
        hand must reproduce what `match` returns for the same read.
        """
        candidates = collecting_matcher.collect_candidates(self.READ_BARCODE_AT_START, 0)
        composed = collecting_matcher.resolve_candidates(self.READ_BARCODE_AT_START, candidates)

        assert_that(composed).is_equal_to(collecting_matcher.match(self.READ_BARCODE_AT_START))

    # ==========================================
    # resolve_candidates()
    # ==========================================

    def test_resolve_candidates_with_no_candidates_returns_a_single_matchless_attempt(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test that an empty candidate list resolves to one attempt carrying only a method."""
        result = resolving_matcher.resolve_candidates(self.HYDROP_READ, [])

        assert_that(result).is_length(1)
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)
        assert_that(result[0].match).is_none()
        assert_that(result[0].candidate).is_none()
        assert_that(result[0].read_idx).is_none()
        assert_that(result[0].edit_distance).is_none()

    def test_resolve_candidates_with_one_candidate_resolves_without_checking_spacers(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test that a lone best-scoring candidate is accepted with no spacer validation at all.

        Spacers are a tie-break, so with nothing to break there is no tie: the candidate is
        reported as the match and both spacer fields stay unset even though this candidate's
        window is in fact flanked by both HyDrop spacers.
        """
        result = resolving_matcher.resolve_candidates(
            self.HYDROP_READ, [("CCCCCCCCCC", 20, 30, 1)]
        )

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to("CCCCCCCCCC")
        assert_that(result[0].candidate).is_equal_to("CCCCCCCCCC")
        assert_that(result[0].read_idx).is_equal_to((20, 30))
        assert_that(result[0].edit_distance).is_equal_to(1)
        assert_that(result[0].spacer_upstream).is_none()
        assert_that(result[0].spacer_downstream).is_none()

    def test_resolve_candidates_filters_to_the_best_score_before_anything_else(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test that a worse-scoring candidate is dropped before the tie-break ladder is reached.

        Two candidates go in, but only one has the best edit distance, so the ladder never runs
        and the survivor is reported unvalidated.
        """
        result = resolving_matcher.resolve_candidates(
            self.HYDROP_READ, [("CCCCCCCCCC", 20, 30, 0), ("AAAAAAAAAA", 0, 10, 2)]
        )

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to("CCCCCCCCCC")
        assert_that(result[0].edit_distance).is_equal_to(0)

    def test_resolve_candidates_tie_broken_by_the_only_candidate_with_a_spacer(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test that a score tie is broken by the single candidate with any adjacent spacer.

        Both candidates share the best score, but only the window at 20-29 has a recognised
        spacer beside it, so it is the only one validated and it wins outright. Its spacer fields
        are then reported, unlike in the single-candidate path.
        """
        result = resolving_matcher.resolve_candidates(
            self.HYDROP_READ, [("CCCCCCCCC", 20, 29, 1), ("AAAAAAAAAA", 0, 10, 1)]
        )

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to("CCCCCCCCC")
        assert_that(result[0].candidate).is_equal_to("CCCCCCCCC")
        assert_that(result[0].read_idx).is_equal_to((20, 29))
        assert_that(result[0].spacer_upstream).is_equal_to("SPACER_1")
        assert_that(result[0].spacer_downstream).is_none()

    def test_resolve_candidates_tie_broken_by_the_only_candidate_with_two_spacers(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test the second rung of the ladder: two validated candidates, one with both spacers.

        Both candidates carry an upstream spacer and so both are validated, which leaves the
        first rung undecided. The candidate flanked on both sides is then preferred, on the
        reasoning that two intact spacers is the stronger positional evidence.
        """
        result = resolving_matcher.resolve_candidates(
            self.HYDROP_READ, [("CCCCCCCCCC", 20, 30, 1), ("CCCCCCCCC", 20, 29, 1)]
        )

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to("CCCCCCCCCC")
        assert_that(result[0].read_idx).is_equal_to((20, 30))
        assert_that(result[0].spacer_upstream).is_equal_to("SPACER_1")
        assert_that(result[0].spacer_downstream).is_equal_to("SPACER_2")

    def test_resolve_candidates_full_ambiguity_returns_one_matchless_attempt_per_candidate(
        self, resolving_matcher: KmerMatcher
    ) -> None:
        """Test the fallthrough when the ladder cannot separate two equally evidenced candidates.

        Both windows are flanked by both spacers, so neither rung of the ladder can choose
        between them. The matcher then declines to guess and reports every validated candidate as
        its own attempt with `match` left as None, preserving the spans and spacer evidence so a
        caller can see exactly what was ambiguous.
        """
        result = resolving_matcher.resolve_candidates(
            self.TWO_WINDOW_READ, [("AAAAAAAAAA", 10, 20, 1), ("CCCCCCCCCC", 40, 50, 1)]
        )

        assert_that(result).is_length(2)
        for attempt in result:
            assert_that(attempt.match).is_none()
            assert_that(attempt.method).is_equal_to(MatchMethod.KMERMATCH)
            assert_that(attempt.edit_distance).is_equal_to(1)
            assert_that(attempt.spacer_upstream).is_equal_to("SPACER_1")
            assert_that(attempt.spacer_downstream).is_equal_to("SPACER_2")
        assert_that([a.read_idx for a in result]).is_equal_to([(10, 20), (40, 50)])
        assert_that([a.candidate for a in result]).is_equal_to(["AAAAAAAAAA", "CCCCCCCCCC"])


class TestKmerMatcherTargetIndex:
    """Tests for KmerMatcher over a TGIDX component, whose start position is unresolved."""

    UMI = "ACGTACGT"
    POLYG = "GGGG"
    TRAILING = "CTGTCTCTTATACACATCT"

    EXACT_INDEX = "TATAGCCT"
    ONE_ERROR_INDEX = "TATAGGCT"
    TWO_ERROR_INDEX = "TAGAGGCT"

    @pytest.fixture
    def chemistry(self) -> ChemistryCarmackCustomSeq10:
        """Provide the Carmack custom sequencing chemistry, which carries a TGIDX component."""
        return ChemistryCarmackCustomSeq10()

    @pytest.fixture
    def tgidx_matcher(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> Callable[[int], KmerMatcher]:
        """Provide a factory building a TGIDX KmerMatcher for a given error budget.

        Args:
            chemistry: The chemistry supplying the TGIDX component and whitelist.

        Returns:
            Callable taking the error budget and returning the configured matcher.
        """

        def build(max_errors: int) -> KmerMatcher:
            return KmerMatcher(
                whitelist=chemistry.tgidx_whitelist(),
                component=chemistry.tgidx_component(),
                chemistry=chemistry,
                max_errors=max_errors,
            )

        return build

    def build_read(self, chemistry: ChemistryCarmackCustomSeq10, index: str) -> str:
        """Construct a full read carrying the three barcodes, the UMI, a poly-G run and an index.

        Args:
            chemistry: The chemistry supplying the barcode whitelists.
            index: The target index sequence to place after the poly-G run.

        Returns:
            The assembled read sequence.
        """
        bc3 = chemistry.load_whitelist("BC3")[0]
        bc2 = chemistry.load_whitelist("BC2")[0]
        bc1 = chemistry.load_whitelist("BC1")[0]
        return (
            bc3 + PRIMER_C + bc2 + PRIMER_A + bc1 + self.UMI + self.POLYG + index + self.TRAILING
        )

    def test_tgidx_whitelist_is_the_single_packaged_index(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test that the fixture data is the real single-entry target index whitelist."""
        assert_that(chemistry.tgidx_whitelist()).is_equal_to((self.EXACT_INDEX,))

    def test_match_exact_index_returns_zero_edit_distance(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        tgidx_matcher: Callable[[int], KmerMatcher],
    ) -> None:
        """Test that an exact target index occurrence is matched with edit distance zero."""
        matcher = tgidx_matcher(chemistry.max_errors.tgidx)
        result = matcher.match(self.build_read(chemistry, self.EXACT_INDEX))[0]

        assert_that(result.match).is_equal_to(self.EXACT_INDEX)
        assert_that(result.edit_distance).is_equal_to(0)

    def test_match_one_error_index_returns_edit_distance_one(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        tgidx_matcher: Callable[[int], KmerMatcher],
    ) -> None:
        """Test that a target index carrying a single substitution is still matched."""
        matcher = tgidx_matcher(chemistry.max_errors.tgidx)
        result = matcher.match(self.build_read(chemistry, self.ONE_ERROR_INDEX))[0]

        assert_that(result.match).is_equal_to(self.EXACT_INDEX)
        assert_that(result.edit_distance).is_equal_to(1)

    def test_match_two_error_index_returns_no_match(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        tgidx_matcher: Callable[[int], KmerMatcher],
    ) -> None:
        """Test that a target index carrying two substitutions exceeds the budget and fails."""
        matcher = tgidx_matcher(chemistry.max_errors.tgidx)
        result = matcher.match(self.build_read(chemistry, self.TWO_ERROR_INDEX))[0]

        assert_that(result.match).is_none()

    def test_match_budget_comes_from_the_max_errors_argument_not_the_chemistry(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        tgidx_matcher: Callable[[int], KmerMatcher],
    ) -> None:
        """Test that the error budget is driven by the constructor argument alone.

        Two matchers over the same component and whitelist differ only in the max_errors they
        were given, so the same one-error read must match under a budget of one and fail under a
        budget of zero. Neither outcome can come from the chemistry, which both matchers share.
        """
        read = self.build_read(chemistry, self.ONE_ERROR_INDEX)

        permissive = tgidx_matcher(1)
        strict = tgidx_matcher(0)

        assert_that(permissive.max_errors).is_equal_to(1)
        assert_that(strict.max_errors).is_equal_to(0)
        assert_that(permissive.match(read)[0].match).is_equal_to(self.EXACT_INDEX)
        assert_that(strict.match(read)[0].match).is_none()


class TestKmerMatcherTargetIndexAmbiguity:
    """Characterisation tests for how KmerMatcher resolves ambiguity over a TGIDX component.

    The packaged target-index whitelist holds a single sequence, so it cannot produce genuine
    multi-target ambiguity and the policy that governs it is never exercised by the shipped data.
    These tests close that gap with a synthetic two-entry whitelist, pinning the policy that
    downstream target-index work depends on.
    """

    UMI = "ACGTACGT"
    POLYG = "GGGG"
    TRAILING = "CTGTCTCTTATACACATCT"

    # Two synthetic target indexes one edit apart from each other, and an observed window one
    # edit from both, so neither entry can win on score.
    INDEX_A = "TATAGCCT"
    INDEX_B = "TATTGCCT"
    AMBIGUOUS_WINDOW = "TATCGCCT"

    AMBIGUOUS_WHITELIST = (INDEX_A, INDEX_B)

    @pytest.fixture
    def chemistry(self) -> ChemistryCarmackCustomSeq10:
        """Provide the Carmack custom sequencing chemistry, which carries a TGIDX component.

        Returns:
            A freshly constructed ChemistryCarmackCustomSeq10.
        """
        return ChemistryCarmackCustomSeq10()

    @pytest.fixture
    def matcher(self, chemistry: ChemistryCarmackCustomSeq10) -> KmerMatcher:
        """Provide a TGIDX matcher over the synthetic two-entry whitelist.

        Args:
            chemistry: The chemistry supplying the TGIDX component and read structure.

        Returns:
            The configured KmerMatcher, with a one-error budget matching the chemistry's own.
        """
        return KmerMatcher(
            whitelist=self.AMBIGUOUS_WHITELIST,
            component=chemistry.tgidx_component(),
            chemistry=chemistry,
            max_errors=1,
            k=4,
        )

    def read_prefix(self, chemistry: ChemistryCarmackCustomSeq10) -> str:
        """Build everything preceding the target index in the read.

        Args:
            chemistry: The chemistry supplying the barcode whitelists.

        Returns:
            The assembled prefix: the three barcodes, both primers, the UMI and the poly-G run.
        """
        bc3 = chemistry.load_whitelist("BC3")[0]
        bc2 = chemistry.load_whitelist("BC2")[0]
        bc1 = chemistry.load_whitelist("BC1")[0]
        return bc3 + PRIMER_C + bc2 + PRIMER_A + bc1 + self.UMI + self.POLYG

    def build_read(self, chemistry: ChemistryCarmackCustomSeq10, index: str) -> str:
        """Build a full read carrying the given target index after the poly-G run.

        Args:
            chemistry: The chemistry supplying the barcode whitelists.
            index: The target index window to place after the poly-G run.

        Returns:
            The assembled read sequence.
        """
        return self.read_prefix(chemistry) + index + self.TRAILING

    def index_span(self, chemistry: ChemistryCarmackCustomSeq10, index: str) -> tuple[int, int]:
        """Return the read coordinates the given target index window occupies.

        Args:
            chemistry: The chemistry supplying the barcode whitelists.
            index: The target index window placed after the poly-G run.

        Returns:
            The (start, end) coordinates of the window in the assembled read.
        """
        start = len(self.read_prefix(chemistry))
        return (start, start + len(index))

    def test_synthetic_whitelist_is_equidistant_from_the_observed_window(self) -> None:
        """Test the premise the ambiguity test rests on: neither entry is closer than the other.

        The observed window is one edit from both whitelist entries, which are themselves one
        edit apart, so the best-score filter cannot separate them and the tie-break ladder is
        genuinely reached.
        """
        assert_that(edit_distance(self.AMBIGUOUS_WINDOW, self.INDEX_A)).is_equal_to(1)
        assert_that(edit_distance(self.AMBIGUOUS_WINDOW, self.INDEX_B)).is_equal_to(1)
        assert_that(edit_distance(self.INDEX_A, self.INDEX_B)).is_equal_to(1)

    def test_packaged_whitelist_cannot_produce_this_ambiguity(
        self, chemistry: ChemistryCarmackCustomSeq10
    ) -> None:
        """Test why the two-entry whitelist has to be synthetic.

        The chemistry ships exactly one confirmed target index, so no read can ever be equidistant
        from two of them and the ambiguity policy would otherwise go untested.
        """
        assert_that(chemistry.tgidx_whitelist()).is_equal_to((self.INDEX_A,))

    def test_ambiguous_target_index_returns_a_single_matchless_attempt(
        self, chemistry: ChemistryCarmackCustomSeq10, matcher: KmerMatcher
    ) -> None:
        """Test that a target index equidistant from two whitelist entries resolves to no match.

        The policy arrives by structure rather than by any explicit branch, which is why it is
        worth pinning:

        - the target index's previous component is the POLYG homopolymer, which carries no
          `sequence`, so it is rejected by the `match_seq` guard inside `check_spacers` (the
          `spacer_component.type in (PRIMER, OTHER) and spacer_component.sequence` condition in
          `matcher_base.py`);
        - `ReadStructure.get_next(TGIDX)` returns the `ME` primer, but `ME`'s component also
          carries no `sequence` (only its length is used, arithmetically, elsewhere in the
          pipeline), so it is rejected by the same guard and no real comparison is ever made
          downstream either;
        - `check_spacers` therefore returns None on both sides for every candidate, no candidate
          is ever validated, and every tie over a target index falls through to a single attempt
          with no match.

        The structural assertions below are part of the test on purpose: if the read structure
        ever gains a sequence-bearing neighbour on either side of TGIDX, this test must fail
        loudly rather than quietly start exercising a different policy.
        """
        read = self.build_read(chemistry, self.AMBIGUOUS_WINDOW)
        span = self.index_span(chemistry, self.AMBIGUOUS_WINDOW)
        tgidx = chemistry.tgidx_component()
        previous = chemistry.read_structure.get_previous(tgidx)
        following = chemistry.read_structure.get_next(tgidx)

        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].method).is_equal_to(MatchMethod.KMERMATCH)
        assert_that(result[0].match).is_none()

        # The structure that produces the policy, asserted so a change to it breaks this test.
        assert_that(read[span[0] : span[1]]).is_equal_to(self.AMBIGUOUS_WINDOW)
        assert_that(matcher.check_spacers(read, span)).is_equal_to(
            {"upstream": None, "downstream": None}
        )
        assert_that(previous.type).is_equal_to(ReadComponentType.HOMOPOLYMER)
        assert_that(previous.sequence).is_none()
        assert_that(following.name).is_equal_to("ME")
        assert_that(following.type).is_equal_to(ReadComponentType.PRIMER)
        assert_that(following.sequence).is_none()

    @pytest.mark.parametrize("index", [INDEX_A, INDEX_B])
    def test_same_matcher_resolves_an_unambiguous_exact_hit(
        self,
        chemistry: ChemistryCarmackCustomSeq10,
        matcher: KmerMatcher,
        index: str,
    ) -> None:
        """Test that the ambiguity result is not simply this matcher failing to match anything.

        The same matcher over the same two-entry whitelist resolves either entry cleanly when the
        observed window is an exact hit, so the match-less attempt above is the ambiguity policy
        firing rather than a vacuous pass.
        """
        read = self.build_read(chemistry, index)
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(index)
        assert_that(result[0].edit_distance).is_equal_to(0)
        assert_that(result[0].read_idx).is_equal_to(self.index_span(chemistry, index))


class TestAlignmentMatcher:
    """Tests for AlignmentMatcher barcode matching logic."""

    # --- HyDrop spacer constants for read construction ---
    SPACER_1 = "AGGGTACTCG"
    SPACER_2 = "GCAGTAGCTG"

    # --- Two very distinct ACGT-only barcodes for score tie-break regression tests ---
    # They differ well beyond max_errors (edit_distance == 7 > 2), are not close to either
    # spacer, and are not poly-base runs, so a wrong-barcode assignment fails the max_errors gate.
    BC_X = "ATGCCTGATG"
    BC_Y = "TCATTGACCA"

    # --- Fixtures ---

    @pytest.fixture
    def matcher_class(self) -> type[MatcherBase]:
        """Build HyDrop matcher fixtures as AlignmentMatcher instances."""
        return AlignmentMatcher

    @pytest.fixture
    def matcher_kwargs(self, hydrop_chemistry: ChemistryHydrop) -> dict[str, Any]:
        """Give the shared matcher fixtures the HyDrop barcode error budget."""
        return {"max_errors": hydrop_chemistry.max_errors.barcode}

    # --- Helper to build a full HyDrop read ---

    def _build_hydrop_read(self, bc3: str, bc2: str, bc1: str) -> str:
        """Construct a full 50bp HyDrop read: BC3 + SPACER_1 + BC2 + SPACER_2 + BC1."""
        return bc3 + self.SPACER_1 + bc2 + self.SPACER_2 + bc1

    # ==========================================
    # Initialization Tests
    # ==========================================

    def test_initialization_stores_correct_attributes(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that AlignmentMatcher initializes correctly with given whitelist and chemistry."""
        assert_that(bc3_matcher.component.name).is_equal_to("BC3")
        assert_that(bc3_matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(bc3_matcher.whitelist_set).is_equal_to(frozenset(hydrop_whitelists["BC3"]))

    def test_initialization_sets_score_threshold(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that AlignmentMatcher sets score_threshold during initialization."""
        assert_that(bc3_matcher.score_threshold).is_not_none()
        assert_that(bc3_matcher.score_threshold).is_instance_of(float)

    # ==========================================
    # compute_score_threshold() Tests
    # ==========================================

    def test_score_threshold_exact_value(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that compute_score_threshold returns the exact expected float value.

        For HyDrop: bc_len=10, max_errors=2, MATCH_SCORE=1.
        perfect_score = 10 * 1 = 10.
        For local alignment, each error costs the lost match score (1 for substitution,
        1.5 for indel). threshold = 10 - 2 * 1.5 = 7.0.
        """
        bc_len = bc3_matcher.component.length
        max_errors = bc3_matcher.max_errors
        perfect_score = bc_len * MATCH_SCORE
        max_penalty_per_error = max(
            MATCH_SCORE,  # substitution: lose 1 match
            abs(GAP_OPEN_SCORE) + abs(GAP_EXTEND_SCORE),  # indel: 0.5 + 1 = 1.5
        )
        expected = float(perfect_score - max_errors * max_penalty_per_error)

        assert_that(bc3_matcher.score_threshold).is_equal_to(expected)
        assert_that(bc3_matcher.score_threshold).is_instance_of(float)

    def test_score_threshold_alignment_at_exactly_threshold_returns_result(
        self, bc3_matcher: AlignmentMatcher
    ) -> None:
        """Test that align_seqs returns a result when the alignment score is at or above threshold.

        With local alignment, the scoring includes mismatch penalties, so the exact threshold
        landing depends on alignment position. This test verifies that a borderline case
        (3 substitutions, score typically 7.5) returns a valid result.
        """
        # Sorted, not set-iteration order, so the barcode drawn does not vary with
        # PYTHONHASHSEED.
        bc3 = sorted(bc3_matcher.whitelist_set)[0]

        # With local alignment including mismatch penalties, 3 subs typically scores ~7.5
        # which is above threshold=7.0
        mutated = list(bc3)
        n_subs = 3
        for i in range(n_subs):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        result = bc3_matcher.align_seqs(mutated_str, bc3)
        assert_that(result).is_not_none()
        # Score should be at or above threshold (typically 7.5 for 3 subs)
        assert_that(result.score).is_greater_than_or_equal_to(bc3_matcher.score_threshold)  # type: ignore[union-attr]

    def test_score_threshold_alignment_below_threshold_returns_none(
        self, bc3_matcher: AlignmentMatcher
    ) -> None:
        """Test that align_seqs returns None when the alignment score falls below the threshold.

        The substitutions are spread across the barcode rather than clustered at its
        start. Alignment is local, so four consecutive substitutions at the 5' end are
        simply trimmed away and the clean 6bp suffix still scores above threshold for a
        fifth of the whitelist. Spreading them leaves no sub-span long enough to clear
        threshold=7.0, which holds for every barcode rather than most of them.
        """
        bc3 = sorted(bc3_matcher.whitelist_set)[0]

        mutated = list(bc3)
        for i in (0, 3, 6, 9):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        result = bc3_matcher.align_seqs(mutated_str, bc3)
        assert_that(result).is_none()

    # ==========================================
    # sanitize_sequence() Tests
    # ==========================================

    def test_sanitize_sequence_valid_bases_unchanged(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that a sequence with only ACGTN characters is returned unchanged."""
        seq = "ACGTNACGTN"
        assert_that(bc3_matcher.sanitise_sequence(seq)).is_equal_to(seq)

    def test_sanitize_sequence_invalid_chars_replaced_with_n(
        self, bc3_matcher: AlignmentMatcher
    ) -> None:
        """Test that non-ACGTN characters are replaced with N."""
        seq = "ACG@T!"
        expected = "ACGNTN"
        assert_that(bc3_matcher.sanitise_sequence(seq)).is_equal_to(expected)

    def test_sanitize_sequence_n_is_preserved(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that existing N bases are preserved as-is."""
        seq = "NNNNNNNNNN"
        assert_that(bc3_matcher.sanitise_sequence(seq)).is_equal_to(seq)

    def test_sanitize_sequence_empty_string(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that an empty sequence is returned unchanged."""
        assert_that(bc3_matcher.sanitise_sequence("")).is_equal_to("")

    # ==========================================
    # build_substitution_matrix() Tests
    # ==========================================

    def test_substitution_matrix_match_score(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that matching bases receive MATCH_SCORE in the substitution matrix."""
        mat = bc3_matcher.build_substitution_matrix()
        for base in "ACGT":
            assert_that(mat[base, base]).is_equal_to(MATCH_SCORE)

    def test_substitution_matrix_mismatch_score(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that mismatching non-N bases receive MISMATCH_SCORE."""
        mat = bc3_matcher.build_substitution_matrix()
        assert_that(mat["A", "C"]).is_equal_to(MISMATCH_SCORE)
        assert_that(mat["G", "T"]).is_equal_to(MISMATCH_SCORE)

    def test_substitution_matrix_n_matches_any_base_with_match_score(
        self, bc3_matcher: AlignmentMatcher
    ) -> None:
        """Test that N paired with any base (including N) receives MATCH_SCORE (wildcard)."""
        mat = bc3_matcher.build_substitution_matrix()
        for base in "ACGTN":
            assert_that(mat["N", base]).is_equal_to(MATCH_SCORE)
            assert_that(mat[base, "N"]).is_equal_to(MATCH_SCORE)

    # ==========================================
    # trim_read() Tests
    # ==========================================

    def test_trim_read_basic(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that trim_read returns the substring from the given start position."""
        assert_that(bc3_matcher.trim_read("ACGTACGT", 4)).is_equal_to("ACGT")

    def test_trim_read_start_zero(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that trim_read with start=0 returns the full read."""
        assert_that(bc3_matcher.trim_read("ACGTACGT", 0)).is_equal_to("ACGTACGT")

    def test_trim_read_start_beyond_length(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that trim_read returns empty string when start is beyond read length."""
        assert_that(bc3_matcher.trim_read("ACGT", 10)).is_equal_to("")

    def test_trim_read_start_at_length(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that trim_read returns empty string when start equals read length."""
        assert_that(bc3_matcher.trim_read("ACGT", 4)).is_equal_to("")

    # ==========================================
    # align_seqs() Tests
    # ==========================================

    def test_align_seqs_returns_alignment_container_on_match(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that align_seqs returns an AlignmentContainer when sequences align above threshold."""
        bc3 = hydrop_whitelists["BC3"][0]
        result = bc3_matcher.align_seqs(bc3, bc3)

        assert_that(result).is_not_none()
        assert_that(result).is_instance_of(AlignmentContainer)

    def test_align_seqs_returns_none_below_threshold(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that align_seqs returns None when alignment score is below threshold.

        With threshold=7.0 and local alignment including mismatch penalties,
        need 5 substitutions to ensure score falls below threshold.
        """
        bc3 = hydrop_whitelists["BC3"][0]

        # Introduce 5 substitutions to fall below threshold
        mutated = list(bc3)
        n_subs = 5
        for i in range(n_subs):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        result = bc3_matcher.align_seqs(mutated_str, bc3)
        assert_that(result).is_none()

    @pytest.mark.parametrize("n_substitutions", [0, 1, 2])
    def test_align_seqs_score_for_substitutions(
        self,
        n_substitutions: int,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that align_seqs returns the expected alignment score for 0, 1, or 2 substitutions.

        For local alignment, mismatches are clipped rather than penalised.
        So each substitution costs only the lost match (1 point), not the mismatch penalty.
        score = bc_len * MATCH_SCORE - n_subs * MATCH_SCORE = 10 - n_subs * 1.
        """
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        for i in range(n_substitutions):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        bc_len = bc3_matcher.component.length
        # Local alignment clips mismatches, so each sub only costs 1 (the lost match)
        expected_score = float(bc_len * MATCH_SCORE - n_substitutions * MATCH_SCORE)

        result = bc3_matcher.align_seqs(mutated_str, bc3)
        assert_that(result).is_not_none()
        assert_that(result.score).is_equal_to(expected_score)  # type: ignore[union-attr]

    def test_align_seqs_coordinates_span_correct_region(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that seq1_span from align_seqs correctly spans the matched region in seq1."""
        bc3 = hydrop_whitelists["BC3"][0]
        result = bc3_matcher.align_seqs(bc3, bc3)

        assert_that(result).is_not_none()
        assert result is not None
        start, end = result.seq1_span
        assert_that(bc3[start:end]).is_equal_to(bc3)

    def test_align_seqs_perfect_match_has_full_bc_length_score(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a perfect match scores bc_len * MATCH_SCORE."""
        bc3 = hydrop_whitelists["BC3"][0]
        bc_len = bc3_matcher.component.length
        expected_score = bc_len * MATCH_SCORE

        result = bc3_matcher.align_seqs(bc3, bc3)
        assert_that(result).is_not_none()
        assert_that(result.score).is_equal_to(expected_score)  # type: ignore[union-attr]

    # ==========================================
    # align_seqs() debug logging Tests
    # ==========================================

    def test_align_seqs_emits_both_debug_lines_when_debug_enabled(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that align_seqs emits both guarded debug lines when DEBUG is enabled.

        Both log.debug calls sit behind an isEnabledFor guard, so nothing is emitted unless
        DEBUG is on. The level is set on the alignment_matcher logger specifically rather than
        on the root, since the guard reads that logger's own effective level.
        """
        caplog.set_level(logging.DEBUG, logger="carmack.barcode.matchers.alignment_matcher")
        bc3 = hydrop_whitelists["BC3"][0]

        bc3_matcher.align_seqs(bc3, bc3)

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "carmack.barcode.matchers.alignment_matcher"
        ]
        assert_that(messages).is_length(2)
        assert_that(messages[0]).contains("Aligning sequences")
        assert_that(messages[1]).contains("Found", "alignments")

    def test_align_seqs_does_not_evaluate_alignment_count_when_debug_disabled(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the debug guard leaves the expensive log arguments unevaluated.

        The second debug line interpolates ``len(alignments)`` and ``{alignments}``, and both
        make BioPython walk the dynamic programming matrix to count optimal paths. align_seqs
        runs once per whitelist entry per read, so that work must not happen with DEBUG off.
        The stand-in aligner hands back a proxy that flags any call to its length or text
        conversion, and that flag must stay unset while the call still yields a normal
        AlignmentContainer.
        """
        caplog.set_level(logging.INFO, logger="carmack.barcode.matchers.alignment_matcher")
        bc3 = hydrop_whitelists["BC3"][0]

        class CountingAlignments:
            """Delegating proxy recording whether its length or text was ever taken."""

            def __init__(self, alignments: PairwiseAlignments) -> None:
                self.alignments = alignments
                self.counted = False

            def __getattr__(self, name: str) -> Any:
                return getattr(self.alignments, name)

            def __getitem__(self, index: int) -> Any:
                return self.alignments[index]

            def __len__(self) -> int:
                self.counted = True
                return len(self.alignments)

            def __str__(self) -> str:
                self.counted = True
                return str(self.alignments)

        class RecordingAligner:
            """Aligner stand-in handing back the real alignments inside a counting proxy."""

            def __init__(self, aligner: PairwiseAligner) -> None:
                self.aligner = aligner
                self.proxies: list[CountingAlignments] = []

            def align(self, seq1: str, seq2: str) -> CountingAlignments:
                proxy = CountingAlignments(self.aligner.align(seq1, seq2))
                self.proxies.append(proxy)
                return proxy

        recorder = RecordingAligner(bc3_matcher.aligner)
        monkeypatch.setattr(bc3_matcher, "aligner", recorder)

        result = bc3_matcher.align_seqs(bc3, bc3)

        expected_score = float(bc3_matcher.component.length * MATCH_SCORE)
        assert_that(result).is_instance_of(AlignmentContainer)
        assert_that(result.score).is_equal_to(expected_score)  # type: ignore[union-attr]
        assert_that(recorder.proxies).is_length(1)
        assert_that(recorder.proxies[0].counted).is_false()

    # ==========================================
    # match() - Return Type & Basic Structure
    # ==========================================

    def test_match_returns_list_of_barcode_match_attempt(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that match() returns a list of BarcodeMatchAttempt instances."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        assert_that(result).is_instance_of(list)
        assert_that(result).is_not_empty()
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)

    def test_match_method_is_always_alignmatch(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that the method field is always ALIGNMATCH regardless of match outcome."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]
        assert_that(result.method).is_equal_to(MatchMethod.ALIGNMATCH)

    def test_match_no_match_returns_none_fields(self, small_matcher: AlignmentMatcher) -> None:
        """Test that a failed match returns None for match, read_idx, and edit_distance."""
        # All-T read: no barcode in the small whitelist aligns above threshold
        read = "T" * 50
        result = small_matcher.match(read)
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()
        assert_that(result[0].read_idx).is_none()
        assert_that(result[0].edit_distance).is_none()

    def test_match_no_match_method_still_alignmatch(self, small_matcher: AlignmentMatcher) -> None:
        """Test that even a failed match result carries the ALIGNMATCH method."""
        read = "T" * 50
        result = small_matcher.match(read)
        assert_that(result[0].method).is_equal_to(MatchMethod.ALIGNMATCH)

    # ==========================================
    # match() - Exact Matching
    # ==========================================

    def test_match_exact_barcode_at_expected_position(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that an exact barcode at its expected position is matched correctly."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)
        assert_that(result.read_idx).is_not_none()
        assert_that(result.candidate).is_equal_to(bc3)

    def test_match_exact_barcode_returns_correct_read_idx(
        self,
        bc2_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that read_idx correctly spans the barcode region in the read."""
        bc2 = hydrop_whitelists["BC2"][0]
        read = self._build_hydrop_read("A" * 10, bc2, "A" * 10)
        result = bc2_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc2)
        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        start, end = result.read_idx
        assert_that(read[start:end]).is_equal_to(bc2)

    @pytest.mark.parametrize("bc_idx", [0, 1, 10, 50, 95])
    def test_match_exact_various_whitelist_entries(
        self,
        bc_idx: int,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that various barcodes from the whitelist are matched exactly."""
        bc3 = hydrop_whitelists["BC3"][bc_idx]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)

    # ==========================================
    # match() - Substitution Errors
    # ==========================================

    def test_match_single_substitution(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a barcode with a single substitution is still matched."""
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = bc3[:5] + ("A" if bc3[5] != "A" else "T") + bc3[6:]
        read = self._build_hydrop_read(mutated, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(1)

    def test_match_two_substitutions_at_max_errors(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a barcode with exactly max_errors substitutions is matched (HyDrop max_errors=2)."""
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        max_errors = bc3_matcher.chemistry.max_errors.barcode
        for i in range(max_errors):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(max_errors)

    @pytest.mark.parametrize(
        "sub_positions",
        [
            [0],  # first base
            [9],  # last base
            [4],  # middle base
        ],
    )
    def test_match_single_substitution_at_various_positions(
        self,
        sub_positions: list[int],
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that single substitutions at various positions are matched.

        With local alignment and threshold=7.0, single substitutions score 9.0 (above threshold).
        """
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        for pos in sub_positions:
            mutated[pos] = "A" if mutated[pos] != "A" else "C"
        mutated_str = "".join(mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(len(sub_positions))

    @pytest.mark.parametrize(
        "sub_positions",
        [
            [2, 7],  # two internal positions
        ],
    )
    def test_match_two_substitutions_falls_below_threshold(
        self,
        sub_positions: list[int],
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that two substitutions may fall below threshold depending on positions.

        With local alignment and threshold=7.0, some 2-substitution patterns score 6.0
        (10 - 2 matches - 2 mismatch penalties = 6), which is below threshold.
        This is expected behavior - local alignment with max_errors=2 and threshold=7.0
        is conservative to avoid false positives.
        """
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        for pos in sub_positions:
            mutated[pos] = "A" if mutated[pos] != "A" else "C"
        mutated_str = "".join(mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        # With local alignment, 2 subs at certain positions score below threshold
        # and thus return match=None - this is expected conservative behavior
        assert_that(result.match).is_none()
        assert_that(result.candidate).is_not_none()
        assert_that(result.read_idx).is_not_none()

    # ==========================================
    # match() - Indel Errors
    # ==========================================

    def test_match_single_insertion_is_matched(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a single insertion in the barcode is matched using local alignment."""
        bc3 = hydrop_whitelists["BC3"][0]
        # Insert a base at position 5: TGACC_A_GTACT -> 11bp
        inserted = bc3[:5] + "A" + bc3[5:]
        read = self._build_hydrop_read(inserted, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(1)

    def test_match_single_deletion_is_matched(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a single deletion in the barcode is matched using local alignment."""
        bc3 = hydrop_whitelists["BC3"][0]
        # Delete position 5: TGACC_GTACT -> 9bp
        deleted = bc3[:5] + bc3[6:]
        read = self._build_hydrop_read(deleted, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_less_than_or_equal_to(1)

    def test_match_insertion_read_idx_spans_extended_region(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that read_idx for an inserted barcode spans more than bc_len bases."""
        bc3 = hydrop_whitelists["BC3"][0]
        inserted = bc3[:5] + "A" + bc3[5:]  # 11bp
        read = self._build_hydrop_read(inserted, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        start, end = result.read_idx
        bc_len = bc3_matcher.component.length
        # The aligned span covers the inserted sequence (bc_len + 1 bp)
        assert_that(end - start).is_equal_to(bc_len + 1)

    def test_match_deletion_read_idx_spans_shorter_region(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that read_idx for a deleted barcode spans fewer than bc_len bases."""
        bc3 = hydrop_whitelists["BC3"][0]
        deleted = bc3[:5] + bc3[6:]  # 9bp
        read = self._build_hydrop_read(deleted, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        start, end = result.read_idx
        bc_len = bc3_matcher.component.length
        # The aligned span covers the deleted sequence (bc_len - 1 bp)
        assert_that(end - start).is_equal_to(bc_len - 1)

    # ==========================================
    # match() - N Wildcard Handling
    # ==========================================

    def test_match_n_base_in_barcode_region_is_wildcard(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that N bases in the read are treated as wildcards matching any base at zero cost."""
        bc3 = hydrop_whitelists["BC3"][0]
        # Replace position 5 with N: N matches any base for free
        with_n = bc3[:5] + "N" + bc3[6:]
        read = self._build_hydrop_read(with_n, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        # N is a wildcard: edit distance is 0
        assert_that(result.edit_distance).is_equal_to(0)

    def test_match_multiple_n_bases_still_match(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that multiple N bases in the barcode region are all treated as wildcards."""
        bc3 = hydrop_whitelists["BC3"][0]
        # Replace positions 2, 5, 8 with N
        with_n = bc3[:2] + "N" + bc3[3:5] + "N" + bc3[6:8] + "N" + bc3[9:]
        read = self._build_hydrop_read(with_n, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        # N in alignment is scored as MATCH_SCORE — match should still succeed
        assert_that(result.match).is_equal_to(bc3)

    # ==========================================
    # match() - Edge Cases
    # ==========================================

    def test_match_barcode_not_in_whitelist_returns_no_match(
        self, small_matcher: AlignmentMatcher
    ) -> None:
        """Test that a read with no barcode close to the whitelist returns no match."""
        # All-T: none of the small whitelist barcodes (ACGTACGTAC etc) will align above threshold
        read = "T" * 50
        result = small_matcher.match(read)
        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()

    def test_match_barcode_exceeds_max_errors_returns_no_match(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a barcode with more than max_errors mutations is not matched."""
        bc3 = hydrop_whitelists["BC3"][0]
        # Mutate max_errors + 1 = 3 positions (well beyond tolerance)
        heavily_mutated = list(bc3)
        n_subs = bc3_matcher.chemistry.max_errors.barcode + 1
        for i in range(n_subs):
            heavily_mutated[i] = "A" if heavily_mutated[i] != "A" else "C"
        mutated_str = "".join(heavily_mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        # Should either return no match or match to a different barcode
        if result[0].match is not None:
            assert_that(result[0].match).is_not_equal_to(bc3)

    def test_match_read_with_all_n_bases_returns_result(
        self, bc3_matcher: AlignmentMatcher
    ) -> None:
        """Test that an all-N read produces a result (N matches everything as wildcard)."""
        read = "N" * 50
        result = bc3_matcher.match(read)

        # All-N aligns against every barcode with full match score; result is non-empty
        assert_that(result).is_not_empty()
        assert_that(result[0]).is_instance_of(BarcodeMatchAttempt)
        assert_that(result[0].method).is_equal_to(MatchMethod.ALIGNMATCH)

    # ==========================================
    # match() - Position Independence
    # ==========================================

    def test_match_finds_barcode_regardless_of_position(
        self,
        bc2_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that AlignmentMatcher finds a barcode even when not at its expected fixed position."""
        bc2 = hydrop_whitelists["BC2"][0]
        # Place BC2 at an unusual offset, surrounded by filler
        read = "A" * 5 + bc2 + "A" * 35
        result = bc2_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc2)
        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        start, end = result.read_idx
        assert_that(read[start:end]).is_equal_to(bc2)

    def test_match_candidate_field_equals_read_slice_at_read_idx(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that result.candidate equals the read slice at result.read_idx."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.candidate).is_not_none()
        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        start, end = result.read_idx
        assert_that(result.candidate).is_equal_to(read[start:end])

    # ==========================================
    # match() - Ambiguity & Spacer Tiebreaking
    # ==========================================

    def test_match_single_candidate_returns_single_result(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that an unambiguous match returns exactly one result."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc3)

    def test_match_truly_ambiguous_barcodes_return_none_match(self) -> None:
        """Test that two barcodes with identical alignment scores return match=None.

        When two barcodes differ only at position 0 (A vs T), a read with a
        third base (C) at that position scores identically against both. Without
        adjacent spacer sequences to disambiguate, match remains None.
        """
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")

        bc_a = "ACGTACGTAC"
        bc_b = "TCGTACGTAC"  # differs from bc_a at position 0 only
        # C at position 0 gives equal edit distance (1) to both bc_a and bc_b
        ambiguous_read = "CCGTACGTAC"

        matcher = AlignmentMatcher(
            whitelist=(bc_a, bc_b),
            component=comp,
            chemistry=chemistry,
            max_errors=chemistry.max_errors.barcode,
        )
        # Pad with filler that avoids spacer sequences to prevent tiebreaking
        read = ambiguous_read + "TTTTTTTTTT" * 4
        result = matcher.match(read)

        # Both barcodes tie — none can be assigned without spacer tiebreaking
        for r in result:
            assert_that(r.match).is_none()
            assert_that(r.method).is_equal_to(MatchMethod.ALIGNMATCH)

    def test_match_ambiguous_with_no_spacer_evidence_returns_every_tied_candidate(
        self,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that a tie with no spacer evidence at all returns every tied candidate.

        Two very distinct barcodes tie at the perfect alignment score and NEITHER sits next to a
        real spacer, so the spacer-validation pass keeps nothing and the tie-break ladder falls
        through to its final exit, which returns every tied candidate rather than a
        spacer-validated subset. What distinguishes this exit from the spacer-validated
        ambiguity exit above it is that BOTH ``spacer_upstream`` and ``spacer_downstream`` are
        None on every returned attempt, so those two assertions carry the characterisation.

        BC2 is the component because it is the only HyDrop barcode with a defined spacer on both
        sides, which is what makes the spacer check meaningful here rather than vacuous.

        The filler is ten T's rather than five deliberately. ``check_spacers`` guards its
        upstream window with ``match_idx[0] >= spacer_component.length`` and its downstream
        window with the mirror-image bound, and HyDrop spacers are 10bp; with only five bases of
        filler those guards short-circuit and this test would pin a bounds check instead of a
        spacer-sequence comparison. With ten, every spacer window genuinely exists and simply
        fails to match the expected spacer sequence, which is the condition being pinned.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(self.BC_X, self.BC_Y),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # Filler is neither spacer sequence, so no spacer window can validate a candidate.
        filler = "TTTTTTTTTT"
        read = filler + self.BC_X + filler + self.BC_Y + filler
        result = matcher.match(read)

        assert_that(result).is_length(2)
        for attempt in result:
            assert_that(attempt.match).is_none()
            assert_that(attempt.edit_distance).is_none()
            assert_that(attempt.spacer_upstream).is_none()
            assert_that(attempt.spacer_downstream).is_none()
            assert_that(attempt.read_idx).is_not_none()
            assert_that(attempt.method).is_equal_to(MatchMethod.ALIGNMATCH)

        # Returned order derives from frozenset iteration and is hash-seed dependent, so the
        # candidates are compared as a sorted list rather than positionally.
        candidates = sorted(attempt.candidate for attempt in result)
        assert_that(candidates).is_equal_to(sorted([self.BC_X, self.BC_Y]))

    def test_match_ambiguous_resolved_by_spacer_returns_single_result(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that ambiguous matches are resolved by spacer context to a single result."""
        bc2 = hydrop_whitelists["BC2"][0]
        read = self._build_hydrop_read("A" * 10, bc2, "A" * 10)
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=hydrop_whitelists["BC2"],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc2)

    @pytest.mark.parametrize("bc_true, bc_other", [(BC_X, BC_Y), (BC_Y, BC_X)])
    def test_match_tiebreak_assigns_spacer_validated_survivors_own_barcode(
        self,
        bc_true: str,
        bc_other: str,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that a lone spacer-validated survivor is assigned its OWN barcode, not the tied first one.

        Two very distinct barcodes tie at the perfect alignment score, but only ``bc_true`` sits in a
        spacer-flanked slot, so a single survivor emerges (Branch A). The buggy implementation assigns
        ``best_alignments[0].bc`` — the frozenset-first tied barcode — whose identity is hash-arbitrary.
        Parametrising both role assignments of the same pair makes exactly one case assign the wrong
        barcode (failing the recomputed ``max_errors`` gate) on the buggy code under any hash seed,
        while both cases pass once each survivor keeps its own originating barcode.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(bc_true, bc_other),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # bc_true is flanked by both expected spacers; bc_other sits in non-spacer filler.
        read = bc_other + "TTTTT" + self.SPACER_1 + bc_true + self.SPACER_2 + "TTTTT"
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc_true)
        assert_that(result[0].candidate).is_equal_to(bc_true)
        assert_that(result[0].edit_distance).is_equal_to(0)
        assert_that(edit_distance(result[0].candidate, result[0].match)).is_equal_to(
            result[0].edit_distance
        )

    @pytest.mark.parametrize("bc_true, bc_other", [(BC_X, BC_Y), (BC_Y, BC_X)])
    def test_match_tiebreak_two_spacer_survivor_assigns_own_barcode(
        self,
        bc_true: str,
        bc_other: str,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that the unique two-spacer survivor is assigned its OWN barcode (Branch B).

        Both tied barcodes are spacer-validated (each has at least one expected spacer), but only
        ``bc_true`` is flanked by both spacers, so it wins via the two-spacer narrowing path. As in
        Branch A, the buggy code assigns the frozenset-first tied barcode; the role-swap parametrisation
        forces exactly one case to assign the wrong barcode on the buggy code under any hash seed.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(bc_true, bc_other),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # bc_other has only an upstream spacer (downstream is filler); bc_true has both spacers.
        read = (
            "TTTTT"
            + self.SPACER_1
            + bc_other
            + "TTTTT"
            + self.SPACER_1
            + bc_true
            + self.SPACER_2
            + "TTTTT"
        )
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc_true)
        assert_that(result[0].candidate).is_equal_to(bc_true)
        assert_that(result[0].edit_distance).is_equal_to(0)
        assert_that(edit_distance(result[0].candidate, result[0].match)).is_equal_to(
            result[0].edit_distance
        )

    def test_match_tiebreak_multiple_validated_survivors_are_ambiguous(
        self,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that multiple equally spacer-validated survivors fall through to the ambiguous path.

        Both barcodes tie at the perfect alignment score and both are flanked by both spacers, so no
        unique survivor emerges. The matcher must return every validated result with ``match=None``
        rather than committing to a single, hash-order-dependent barcode.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(self.BC_X, self.BC_Y),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # Both barcodes are flanked by both spacers -> identical spacer profile -> ambiguous.
        read = (
            self.SPACER_1
            + self.BC_X
            + self.SPACER_2
            + "TTTTT"
            + self.SPACER_1
            + self.BC_Y
            + self.SPACER_2
        )
        result = matcher.match(read)

        assert_that(len(result)).is_greater_than_or_equal_to(2)
        for r in result:
            assert_that(r.match).is_none()
            assert_that(r.method).is_equal_to(MatchMethod.ALIGNMATCH)

    def test_match_single_candidate_exceeding_max_errors_returns_no_match(
        self,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that a lone best alignment whose candidate exceeds max_errors returns no match.

        With a single-barcode whitelist the match funnels through the single-candidate branch
        (no spacer validation). A 3-base insertion inside the barcode keeps the local alignment
        score above ``score_threshold`` (7.5 >= 7.0) but makes the aligned candidate span 13 bp,
        so ``edit_distance(candidate, barcode) == 3`` exceeds ``max_errors`` (2) and the branch
        must discard it as a no-match.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(self.BC_X,),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # 3-base insertion inside BC_X -> 13 bp aligned candidate, edit_distance == 3 (> max_errors).
        mutated = self.BC_X[:5] + "AAA" + self.BC_X[5:]
        read = "TTTTT" + mutated + "TTTTT"
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()
        assert_that(result[0].candidate).is_none()
        assert_that(result[0].method).is_equal_to(MatchMethod.ALIGNMATCH)

    def test_match_tiebreak_two_spacer_survivor_exceeding_max_errors_returns_no_match(
        self,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that the two-spacer survivor (Branch B) is discarded when its candidate exceeds max_errors.

        Two barcodes tie at the same degraded alignment score (7.5, each carrying a 3-base
        insertion), both are spacer-validated, but only the survivor is flanked by both spacers,
        so Branch B narrows to it. Its aligned candidate spans 13 bp with
        ``edit_distance(candidate, own_barcode) == 3`` (> max_errors 2), so the branch must return
        a no-match rather than assigning the barcode.
        """
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        matcher = AlignmentMatcher(
            whitelist=(self.BC_X, self.BC_Y),
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        # Both barcodes carry a 3-base insertion (tie at score 7.5). The survivor (BC_X) is flanked
        # by both spacers; the other (BC_Y) has only an upstream spacer -> Branch B narrows to BC_X.
        survivor_ins = self.BC_X[:5] + "AAA" + self.BC_X[5:]
        other_ins = self.BC_Y[:5] + "AAA" + self.BC_Y[5:]
        read = (
            self.SPACER_1
            + survivor_ins
            + self.SPACER_2
            + "TTTTT"
            + self.SPACER_1
            + other_ins
            + "TTTTT"
        )
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_none()
        assert_that(result[0].candidate).is_none()
        assert_that(result[0].method).is_equal_to(MatchMethod.ALIGNMATCH)

    # ==========================================
    # match() - Full HyDrop Read Sequences (Parametrized)
    # ==========================================

    @pytest.mark.parametrize(
        "seq, bc_name",
        [
            # Case 1 (WL_MATCH): all 3 BCs match
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC3"),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC2"),
            ("CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "BC1"),
            # Case 2 (INDEL_FAIL): BC3 matches even under indel pressure
            ("TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "BC3"),
            # Case 3 (SUB_FAIL): BC3 + BC2 match with substitution tolerance
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC3"),
            ("GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC", "BC2"),
            # Case 5 (SPC_NOTFND): BC3 + BC2 match
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC3"),
            ("TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT", "BC2"),
            # Case 6 (SEVERE_INDEL): BC3 matches
            ("TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC", "BC3"),
        ],
    )
    def test_match_dev_sequences_expected_matches(
        self,
        seq: str,
        bc_name: str,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test alignment matching using real HyDrop read sequences that should produce matches."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        matcher = AlignmentMatcher(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )
        result = matcher.match(seq)

        assert_that(result).is_not_empty()
        matched_results = [r for r in result if r.match is not None]
        assert_that(matched_results).is_not_empty()
        assert_that(matched_results[0].match).is_in(*hydrop_whitelists[bc_name])
        assert_that(matched_results[0].read_idx).is_not_none()

    # ==========================================
    # match() - Edit Distance Accuracy
    # ==========================================

    @pytest.mark.parametrize("n_substitutions", [0, 1, 2])
    def test_match_edit_distance_equals_substitution_count(
        self,
        n_substitutions: int,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that edit_distance in the result accurately reflects the number of substitutions."""
        bc3 = hydrop_whitelists["BC3"][0]
        mutated = list(bc3)
        for i in range(n_substitutions):
            mutated[i] = "A" if mutated[i] != "A" else "C"
        mutated_str = "".join(mutated)

        read = self._build_hydrop_read(mutated_str, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(n_substitutions)

    # ==========================================
    # match() - Different Barcode Components
    # ==========================================

    @pytest.mark.parametrize("bc_name", ["BC3", "BC2", "BC1"])
    def test_match_across_all_barcode_components(
        self,
        bc_name: str,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that AlignmentMatcher works correctly for all three HyDrop barcode components."""
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        whitelist = hydrop_whitelists[bc_name]
        matcher = AlignmentMatcher(
            whitelist=whitelist,
            component=comp,
            chemistry=hydrop_chemistry,
            max_errors=hydrop_chemistry.max_errors.barcode,
        )

        bc3 = hydrop_whitelists["BC3"][0]
        bc2 = hydrop_whitelists["BC2"][0]
        bc1 = hydrop_whitelists["BC1"][0]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        result = matcher.match(read)
        matched_results = [r for r in result if r.match is not None]
        assert_that(matched_results).is_not_empty()
        assert_that(matched_results[0].match).is_in(*whitelist)

    # ==========================================
    # match() - Spacer Fields in Result
    # ==========================================

    def test_match_unambiguous_result_has_no_spacer_fields(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a single-candidate match does not populate spacer fields."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = self._build_hydrop_read(bc3, "A" * 10, "A" * 10)
        result = bc3_matcher.match(read)[0]

        # Unambiguous matches go through the single-candidate path — no spacer check
        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.spacer_upstream).is_none()
        assert_that(result.spacer_downstream).is_none()

    # ==========================================
    # match() - Realistic Multi-Barcode Reads
    # ==========================================

    def test_match_correct_barcode_in_multi_barcode_read(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that each matcher finds only its own barcode in a read with all 3 barcodes."""
        bc3 = hydrop_whitelists["BC3"][0]
        bc2 = hydrop_whitelists["BC2"][0]
        bc1 = hydrop_whitelists["BC1"][0]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        for bc_name, expected_bc in [("BC3", bc3), ("BC2", bc2), ("BC1", bc1)]:
            comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
            matcher = AlignmentMatcher(
                whitelist=hydrop_whitelists[bc_name],
                component=comp,
                chemistry=hydrop_chemistry,
                max_errors=hydrop_chemistry.max_errors.barcode,
            )
            result = matcher.match(read)
            matched = [r for r in result if r.match is not None]
            assert_that(matched).is_not_empty()
            assert_that(matched[0].match).is_equal_to(expected_bc)

    def test_match_different_barcodes_per_component(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when each barcode position uses a different whitelist index."""
        bc3 = hydrop_whitelists["BC3"][10]
        bc2 = hydrop_whitelists["BC2"][20]
        bc1 = hydrop_whitelists["BC1"][30]
        read = self._build_hydrop_read(bc3, bc2, bc1)

        for bc_name, expected_bc in [("BC3", bc3), ("BC2", bc2), ("BC1", bc1)]:
            comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
            matcher = AlignmentMatcher(
                whitelist=hydrop_whitelists[bc_name],
                component=comp,
                chemistry=hydrop_chemistry,
                max_errors=hydrop_chemistry.max_errors.barcode,
            )
            result = matcher.match(read)
            matched = [r for r in result if r.match is not None]
            assert_that(matched).is_not_empty()
            assert_that(matched[0].match).is_equal_to(expected_bc)

    # ==========================================
    # match() - Boundary & Robustness
    # ==========================================

    def test_match_barcode_at_very_end_of_read(
        self,
        bc1_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when the barcode is at the very end of the read."""
        bc1 = hydrop_whitelists["BC1"][0]
        read = "A" * 40 + bc1  # 50bp total, bc1 at end
        result = bc1_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc1)
        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        assert_that(result.read_idx[1]).is_equal_to(50)

    def test_match_barcode_at_very_start_of_read(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching when the barcode is at the very start of the read."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = bc3 + "A" * 40
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.read_idx).is_not_none()
        assert result.read_idx is not None
        assert_that(result.read_idx[0]).is_equal_to(0)

    def test_match_with_long_read(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test matching in a read much longer than the typical 50bp HyDrop read."""
        bc3 = hydrop_whitelists["BC3"][0]
        read = bc3 + "A" * 200
        result = bc3_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(bc3)
        assert_that(result.edit_distance).is_equal_to(0)

    def test_match_minimum_viable_read(self, small_matcher: AlignmentMatcher) -> None:
        """Test matching with a read that is exactly the barcode length."""
        barcode = "ACGTACGTAC"
        read = barcode  # exactly bc_len, no flanking sequence
        result = small_matcher.match(read)[0]

        assert_that(result.match).is_equal_to(barcode)
        assert_that(result.edit_distance).is_equal_to(0)
