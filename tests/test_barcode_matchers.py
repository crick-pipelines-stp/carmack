"""
Tests for barcode matcher classes: MatcherBase and FixedPositionMatcher.
"""

import pytest
from assertpy import assert_that

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent


class TestBarcodeMatcherBase:
    """Tests for MatcherBase abstract class functionality, exercised via FixedPositionMatcher."""

    @pytest.fixture
    def barcode_component(self) -> ReadComponent:
        """Provide a barcode ReadComponent with start position set."""
        comp = ReadComponent(name="BC_TEST", is_barcode=True, length=10)
        comp.start = 5
        return comp

    @pytest.fixture
    def whitelist(self) -> tuple[str, ...]:
        """Provide a small test whitelist."""
        return ("AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG", "TTTTTTTTTT")

    def test_init_with_valid_barcode_component(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...]
    ) -> None:
        """Test that FixedPositionMatcher initializes successfully with a valid barcode component."""
        matcher = FixedPositionMatcher(whitelist=whitelist, barcode_component=barcode_component)
        assert_that(matcher.barcode_component).is_equal_to(barcode_component)

    def test_init_rejects_non_barcode_component(self, whitelist: tuple[str, ...]) -> None:
        """Test that initializing with a non-barcode component raises ValueError."""
        non_barcode = ReadComponent(
            name="SPACER", is_barcode=False, length=10, sequence="AGGGTACTCG"
        )
        with pytest.raises(ValueError):
            FixedPositionMatcher(whitelist=whitelist, barcode_component=non_barcode)

    def test_whitelist_stored_as_frozenset(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...]
    ) -> None:
        """Test that the whitelist is stored as a frozenset containing all entries."""
        matcher = FixedPositionMatcher(whitelist=whitelist, barcode_component=barcode_component)
        assert_that(matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(matcher.whitelist_set).is_equal_to(frozenset(whitelist))

    def test_check_read_len_sufficient(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...]
    ) -> None:
        """Test that check_read_len returns True when the read is long enough."""
        matcher = FixedPositionMatcher(whitelist=whitelist, barcode_component=barcode_component)
        # start=5, length=10, so need >= 15
        read = "A" * 20
        assert_that(matcher.check_read_len(read)).is_true()

    def test_check_read_len_insufficient(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...]
    ) -> None:
        """Test that check_read_len returns False when the read is too short."""
        matcher = FixedPositionMatcher(whitelist=whitelist, barcode_component=barcode_component)
        # start=5, length=10, so need >= 15
        read = "A" * 10
        assert_that(matcher.check_read_len(read)).is_false()

    def test_check_read_len_exact_boundary(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...]
    ) -> None:
        """Test that check_read_len returns True when the read length exactly meets the requirement."""
        matcher = FixedPositionMatcher(whitelist=whitelist, barcode_component=barcode_component)
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
        comp = ReadComponent(name="BC_TEST", is_barcode=True, length=bc_len)
        comp.start = start
        matcher = FixedPositionMatcher(whitelist=("AAAAAAAAAA",), barcode_component=comp)
        read = "A" * read_len
        assert_that(matcher.check_read_len(read)).is_equal_to(expected)


class TestFixedPositionMatcher:
    """Tests for FixedPositionMatcher barcode matching logic."""

    @pytest.fixture
    def hydrop_chemistry(self) -> ChemistryHydrop:
        """Provide a HyDrop chemistry instance."""
        return ChemistryHydrop()

    @pytest.fixture
    def hydrop_whitelists(self, hydrop_chemistry: ChemistryHydrop) -> dict[str, tuple[str, ...]]:
        """Provide stripped HyDrop whitelists (10bp variable regions)."""
        return hydrop_chemistry.barcode_whitelists

    @pytest.fixture
    def bc3_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher for HyDrop BC3 (start=0, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC3")
        return FixedPositionMatcher(whitelist=hydrop_whitelists["BC3"], barcode_component=comp)

    @pytest.fixture
    def bc2_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher for HyDrop BC2 (start=20, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        return FixedPositionMatcher(whitelist=hydrop_whitelists["BC2"], barcode_component=comp)

    @pytest.fixture
    def bc1_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher for HyDrop BC1 (start=40, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC1")
        return FixedPositionMatcher(whitelist=hydrop_whitelists["BC1"], barcode_component=comp)

    def test_match_returns_barcode_match_attempt(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that match() returns a BarcodeMatchAttempt instance."""
        read = "A" * 52
        result = bc3_matcher.match(read)
        assert_that(result).is_instance_of(BarcodeMatchAttempt)

    def test_match_method_is_always_exactmatch(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that the method field is always EXACTMATCH regardless of match outcome."""
        read = "A" * 52
        result = bc3_matcher.match(read)
        assert_that(result.method).is_equal_to(MatchMethod.EXACTMATCH)

    def test_match_candidate_is_extracted_substring(
        self, bc3_matcher: FixedPositionMatcher
    ) -> None:
        """Test that candidate is the substring at [start:start+length] from the read."""
        read = "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT"
        result = bc3_matcher.match(read)
        # BC3: start=0, length=10
        assert_that(result.candidate).is_equal_to(read[0:10])

    def test_match_read_too_short_returns_no_match(
        self, bc1_matcher: FixedPositionMatcher
    ) -> None:
        """Test that a read shorter than required returns no match."""
        # BC1 needs start=40 + length=10 = 50bp minimum
        read = "A" * 30
        result = bc1_matcher.match(read)
        assert_that(result.match).is_none()
        assert_that(result.read_idx).is_none()

    def test_match_exact_returns_correct_read_idx(
        self, bc3_matcher: FixedPositionMatcher, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> None:
        """Test that a successful match returns the correct read index tuple."""
        bc3_barcode = hydrop_whitelists["BC3"][0]
        read = bc3_barcode + "A" * 42  # BC3 at [0:10], padded to 52bp
        result = bc3_matcher.match(read)
        assert_that(result.match).is_equal_to(bc3_barcode)
        assert_that(result.read_idx).is_equal_to((0, 10))

    def test_match_no_match_returns_none_idx(self, bc3_matcher: FixedPositionMatcher) -> None:
        """Test that a failed match returns None for both match and read_idx."""
        read = "ZZZZZZZZZZ" + "A" * 42
        result = bc3_matcher.match(read)
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
            whitelist=hydrop_whitelists[bc_name], barcode_component=comp
        )
        result = matcher.match(seq)

        if expected_match:
            assert_that(result.match).is_not_none()
            assert_that(result.match).is_in(*hydrop_whitelists[bc_name])
            assert_that(result.read_idx).is_not_none()
        else:
            assert_that(result.match).is_none()
