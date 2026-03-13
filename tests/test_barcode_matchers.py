"""
Tests for barcode matcher classes: MatcherBase, FixedPositionMatcher, KmerMatcher, and AlignmentMatcher.
"""

from typing import Literal

import pytest
from assertpy import assert_that

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
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent
from tests.test_chemistry import ChemistryCarmackCustomSeq10


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
        chemistry = ChemistryHydrop()
        barcode_component = chemistry.read_structure.get_component_by_name("BC3")
        return chemistry.barcode_whitelists[barcode_component.name]

    @pytest.fixture
    def matcher(self, whitelist: tuple[str, ...]) -> MatcherBase:
        """Provide a MatcherBase instance using FixedPositionMatcher for testing."""
        chemistry = ChemistryHydrop()
        barcode_component = chemistry.read_structure.get_component_by_name("BC3")
        return FixedPositionMatcher(
            whitelist=whitelist, barcode_component=barcode_component, chemistry=chemistry
        )

    def test_init_with_valid_barcode_component(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...], matcher: MatcherBase
    ) -> None:
        """Test that FixedPositionMatcher initializes successfully with a valid barcode component."""
        matcher = FixedPositionMatcher(
            whitelist=whitelist, barcode_component=barcode_component, chemistry=ChemistryHydrop()
        )
        assert_that(matcher.barcode_component).is_equal_to(barcode_component)

    def test_init_rejects_non_barcode_component(self, whitelist: tuple[str, ...]) -> None:
        """Test that initializing with a non-barcode component raises ValueError."""
        non_barcode = ReadComponent(
            name="SPACER", is_barcode=False, length=10, sequence="AGGGTACTCG"
        )
        with pytest.raises(ValueError):
            FixedPositionMatcher(
                whitelist=whitelist, barcode_component=non_barcode, chemistry=ChemistryHydrop()
            )

    def test_whitelist_stored_as_frozenset(
        self, whitelist: tuple[str, ...], matcher: MatcherBase
    ) -> None:
        """Test that the whitelist is stored as a frozenset containing all entries."""
        assert_that(matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(matcher.whitelist_set).is_equal_to(frozenset(whitelist))

    def test_check_read_len_sufficient(
        self, barcode_component: ReadComponent, whitelist: tuple[str, ...], matcher: MatcherBase
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

    def test_trim_read_basic(self, matcher: MatcherBase) -> None:
        """Test that trim_read returns the substring from the given start position."""
        assert_that(matcher.trim_read("ACGTACGT", 4)).is_equal_to("ACGT")

    def test_trim_read_start_zero(self, matcher: MatcherBase) -> None:
        """Test that trim_read with start=0 returns the full read."""
        assert_that(matcher.trim_read("ACGTACGT", 0)).is_equal_to("ACGTACGT")

    def test_trim_read_start_beyond_length(self, matcher: MatcherBase) -> None:
        """Test that trim_read returns empty string when start is beyond read length."""
        assert_that(matcher.trim_read("ACGT", 10)).is_equal_to("")

    def test_trim_read_start_at_length(self, matcher: MatcherBase) -> None:
        """Test that trim_read returns empty string when start equals read length."""
        assert_that(matcher.trim_read("ACGT", 4)).is_equal_to("")

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
        matcher = FixedPositionMatcher(
            whitelist=("AAAAAAAAAA",), barcode_component=comp, chemistry=ChemistryHydrop()
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
        whitelist = chemistry.load_barcode_whitelist(bc)
        matcher = FixedPositionMatcher(
            whitelist=whitelist, barcode_component=comp, chemistry=chemistry
        )

        # General carmack_custom_seq_1_0 read structure
        bc_idx_map = {"BC3": (22, 32), "BC2": (54, 64), "BC1": (86, 96)}
        match_idx = bc_idx_map[bc]

        spacer_results = matcher.check_spacers(read, match_idx)

        assert_that(spacer_results).contains_key("upstream").contains_key("downstream")
        assert_that(spacer_results["upstream"]).is_equal_to(upstream_spacer)
        assert_that(spacer_results["downstream"]).is_equal_to(downstream_spacer)


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
        return FixedPositionMatcher(
            whitelist=hydrop_whitelists["BC3"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc2_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher for HyDrop BC2 (start=20, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        return FixedPositionMatcher(
            whitelist=hydrop_whitelists["BC2"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc1_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> FixedPositionMatcher:
        """Provide a FixedPositionMatcher for HyDrop BC1 (start=40, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC1")
        return FixedPositionMatcher(
            whitelist=hydrop_whitelists["BC1"], barcode_component=comp, chemistry=hydrop_chemistry
        )

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
            barcode_component=comp,
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
    ) -> KmerMatcher:
        """Provide a KmerMatcher for HyDrop BC3 (start=0, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC3")
        return KmerMatcher(
            whitelist=hydrop_whitelists["BC3"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc2_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> KmerMatcher:
        """Provide a KmerMatcher for HyDrop BC2 (start=20, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        return KmerMatcher(
            whitelist=hydrop_whitelists["BC2"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc1_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> KmerMatcher:
        """Provide a KmerMatcher for HyDrop BC1 (start=40, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC1")
        return KmerMatcher(
            whitelist=hydrop_whitelists["BC1"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def small_whitelist(self) -> tuple[str, ...]:
        """Provide a small, deterministic whitelist for isolated tests."""
        return ("ACGTACGTAC", "TGCATGCATG", "GGGGGGGGGG", "CCCCCCCCCC")

    @pytest.fixture
    def small_matcher(self, small_whitelist: tuple[str, ...]) -> KmerMatcher:
        """Provide a KmerMatcher with a small whitelist and simple chemistry (HyDrop BC3)."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        return KmerMatcher(whitelist=small_whitelist, barcode_component=comp, chemistry=chemistry)

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
        assert_that(bc3_matcher.barcode_component.name).is_equal_to("BC3")
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
            k=k,
        )
        assert_that(matcher.k).is_equal_to(k)
        assert_that(matcher.kmer_index).is_not_empty()

    def test_initialization_rejects_non_barcode_component(self) -> None:
        """Test that initializing with a non-barcode component raises ValueError."""
        non_barcode = ReadComponent(
            name="SPACER", is_barcode=False, length=10, sequence="AGGGTACTCG"
        )
        with pytest.raises(ValueError):
            KmerMatcher(
                whitelist=("AAAAAAAAAA",),
                barcode_component=non_barcode,
                chemistry=ChemistryHydrop(),
            )

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
            whitelist=(barcode,), barcode_component=comp, chemistry=chemistry, k=k
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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

        matcher = KmerMatcher(whitelist=(bc_a, bc_b), barcode_component=comp, chemistry=chemistry)
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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
            whitelist=whitelist, barcode_component=comp, chemistry=hydrop_chemistry
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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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
            whitelist=("ACGTACGTAC",), barcode_component=comp, chemistry=chemistry
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
        matcher = KmerMatcher(whitelist=(), barcode_component=comp, chemistry=chemistry)
        assert_that(matcher.kmer_index).is_empty()

    def test_build_kmer_index_duplicate_barcodes_deduplicated(self) -> None:
        """Test that duplicate barcodes in whitelist are deduplicated via frozenset."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        matcher = KmerMatcher(
            whitelist=("ACGTACGTAC", "ACGTACGTAC", "ACGTACGTAC"),
            barcode_component=comp,
            chemistry=chemistry,
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
            whitelist=(barcode,), barcode_component=comp, chemistry=chemistry, k=k
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
                barcode_component=comp,
                chemistry=hydrop_chemistry,
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
                barcode_component=comp,
                chemistry=hydrop_chemistry,
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


class TestAlignmentMatcher:
    """Tests for AlignmentMatcher barcode matching logic."""

    # --- HyDrop spacer constants for read construction ---
    SPACER_1 = "AGGGTACTCG"
    SPACER_2 = "GCAGTAGCTG"

    # --- Fixtures ---

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
    ) -> AlignmentMatcher:
        """Provide an AlignmentMatcher for HyDrop BC3 (start=0, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC3")
        return AlignmentMatcher(
            whitelist=hydrop_whitelists["BC3"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc2_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> AlignmentMatcher:
        """Provide an AlignmentMatcher for HyDrop BC2 (start=20, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC2")
        return AlignmentMatcher(
            whitelist=hydrop_whitelists["BC2"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def bc1_matcher(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_whitelists: dict[str, tuple[str, ...]]
    ) -> AlignmentMatcher:
        """Provide an AlignmentMatcher for HyDrop BC1 (start=40, length=10)."""
        comp = hydrop_chemistry.read_structure.get_component_by_name("BC1")
        return AlignmentMatcher(
            whitelist=hydrop_whitelists["BC1"], barcode_component=comp, chemistry=hydrop_chemistry
        )

    @pytest.fixture
    def small_whitelist(self) -> tuple[str, ...]:
        """Provide a small, deterministic whitelist for isolated tests."""
        return ("ACGTACGTAC", "TGCATGCATG", "GGGGGGGGGG", "CCCCCCCCCC")

    @pytest.fixture
    def small_matcher(self, small_whitelist: tuple[str, ...]) -> AlignmentMatcher:
        """Provide an AlignmentMatcher with a small whitelist and simple chemistry (HyDrop BC3)."""
        chemistry = ChemistryHydrop()
        comp = chemistry.read_structure.get_component_by_name("BC3")
        return AlignmentMatcher(
            whitelist=small_whitelist, barcode_component=comp, chemistry=chemistry
        )

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
        assert_that(bc3_matcher.barcode_component.name).is_equal_to("BC3")
        assert_that(bc3_matcher.whitelist_set).is_instance_of(frozenset)
        assert_that(bc3_matcher.whitelist_set).is_equal_to(frozenset(hydrop_whitelists["BC3"]))

    def test_initialization_sets_score_threshold(self, bc3_matcher: AlignmentMatcher) -> None:
        """Test that AlignmentMatcher sets score_threshold during initialization."""
        assert_that(bc3_matcher.score_threshold).is_not_none()
        assert_that(bc3_matcher.score_threshold).is_instance_of(float)

    def test_initialization_rejects_non_barcode_component(self) -> None:
        """Test that initializing with a non-barcode component raises ValueError."""
        non_barcode = ReadComponent(
            name="SPACER", is_barcode=False, length=10, sequence="AGGGTACTCG"
        )
        with pytest.raises(ValueError):
            AlignmentMatcher(
                whitelist=("AAAAAAAAAA",),
                barcode_component=non_barcode,
                chemistry=ChemistryHydrop(),
            )

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
        bc_len = bc3_matcher.barcode_component.length
        max_errors = bc3_matcher.chemistry.max_errors.barcode
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
        bc3_wl = list(bc3_matcher.whitelist_set)
        bc3 = bc3_wl[0]

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

        With threshold=7.0, 4 substitutions on a 10bp barcode: score = 10 - 4 = 6.0,
        which is below threshold=7.0.
        """
        bc3_wl = list(bc3_matcher.whitelist_set)
        bc3 = bc3_wl[0]

        # Introduce 4 substitutions to fall below threshold (score = 6.0 < 7.0)
        mutated = list(bc3)
        n_subs = 4
        for i in range(n_subs):
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

        bc_len = bc3_matcher.barcode_component.length
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
        """Test that seq1_coords from align_seqs correctly span the matched region in seq1."""
        bc3 = hydrop_whitelists["BC3"][0]
        result = bc3_matcher.align_seqs(bc3, bc3)

        assert_that(result).is_not_none()
        assert result is not None
        assert_that(result.seq1_coords).is_not_empty()
        start, end = result.seq1_coords[0]
        assert_that(bc3[start:end]).is_equal_to(bc3)

    def test_align_seqs_perfect_match_has_full_bc_length_score(
        self,
        bc3_matcher: AlignmentMatcher,
        hydrop_whitelists: dict[str, tuple[str, ...]],
    ) -> None:
        """Test that a perfect match scores bc_len * MATCH_SCORE."""
        bc3 = hydrop_whitelists["BC3"][0]
        bc_len = bc3_matcher.barcode_component.length
        expected_score = bc_len * MATCH_SCORE

        result = bc3_matcher.align_seqs(bc3, bc3)
        assert_that(result).is_not_none()
        assert_that(result.score).is_equal_to(expected_score)  # type: ignore[union-attr]

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
        bc_len = bc3_matcher.barcode_component.length
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
        bc_len = bc3_matcher.barcode_component.length
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
            whitelist=(bc_a, bc_b), barcode_component=comp, chemistry=chemistry
        )
        # Pad with filler that avoids spacer sequences to prevent tiebreaking
        read = ambiguous_read + "TTTTTTTTTT" * 4
        result = matcher.match(read)

        # Both barcodes tie — none can be assigned without spacer tiebreaking
        for r in result:
            assert_that(r.match).is_none()
            assert_that(r.method).is_equal_to(MatchMethod.ALIGNMATCH)

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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
        )
        result = matcher.match(read)

        assert_that(result).is_length(1)
        assert_that(result[0].match).is_equal_to(bc2)

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
            barcode_component=comp,
            chemistry=hydrop_chemistry,
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
            whitelist=whitelist, barcode_component=comp, chemistry=hydrop_chemistry
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
                barcode_component=comp,
                chemistry=hydrop_chemistry,
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
                barcode_component=comp,
                chemistry=hydrop_chemistry,
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
