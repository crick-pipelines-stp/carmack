"""
Tests for barcode extraction pipeline: HybridExtractor, dataclasses, and utility functions.
"""

import numpy as np
import pytest
from assertpy import assert_that

from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.barcode.barcode_utils import edit_distance, hamming_distance
from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    BarcodeMatchHistory,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.hybrid_extractor import HybridExtractor
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop


R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"


class TestBarcodeExtractor:
    """Tests for barcode extractor class with Hydrop chemistry."""

    @pytest.fixture(scope="class")
    def barcode_extractor(self) -> BarcodeExtractor:
        """Provide a BarcodeExtractor instance initialized with the test FASTQ and HyDrop chemistry."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="HyDrop",
            n_workers=1,  # Use single worker for testing
        )
        return extractor

    ...


class TestBarcodeExtractorDataclasses:
    """Tests for barcode extraction dataclasses: MatchMethod, BarcodeMatchAttempt, BarcodeMatchHistory, ReadMatchResult."""

    # ===== MatchMethod enum =====

    def test_match_method_enum_values(self) -> None:
        """Test that MatchMethod enum contains the expected values."""
        assert_that(MatchMethod.EXACTMATCH.value).is_equal_to("EXACTMATCH")
        assert_that(MatchMethod.KMERMATCH.value).is_equal_to("KMERMATCH")
        assert_that(MatchMethod.ALIGNMATCH.value).is_equal_to("ALIGNMATCH")

    def test_match_method_enum_has_three_members(self) -> None:
        """Test that MatchMethod has exactly three members."""
        assert_that(len(MatchMethod)).is_equal_to(3)

    # ===== BarcodeMatchAttempt =====

    def test_barcode_match_attempt_creation(self) -> None:
        """Test that BarcodeMatchAttempt can be created with required fields."""
        attempt = BarcodeMatchAttempt(
            candidate="ACGTACGTAC",
            method=MatchMethod.EXACTMATCH,
            match="ACGTACGTAC",
            read_idx=(10, 20),
        )
        assert_that(attempt.candidate).is_equal_to("ACGTACGTAC")
        assert_that(attempt.method).is_equal_to(MatchMethod.EXACTMATCH)
        assert_that(attempt.match).is_equal_to("ACGTACGTAC")
        assert_that(attempt.read_idx).is_equal_to((10, 20))

    def test_barcode_match_attempt_defaults(self) -> None:
        """Test that optional fields on BarcodeMatchAttempt default to None."""
        attempt = BarcodeMatchAttempt(candidate="ACGTACGTAC", method=MatchMethod.EXACTMATCH)
        assert_that(attempt.match).is_none()
        assert_that(attempt.read_idx).is_none()
        assert_that(attempt.edit_distance).is_none()
        assert_that(attempt.spacer_upstream).is_none()
        assert_that(attempt.spacer_downstream).is_none()

    # ===== BarcodeMatchHistory =====

    def test_barcode_match_history_record_attempt_success(self) -> None:
        """Test that recording a successful attempt sets success to True."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.EXACTMATCH, match="ACGTACGTAC"
        )
        history.record_attempt(attempt, success=True)

        assert_that(history.success).is_true()
        assert_that(history.attempts).is_length(1)
        assert_that(history.attempts[0]).is_equal_to(attempt)

    def test_barcode_match_history_record_attempt_failure(self) -> None:
        """Test that recording a failed attempt leaves success as False."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt = BarcodeMatchAttempt(candidate="ZZZZZZZZZZ", method=MatchMethod.EXACTMATCH)
        history.record_attempt(attempt, success=False)

        assert_that(history.success).is_false()
        assert_that(history.attempts).is_length(1)

    def test_barcode_match_history_multiple_attempts(self) -> None:
        """Test that multiple attempts are recorded in order."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt1 = BarcodeMatchAttempt(candidate="AAAAAAAAAA", method=MatchMethod.EXACTMATCH)
        attempt2 = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.KMERMATCH, match="ACGTACGTAC"
        )
        history.record_attempt(attempt1, success=False)
        history.record_attempt(attempt2, success=True)

        assert_that(history.attempts).is_length(2)
        assert_that(history.success).is_true()

    def test_barcode_match_history_succeeded_at_returns_method(self) -> None:
        """Test that succeeded_at returns the method of the last attempt when successful."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.KMERMATCH, match="ACGTACGTAC"
        )
        history.record_attempt(attempt, success=True)

        assert_that(history.succeeded_at).is_equal_to(MatchMethod.KMERMATCH)

    def test_barcode_match_history_succeeded_at_returns_none_on_failure(self) -> None:
        """Test that succeeded_at returns None when matching was not successful."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt = BarcodeMatchAttempt(candidate="ZZZZZZZZZZ", method=MatchMethod.EXACTMATCH)
        history.record_attempt(attempt, success=False)

        assert_that(history.succeeded_at).is_none()

    def test_barcode_match_history_ambiguous_matches_no_duplicates(self) -> None:
        """Test that ambiguous_matches reports False when each method appears once."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt1 = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.EXACTMATCH, match="ACGTACGTAC"
        )
        attempt2 = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.KMERMATCH, match="ACGTACGTAC"
        )
        history.record_attempt(attempt1, success=False)
        history.record_attempt(attempt2, success=True)

        ambiguous = history.ambiguous_matches
        assert_that(ambiguous.get(MatchMethod.EXACTMATCH, False)).is_false()
        assert_that(ambiguous.get(MatchMethod.KMERMATCH, False)).is_false()

    def test_barcode_match_history_ambiguous_matches_with_duplicates(self) -> None:
        """Test that ambiguous_matches reports True when a method appears multiple times."""
        history = BarcodeMatchHistory(bc_name="BC1")
        attempt1 = BarcodeMatchAttempt(
            candidate="ACGTACGTAC", method=MatchMethod.KMERMATCH, match="ACGTACGTAC"
        )
        attempt2 = BarcodeMatchAttempt(
            candidate="TGCATGCATG", method=MatchMethod.KMERMATCH, match="TGCATGCATG"
        )
        history.record_attempt(attempt1, success=False)
        history.record_attempt(attempt2, success=True)

        ambiguous = history.ambiguous_matches
        assert_that(ambiguous[MatchMethod.KMERMATCH]).is_true()

    @pytest.mark.parametrize(
        "attempts_data, expected_status",
        [
            # Single exact match success
            (
                [("ACGT", MatchMethod.EXACTMATCH, "ACGT", None, None, None)],
                "BC1:EXACTMATCH",
            ),
            # Single no match
            (
                [("ZZZZ", MatchMethod.EXACTMATCH, None, None, None, None)],
                "BC1:NOMATCH",
            ),
            # Exact match fail then kmer match success
            (
                [
                    ("ZZZZ", MatchMethod.EXACTMATCH, None, None, None, None),
                    ("ACGT", MatchMethod.KMERMATCH, "ACGT", None, None, None),
                ],
                "BC1:KMERMATCH",
            ),
            # Match with edit distance
            (
                [("ACGT", MatchMethod.KMERMATCH, "ACGT", 1, None, None)],
                "BC1:KMERMATCH-ED1",
            ),
            # Match with upstream spacer
            (
                [("ACGT", MatchMethod.ALIGNMATCH, "ACGT", None, "SPACER_1", None)],
                "BC1:ALIGNMATCH-spUp",
            ),
            # Match with downstream spacer
            (
                [("ACGT", MatchMethod.ALIGNMATCH, "ACGT", None, None, "SPACER_2")],
                "BC1:ALIGNMATCH-spDown",
            ),
            # Match with edit distance and both spacers
            (
                [("ACGT", MatchMethod.ALIGNMATCH, "ACGT", 2, "SPACER_1", "SPACER_2")],
                "BC1:ALIGNMATCH-ED2-spUp-spDown",
            ),
        ],
    )
    def test_barcode_match_history_to_status_string(
        self, attempts_data: list[tuple], expected_status: str
    ) -> None:
        """Test to_status_string with various attempt combinations."""
        history = BarcodeMatchHistory(bc_name="BC1")
        for candidate, method, match, ed, sp_up, sp_down in attempts_data:
            attempt = BarcodeMatchAttempt(
                candidate=candidate,
                method=method,
                match=match,
                edit_distance=ed,
                spacer_upstream=sp_up,
                spacer_downstream=sp_down,
            )
            history.record_attempt(attempt, success=(match is not None))

        assert_that(history.to_status_string()).is_equal_to(expected_status)

    # ===== ReadMatchResult =====

    @pytest.fixture
    def hydrop_chemistry(self) -> ChemistryHydrop:
        """Provide a HyDrop chemistry instance."""
        return ChemistryHydrop()

    def _make_successful_history(self, bc_name: str, barcode: str) -> BarcodeMatchHistory:
        """Helper to create a successful BarcodeMatchHistory with an exact match."""
        history = BarcodeMatchHistory(bc_name=bc_name)
        attempt = BarcodeMatchAttempt(
            candidate=barcode,
            method=MatchMethod.EXACTMATCH,
            match=barcode,
            read_idx=(0, len(barcode)),
        )
        history.record_attempt(attempt, success=True)
        return history

    def _make_failed_history(self, bc_name: str, candidate: str) -> BarcodeMatchHistory:
        """Helper to create a failed BarcodeMatchHistory."""
        history = BarcodeMatchHistory(bc_name=bc_name)
        attempt = BarcodeMatchAttempt(
            candidate=candidate,
            method=MatchMethod.EXACTMATCH,
        )
        history.record_attempt(attempt, success=False)
        return history

    def test_read_match_result_success_all_match(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that success is True when all barcode components matched."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.success).is_true()

    def test_read_match_result_failure_partial_match(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that success is False when one barcode component failed."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_failed_history("BC1", "ZZZZZZZZZZ"),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.success).is_false()

    def test_read_match_result_is_perfect_all_exact(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that is_perfect is True when all components matched via EXACTMATCH."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.is_perfect).is_true()

    def test_read_match_result_is_not_perfect_with_kmer_match(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that is_perfect is False when a component matched via a non-exact method."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc3_history = BarcodeMatchHistory(bc_name="BC3")
        bc3_attempt = BarcodeMatchAttempt(
            candidate=whitelists["BC3"][0],
            method=MatchMethod.KMERMATCH,
            match=whitelists["BC3"][0],
        )
        bc3_history.record_attempt(bc3_attempt, success=True)

        bc_results = [
            bc3_history,
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.success).is_true()
        assert_that(result.is_perfect).is_false()

    def test_read_match_result_full_barcode_dev_case1(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test full_barcode construction using verified barcodes from Case 1 (WL_MATCH)."""
        # Case 1 variable regions: BC3=CAGTGTGGAA, BC2=ACGGTGGACT, BC1=GAACAGTAGT
        # Concatenation order is read structure order: BC3 + BC2 + BC1
        bc_results = [
            self._make_successful_history("BC3", "CAGTGTGGAA"),
            self._make_successful_history("BC2", "ACGGTGGACT"),
            self._make_successful_history("BC1", "GAACAGTAGT"),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.full_barcode).is_equal_to("CAGTGTGGAAACGGTGGACTGAACAGTAGT")

    def test_read_match_result_full_barcode_none_on_failure(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that full_barcode is None when not all barcodes matched."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_failed_history("BC2", "ZZZZZZZZZZ"),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        assert_that(result.full_barcode).is_none()

    def test_read_match_result_get_annotated_readname_success(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that annotated readname contains SUCCESS:PERFECT and the full barcode on success."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        annotated = result.get_annotated_readname()
        assert_that(annotated).contains("SUCCESS:PERFECT")
        assert_that(annotated).starts_with("test_read|")
        assert_that(annotated).contains(result.full_barcode)

    def test_read_match_result_get_annotated_readname_failure(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that annotated readname contains FAIL when matching was unsuccessful."""
        bc_results = [
            self._make_failed_history("BC3", "ZZZZZZZZZZ"),
            self._make_failed_history("BC2", "ZZZZZZZZZZ"),
            self._make_failed_history("BC1", "ZZZZZZZZZZ"),
        ]
        result = ReadMatchResult(
            read_name="test_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )
        annotated = result.get_annotated_readname()
        assert_that(annotated).contains("FAIL")
        assert_that(annotated).does_not_contain("SUCCESS")


class TestBarcodeExtractorUtils:
    """Tests for barcode utility functions: hamming_distance and edit_distance."""

    # ===== hamming_distance =====

    def test_hamming_distance_identical(self) -> None:
        """Test that identical arrays have hamming distance 0."""
        a = np.array([1, 2, 3, 4])
        assert_that(hamming_distance(a, a.copy())).is_equal_to(0)

    def test_hamming_distance_one_mismatch(self) -> None:
        """Test that arrays differing at one position have hamming distance 1."""
        a = np.array([1, 2, 3, 4])
        b = np.array([1, 2, 3, 5])
        assert_that(hamming_distance(a, b)).is_equal_to(1)

    def test_hamming_distance_all_different(self) -> None:
        """Test that completely different arrays have hamming distance equal to length."""
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        assert_that(hamming_distance(a, b)).is_equal_to(4)

    def test_hamming_distance_unequal_length_raises(self) -> None:
        """Test that arrays of different lengths raise ValueError."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="equal length"):
            hamming_distance(a, b)

    def test_hamming_distance_non_1d_raises(self) -> None:
        """Test that non-1D arrays raise ValueError."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D"):
            hamming_distance(a, b)

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1, 1, 1, 1], [1, 1, 1, 1], 0),
            ([1, 2, 3, 4], [1, 2, 3, 5], 1),
            ([1, 2, 3, 4], [4, 3, 2, 1], 4),
            ([0], [1], 1),
            ([0], [0], 0),
            ([1, 0, 1, 0, 1], [0, 1, 0, 1, 0], 5),
            ([1, 2, 3, 4, 5], [1, 2, 0, 4, 5], 1),
        ],
    )
    def test_hamming_distance_parametrized(
        self, a: list[int], b: list[int], expected: int
    ) -> None:
        """Test hamming_distance with various input combinations."""
        assert_that(hamming_distance(np.array(a), np.array(b))).is_equal_to(expected)

    def test_hamming_distance_with_dna_bytes(self) -> None:
        """Test hamming distance using byte arrays representing DNA sequences."""
        a = np.frombuffer(b"ACGT", dtype=np.byte)
        b = np.frombuffer(b"ACGA", dtype=np.byte)
        assert_that(hamming_distance(a, b)).is_equal_to(1)

    # ===== edit_distance =====

    def test_edit_distance_identical(self) -> None:
        """Test that identical sequences have edit distance 0."""
        assert_that(edit_distance("ACGT", "ACGT")).is_equal_to(0)

    def test_edit_distance_n_wildcard_matches_any(self) -> None:
        """Test that N matches any character with cost 0 when n_matches_any is True."""
        assert_that(edit_distance("ACGT", "ACNT")).is_equal_to(0)

    def test_edit_distance_n_wildcard_disabled(self) -> None:
        """Test that N is treated as a regular character when n_matches_any is False."""
        assert_that(edit_distance("ACGT", "ACNT", n_matches_any=False)).is_equal_to(1)

    def test_edit_distance_classic_kitten_sitting(self) -> None:
        """Test the classic kitten/sitting example from the docstring."""
        assert_that(edit_distance("kitten", "sitting")).is_equal_to(3)

    def test_edit_distance_single_insertion(self) -> None:
        """Test that a single insertion results in edit distance 1."""
        assert_that(edit_distance("ACGT", "ACGGT")).is_equal_to(1)

    def test_edit_distance_single_deletion(self) -> None:
        """Test that a single deletion results in edit distance 1."""
        assert_that(edit_distance("ACGT", "ACT")).is_equal_to(1)

    def test_edit_distance_empty_strings(self) -> None:
        """Test edit distance with empty strings."""
        assert_that(edit_distance("", "")).is_equal_to(0)
        assert_that(edit_distance("ACGT", "")).is_equal_to(4)
        assert_that(edit_distance("", "ACGT")).is_equal_to(4)

    @pytest.mark.parametrize(
        "seq1, seq2, n_matches_any, expected",
        [
            ("ACGT", "ACGT", True, 0),
            ("ACGT", "ACGA", True, 1),
            ("ACGT", "ACNT", True, 0),
            ("ACGT", "ANNT", True, 0),
            ("ACGT", "NNNN", True, 0),
            ("ACGT", "NNNN", False, 4),
            ("ACGT", "ACNT", False, 1),
            ("A", "T", True, 1),
            ("ACGTACGT", "ACGTACGT", True, 0),
            ("ACGTACGT", "ACGTNCGT", True, 0),
            ("ACGT", "AACGT", True, 1),
            ("ACGT", "AGT", True, 1),
            ("ABC", "AEC", True, 1),
            ("ABC", "ADC", True, 1),
        ],
    )
    def test_edit_distance_parametrized(
        self, seq1: str, seq2: str, n_matches_any: bool, expected: int
    ) -> None:
        """Test edit_distance with various sequence combinations and N-wildcard settings."""
        assert_that(edit_distance(seq1, seq2, n_matches_any=n_matches_any)).is_equal_to(expected)

    def test_edit_distance_symmetry(self) -> None:
        """Test that edit distance is symmetric: d(a,b) == d(b,a)."""
        assert_that(edit_distance("ACGT", "TGCA")).is_equal_to(edit_distance("TGCA", "ACGT"))

    def test_edit_distance_multiple_n_wildcards(self) -> None:
        """Test edit distance with multiple N wildcards in both sequences."""
        assert_that(edit_distance("ANGT", "ACNT")).is_equal_to(0)


class TestHybridExtractor:
    """Tests for the HybridExtractor class."""

    @pytest.fixture
    def hydrop_chemistry(self) -> ChemistryHydrop:
        """Provide a HyDrop chemistry instance."""
        return ChemistryHydrop()

    @pytest.fixture
    def hydrop_matchers(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> dict[MatchMethod, dict[str, FixedPositionMatcher]]:
        """Build a matchers dict with FixedPositionMatcher for each barcode component."""
        whitelists = hydrop_chemistry.barcode_whitelists
        fixed_matchers: dict[str, FixedPositionMatcher] = {}
        for comp in hydrop_chemistry.read_structure.components:
            if comp.is_barcode:
                fixed_matchers[comp.name] = FixedPositionMatcher(
                    whitelist=whitelists[comp.name],
                    barcode_component=comp,
                    chemistry=hydrop_chemistry,
                )
        return {MatchMethod.EXACTMATCH: fixed_matchers}

    @pytest.fixture
    def hybrid_extractor(
        self,
        hydrop_chemistry: ChemistryHydrop,
        hydrop_matchers: dict[str, dict[str, FixedPositionMatcher]],
    ) -> HybridExtractor:
        """Provide a HybridExtractor instance configured with HyDrop chemistry."""
        return HybridExtractor(chemistry=hydrop_chemistry, matchers=hydrop_matchers)

    # ===== process_read =====

    def test_process_read_returns_read_match_result(
        self, hybrid_extractor: HybridExtractor
    ) -> None:
        """Test that process_read returns a ReadMatchResult instance."""
        result = hybrid_extractor.process_read("test_read", "A" * 52, "I" * 52)
        assert_that(result).is_instance_of(ReadMatchResult)

    def test_process_read_preserves_read_name(self, hybrid_extractor: HybridExtractor) -> None:
        """Test that the read name is preserved in the result."""
        result = hybrid_extractor.process_read("my_read_name", "A" * 52, "I" * 52)
        assert_that(result.read_name).is_equal_to("my_read_name")

    def test_process_read_preserves_read_and_qual(self, hybrid_extractor: HybridExtractor) -> None:
        """Test that the read sequence and quality string are preserved in the result."""
        seq = "A" * 52
        qual = "I" * 52
        result = hybrid_extractor.process_read("test", seq, qual)
        assert_that(result.read).is_equal_to(seq)
        assert_that(result.qual).is_equal_to(qual)

    def test_process_read_has_three_bc_results(self, hybrid_extractor: HybridExtractor) -> None:
        """Test that the result contains exactly three BarcodeMatchHistory entries (BC3, BC2, BC1)."""
        result = hybrid_extractor.process_read("test", "A" * 52, "I" * 52)
        assert_that(result.bc_results).is_length(3)
        bc_names = [bc.bc_name for bc in result.bc_results]
        assert_that(bc_names).is_equal_to(["BC3", "BC2", "BC1"])

    def test_process_read_random_sequence_no_match(
        self, hybrid_extractor: HybridExtractor
    ) -> None:
        """Test that a random DNA sequence produces no matches."""
        random_seq = "ATATATAT" * 7  # 56bp of alternating AT — not in any whitelist
        result = hybrid_extractor.process_read("random", random_seq, "I" * len(random_seq))
        assert_that(result.success).is_false()

    def test_process_read_full_barcode(self, hybrid_extractor: HybridExtractor) -> None:
        """Test that Case 1 (WL_MATCH) produces the correct full barcode."""
        seq = "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT"
        result = hybrid_extractor.process_read("case1", seq, "I" * len(seq))
        assert_that(result.full_barcode).is_equal_to("CAGTGTGGAAACGGTGGACTGAACAGTAGT")
        assert_that(result.is_perfect).is_true()

    def test_process_read_annotated_readname(self, hybrid_extractor: HybridExtractor) -> None:
        """Test that Case 1 produces a SUCCESS:PERFECT annotated readname."""
        seq = "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT"
        result = hybrid_extractor.process_read("case1", seq, "I" * len(seq))
        annotated = result.get_annotated_readname()
        assert_that(annotated).contains("SUCCESS:PERFECT")
        assert_that(annotated).contains("CAGTGTGGAAACGGTGGACTGAACAGTAGT")

    @pytest.mark.parametrize(
        "case_name, seq, expected_success, expected_bc3, expected_bc2, expected_bc1",
        [
            # Case 1 (WL_MATCH): all 3 BCs match
            (
                "WL_MATCH",
                "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT",
                True,
                True,
                True,
                True,
            ),
            # Case 2 (INDEL_FAIL): BC3 matches, BC2/BC1 don't
            (
                "INDEL_FAIL",
                "TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA",
                False,
                True,
                False,
                False,
            ),
            # Case 3 (SUB_FAIL): BC3+BC2 match, BC1 substitution error
            (
                "SUB_FAIL",
                "GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC",
                False,
                True,
                True,
                False,
            ),
            # Case 5 (SPC_NOTFND): BC3+BC2 match, BC1 doesn't
            (
                "SPC_NOTFND",
                "TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT",
                False,
                True,
                True,
                False,
            ),
            # Case 6 (SEVERE_INDEL): BC3 matches, BC2+BC1 don't
            (
                "SEVERE_INDEL",
                "TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC",
                False,
                True,
                False,
                False,
            ),
        ],
    )
    def test_process_read_dev_sequences(
        self,
        case_name: str,
        seq: str,
        expected_success: bool,
        expected_bc3: bool,
        expected_bc2: bool,
        expected_bc1: bool,
        hybrid_extractor: HybridExtractor,
    ) -> None:
        """Test process_read using HyDrop barcode read sequences with known match outcomes."""
        result = hybrid_extractor.process_read(case_name, seq, "I" * len(seq))

        assert_that(result.success).is_equal_to(expected_success)

        bc_success = {bc.bc_name: bc.success for bc in result.bc_results}
        assert_that(bc_success["BC3"]).is_equal_to(expected_bc3)
        assert_that(bc_success["BC2"]).is_equal_to(expected_bc2)
        assert_that(bc_success["BC1"]).is_equal_to(expected_bc1)

    def test_process_read_missing_matcher_raises(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that a matchers dict missing a barcode component raises ValueError."""
        # Build matchers with BC3 missing
        whitelists = hydrop_chemistry.barcode_whitelists
        incomplete_matchers: dict[str, FixedPositionMatcher] = {}
        for comp in hydrop_chemistry.read_structure.components:
            if comp.is_barcode and comp.name != "BC3":
                incomplete_matchers[comp.name] = FixedPositionMatcher(
                    whitelist=whitelists[comp.name],
                    barcode_component=comp,
                    chemistry=hydrop_chemistry,
                )

        extractor = HybridExtractor(
            chemistry=hydrop_chemistry, matchers={MatchMethod.EXACTMATCH: incomplete_matchers}
        )

        with pytest.raises(ValueError, match="BC3"):
            extractor.process_read("test", "A" * 52, "I" * 52)
