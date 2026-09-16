"""
Tests for barcode extraction reporting module: OverallStats, PerBarcodeStats, and ExtractionStats.
"""

from collections import Counter

import pytest
from assertpy import assert_that

from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    BarcodeMatchHistory,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.extraction_reporting import (
    BARCODE_RANK_MAX_POINTS,
    ExtractionStats,
    ExtractionStatsAccumulator,
    OverallStats,
    PerBarcodeStats,
    log_spaced_ranks,
    to_mqc_barcode_rank,
)
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME


def build_stats(results, matchers) -> ExtractionStats:
    """Replay results through ExtractionStatsAccumulator and finalize."""
    acc = ExtractionStatsAccumulator(matchers)
    for r in results:
        acc.update(r)
    return acc.finalize()


class TestOverallStats:
    """Tests for OverallStats dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test that OverallStats can be created with all required fields."""
        top_10 = [("barcode1", 100), ("barcode2", 50)]
        stats = OverallStats(
            total_reads=1000,
            perfect=800,
            corrok=150,
            fail=50,
            top_10_barcodes=top_10,
        )

        assert_that(stats.total_reads).is_equal_to(1000)
        assert_that(stats.perfect).is_equal_to(800)
        assert_that(stats.corrok).is_equal_to(150)
        assert_that(stats.fail).is_equal_to(50)
        assert_that(stats.top_10_barcodes).is_equal_to(top_10)

    def test_frozen_dataclass(self) -> None:
        """Test that OverallStats is frozen and cannot be modified after creation."""
        stats = OverallStats(
            total_reads=100,
            perfect=80,
            corrok=15,
            fail=5,
            top_10_barcodes=[],
        )

        with pytest.raises(AttributeError):
            stats.total_reads = 200


class TestPerBarcodeStats:
    """Tests for PerBarcodeStats dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test that PerBarcodeStats can be created with all fields including optional ones."""
        from collections import Counter

        stats = PerBarcodeStats(
            bc_name="BC1",
            method=MatchMethod.EXACTMATCH,
            attempts=100,
            success=95,
            fail=5,
            edit_distance_dist=Counter({0: 95}),
            reads_w_ambiguous_match=2,
            spacer_present=10,
        )

        assert_that(stats.bc_name).is_equal_to("BC1")
        assert_that(stats.method).is_equal_to(MatchMethod.EXACTMATCH)
        assert_that(stats.attempts).is_equal_to(100)
        assert_that(stats.success).is_equal_to(95)
        assert_that(stats.fail).is_equal_to(5)
        assert_that(stats.edit_distance_dist).is_equal_to(Counter({0: 95}))
        assert_that(stats.reads_w_ambiguous_match).is_equal_to(2)
        assert_that(stats.spacer_present).is_equal_to(10)

    def test_creation_without_optional_fields(self) -> None:
        """Test that PerBarcodeStats can be created without optional fields."""
        stats = PerBarcodeStats(
            bc_name="BC2",
            method=MatchMethod.KMERMATCH,
            attempts=50,
            success=40,
            fail=10,
            edit_distance_dist=None,
            reads_w_ambiguous_match=0,
            spacer_present=0,
        )

        assert_that(stats.edit_distance_dist).is_none()
        assert_that(stats.reads_w_ambiguous_match).is_equal_to(0)

    def test_frozen_dataclass(self) -> None:
        """Test that PerBarcodeStats is frozen and cannot be modified after creation."""
        from collections import Counter

        stats = PerBarcodeStats(
            bc_name="BC1",
            method=MatchMethod.EXACTMATCH,
            attempts=100,
            success=95,
            fail=5,
            edit_distance_dist=Counter(),
            reads_w_ambiguous_match=0,
            spacer_present=0,
        )

        with pytest.raises(AttributeError):
            stats.bc_name = "BC2"


class TestExtractionStats:
    """Tests for ExtractionStats, ExtractionStatsAccumulator aggregation, and report generation."""

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

    def _make_kmer_history(
        self, bc_name: str, barcode: str, edit_dist: int = 1
    ) -> BarcodeMatchHistory:
        """Helper to create a BarcodeMatchHistory with a kmer match and edit distance."""
        history = BarcodeMatchHistory(bc_name=bc_name)
        attempt1 = BarcodeMatchAttempt(
            candidate=barcode,
            method=MatchMethod.EXACTMATCH,
        )
        history.record_attempt(attempt1, success=False)
        attempt2 = BarcodeMatchAttempt(
            candidate=barcode,
            method=MatchMethod.KMERMATCH,
            match=barcode,
            edit_distance=edit_dist,
        )
        history.record_attempt(attempt2, success=True)
        return history

    def _make_sample_extraction_stats(self) -> ExtractionStats:
        """Helper to create a sample ExtractionStats instance for report testing."""
        from collections import Counter

        overall = OverallStats(
            total_reads=1000,
            perfect=800,
            corrok=150,
            fail=50,
            top_10_barcodes=[
                ("CAGTGTGGAAACGGTGGACTGAACAGTAGT", 100),
                ("AAAAAAAAAAACGGTGGACTGAACAGTAGT", 50),
            ],
        )

        per_barcode = [
            PerBarcodeStats(
                bc_name="BC3",
                method=MatchMethod.EXACTMATCH,
                attempts=1000,
                success=950,
                fail=50,
                edit_distance_dist=Counter({0: 950}),
                reads_w_ambiguous_match=5,
                spacer_present=100,
            ),
            PerBarcodeStats(
                bc_name="BC2",
                method=MatchMethod.EXACTMATCH,
                attempts=1000,
                success=900,
                fail=100,
                edit_distance_dist=Counter({0: 900}),
                reads_w_ambiguous_match=10,
                spacer_present=80,
            ),
            PerBarcodeStats(
                bc_name="BC1",
                method=MatchMethod.EXACTMATCH,
                attempts=1000,
                success=920,
                fail=80,
                edit_distance_dist=Counter({0: 920}),
                reads_w_ambiguous_match=8,
                spacer_present=90,
            ),
        ]

        return ExtractionStats(
            overall=overall,
            per_barcode=per_barcode,
            bc_names=["BC3", "BC2", "BC1"],
        )

    # ===== ExtractionStats construction and accumulator-driven aggregation =====

    def test_accumulator_creation(self) -> None:
        """Test that ExtractionStats can be constructed directly."""
        from collections import Counter

        overall = OverallStats(
            total_reads=100,
            perfect=80,
            corrok=15,
            fail=5,
            top_10_barcodes=[],
        )
        per_barcode = [
            PerBarcodeStats(
                bc_name="BC3",
                method=MatchMethod.EXACTMATCH,
                attempts=100,
                success=95,
                fail=5,
                edit_distance_dist=Counter({0: 95}),
                reads_w_ambiguous_match=0,
                spacer_present=10,
            ),
        ]

        stats = ExtractionStats(
            overall=overall,
            per_barcode=per_barcode,
            bc_names=["BC3"],
        )

        assert_that(stats.overall).is_equal_to(overall)
        assert_that(stats.per_barcode).is_length(1)
        assert_that(stats.bc_names).is_equal_to(["BC3"])

    def test_frozen_dataclass(self) -> None:
        """Test that ExtractionStats is frozen and cannot be modified after creation."""
        from collections import Counter

        overall = OverallStats(
            total_reads=100,
            perfect=80,
            corrok=15,
            fail=5,
            top_10_barcodes=[],
        )
        per_barcode = [
            PerBarcodeStats(
                bc_name="BC3",
                method=MatchMethod.EXACTMATCH,
                attempts=100,
                success=95,
                fail=5,
                edit_distance_dist=Counter({0: 95}),
                reads_w_ambiguous_match=0,
                spacer_present=10,
            ),
        ]
        stats = ExtractionStats(
            overall=overall,
            per_barcode=per_barcode,
            bc_names=["BC3", "BC2", "BC1"],
        )

        with pytest.raises(AttributeError):
            stats.bc_names = ["BC1"]

    def test_accumulator_finalize_raises_without_results(self) -> None:
        """ExtractionStatsAccumulator.finalize raises if no results were accumulated."""
        acc = ExtractionStatsAccumulator({})

        with pytest.raises(ValueError, match="No results were accumulated"):
            acc.finalize()

    def test_accumulator_single_perfect_match(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator with a single perfect match read."""
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

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats([result], matchers)

        assert_that(stats.overall.total_reads).is_equal_to(1)
        assert_that(stats.overall.perfect).is_equal_to(1)
        assert_that(stats.overall.corrok).is_equal_to(0)
        assert_that(stats.overall.fail).is_equal_to(0)

    def test_accumulator_single_failed_match(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator with a single failed match read."""
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

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats([result], matchers)

        assert_that(stats.overall.total_reads).is_equal_to(1)
        assert_that(stats.overall.perfect).is_equal_to(0)
        assert_that(stats.overall.corrok).is_equal_to(0)
        assert_that(stats.overall.fail).is_equal_to(1)

    def test_accumulator_corrected_match(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator with a corrected (non-perfect) match read."""
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_kmer_history("BC3", whitelists["BC3"][0], edit_dist=1),
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

        matchers = {MatchMethod.EXACTMATCH: {}, MatchMethod.KMERMATCH: {}}
        stats = build_stats([result], matchers)

        assert_that(stats.overall.total_reads).is_equal_to(1)
        assert_that(stats.overall.perfect).is_equal_to(0)
        assert_that(stats.overall.corrok).is_equal_to(1)
        assert_that(stats.overall.fail).is_equal_to(0)

    def test_accumulator_multiple_reads(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator with multiple reads having different outcomes."""
        whitelists = hydrop_chemistry.barcode_whitelists

        result1 = ReadMatchResult(
            read_name="read1",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", whitelists["BC3"][0]),
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        result2 = ReadMatchResult(
            read_name="read2",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_kmer_history("BC3", whitelists["BC3"][0], edit_dist=1),
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        result3 = ReadMatchResult(
            read_name="read3",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_failed_history("BC3", "ZZZZZZZZZZ"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),
                self._make_failed_history("BC1", "ZZZZZZZZZZ"),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}, MatchMethod.KMERMATCH: {}}
        stats = build_stats([result1, result2, result3], matchers)

        assert_that(stats.overall.total_reads).is_equal_to(3)
        assert_that(stats.overall.perfect).is_equal_to(1)
        assert_that(stats.overall.corrok).is_equal_to(1)
        assert_that(stats.overall.fail).is_equal_to(1)

    def test_accumulator_top_10_barcodes(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that the accumulator correctly identifies top 10 most frequent barcodes."""
        whitelists = hydrop_chemistry.barcode_whitelists

        results = []
        for i in range(5):
            result = ReadMatchResult(
                read_name=f"read{i}",
                read="A" * 50,
                qual="I" * 50,
                chemistry=hydrop_chemistry,
                bc_results=[
                    self._make_successful_history("BC3", whitelists["BC3"][0]),
                    self._make_successful_history("BC2", whitelists["BC2"][0]),
                    self._make_successful_history("BC1", whitelists["BC1"][0]),
                ],
            )
            results.append(result)

        for i in range(3):
            result = ReadMatchResult(
                read_name=f"read{i+5}",
                read="A" * 50,
                qual="I" * 50,
                chemistry=hydrop_chemistry,
                bc_results=[
                    self._make_successful_history("BC3", whitelists["BC3"][1]),
                    self._make_successful_history("BC2", whitelists["BC2"][1]),
                    self._make_successful_history("BC1", whitelists["BC1"][1]),
                ],
            )
            results.append(result)

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats(results, matchers)

        assert_that(stats.overall.top_10_barcodes).is_length(2)
        assert_that(stats.overall.top_10_barcodes[0][1]).is_equal_to(5)
        assert_that(stats.overall.top_10_barcodes[1][1]).is_equal_to(3)

    def test_accumulator_per_barcode_statistics(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that the accumulator generates correct per-barcode component statistics."""
        whitelists = hydrop_chemistry.barcode_whitelists

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", whitelists["BC3"][0]),
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats([result], matchers)

        assert_that(stats.per_barcode).is_length(3)

        bc3_stats = [s for s in stats.per_barcode if s.bc_name == "BC3"][0]
        assert_that(bc3_stats.method).is_equal_to(MatchMethod.EXACTMATCH)
        assert_that(bc3_stats.attempts).is_equal_to(1)
        assert_that(bc3_stats.success).is_equal_to(1)
        assert_that(bc3_stats.fail).is_equal_to(0)

    def test_accumulator_edit_distance_distribution(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that the accumulator correctly aggregates edit distance distributions."""
        whitelists = hydrop_chemistry.barcode_whitelists

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_kmer_history("BC3", whitelists["BC3"][0], edit_dist=1),
                self._make_kmer_history("BC2", whitelists["BC2"][0], edit_dist=2),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}, MatchMethod.KMERMATCH: {}}
        stats = build_stats([result], matchers)

        bc3_kmer_stats = [
            s
            for s in stats.per_barcode
            if s.bc_name == "BC3" and s.method == MatchMethod.KMERMATCH
        ][0]
        assert_that(bc3_kmer_stats.edit_distance_dist).is_not_none()
        assert_that(bc3_kmer_stats.edit_distance_dist[1]).is_equal_to(1)

    def test_accumulator_ambiguous_matches_counting(
        self, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that the accumulator correctly counts ambiguous matches per method."""
        whitelists = hydrop_chemistry.barcode_whitelists

        history = BarcodeMatchHistory(bc_name="BC3")
        attempt1 = BarcodeMatchAttempt(
            candidate="AAAAAAAAAA",
            method=MatchMethod.KMERMATCH,
            match=whitelists["BC3"][0],
            edit_distance=1,
        )
        attempt2 = BarcodeMatchAttempt(
            candidate="AAAAAAAAAA",
            method=MatchMethod.KMERMATCH,
            match=whitelists["BC3"][1],
            edit_distance=1,
        )
        history.record_attempt(attempt1, success=False)
        history.record_attempt(attempt2, success=True)

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                history,
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}, MatchMethod.KMERMATCH: {}}
        stats = build_stats([result], matchers)

        bc3_kmer_stats = [
            s
            for s in stats.per_barcode
            if s.bc_name == "BC3" and s.method == MatchMethod.KMERMATCH
        ][0]
        assert_that(bc3_kmer_stats.reads_w_ambiguous_match).is_equal_to(1)

    def test_accumulator_spacer_tracking(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that the accumulator correctly tracks spacer presence in matches."""
        whitelists = hydrop_chemistry.barcode_whitelists

        history = BarcodeMatchHistory(bc_name="BC3")
        attempt = BarcodeMatchAttempt(
            candidate=whitelists["BC3"][0],
            method=MatchMethod.EXACTMATCH,
            match=whitelists["BC3"][0],
            spacer_upstream="spacer1",
            spacer_downstream="spacer2",
        )
        history.record_attempt(attempt, success=True)

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                history,
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats([result], matchers)

        bc3_stats = [s for s in stats.per_barcode if s.bc_name == "BC3"][0]
        assert_that(bc3_stats.spacer_present).is_equal_to(1)

    def test_accumulator_skips_unused_methods(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test that the accumulator skips barcode/method combinations with no attempts."""
        whitelists = hydrop_chemistry.barcode_whitelists

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", whitelists["BC3"][0]),
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}, MatchMethod.ALIGNMATCH: {}}
        stats = build_stats([result], matchers)

        methods = [s.method for s in stats.per_barcode]
        assert_that(methods).contains(MatchMethod.EXACTMATCH)
        assert_that(methods).does_not_contain(MatchMethod.ALIGNMATCH)

    def test_accumulator_all_failed_reads(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator when all reads fail to match."""
        results = [
            ReadMatchResult(
                read_name=f"read{i}",
                read="A" * 50,
                qual="I" * 50,
                chemistry=hydrop_chemistry,
                bc_results=[
                    BarcodeMatchHistory(bc_name="BC3"),
                    BarcodeMatchHistory(bc_name="BC2"),
                    BarcodeMatchHistory(bc_name="BC1"),
                ],
            )
            for i in range(5)
        ]

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats(results, matchers)

        assert_that(stats.overall.total_reads).is_equal_to(5)
        assert_that(stats.overall.perfect).is_equal_to(0)
        assert_that(stats.overall.corrok).is_equal_to(0)
        assert_that(stats.overall.fail).is_equal_to(5)
        assert_that(stats.overall.top_10_barcodes).is_empty()

    def test_accumulator_no_spacers_present(self, hydrop_chemistry: ChemistryHydrop) -> None:
        """Test the accumulator when no spacers are present in matches."""
        whitelists = hydrop_chemistry.barcode_whitelists

        history = BarcodeMatchHistory(bc_name="BC3")
        attempt = BarcodeMatchAttempt(
            candidate=whitelists["BC3"][0],
            method=MatchMethod.EXACTMATCH,
            match=whitelists["BC3"][0],
            spacer_upstream=None,
            spacer_downstream=None,
        )
        history.record_attempt(attempt, success=True)

        result = ReadMatchResult(
            read_name="test",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                history,
                BarcodeMatchHistory(bc_name="BC2"),
                BarcodeMatchHistory(bc_name="BC1"),
            ],
        )

        matchers = {MatchMethod.EXACTMATCH: {}}
        stats = build_stats([result], matchers)

        bc3_stats = [s for s in stats.per_barcode if s.bc_name == "BC3"][0]
        assert_that(bc3_stats.spacer_present).is_equal_to(0)

    # ===== Report generation tests =====

    def test_get_report_contains_overall_stats(self) -> None:
        """Test that get_report includes overall statistics section."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Overall Barcode Extraction Stats")
        assert_that(report).contains("Total reads: 1000")
        assert_that(report).contains("Perfect matches: 800")
        assert_that(report).contains("Corrected matches: 150")
        assert_that(report).contains("Failed matches: 50")

    def test_get_report_contains_percentages(self) -> None:
        """Test that get_report includes percentage calculations."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("80.00%")
        assert_that(report).contains("15.00%")
        assert_that(report).contains("5.00%")

    def test_get_report_contains_top_barcodes(self) -> None:
        """Test that get_report includes top 10 barcode fractions section."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Top 10 barcode fractions:")
        assert_that(report).contains("CAGTGTGGAAACGGTGGACTGAACAGTAGT")
        assert_that(report).contains("AAAAAAAAAAACGGTGGACTGAACAGTAGT")

    def test_get_report_contains_per_barcode_stats(self) -> None:
        """Test that get_report includes per-barcode component statistics."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Per-Barcode Component Stats")
        assert_that(report).contains("## BC3")
        assert_that(report).contains("## BC2")
        assert_that(report).contains("## BC1")

    def test_get_report_contains_matching_method(self) -> None:
        """Test that get_report includes matching method information."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Matching method: EXACTMATCH")

    def test_get_report_contains_edit_distance_distribution(self) -> None:
        """Test that get_report includes edit distance distribution section."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Edit distance distribution")

    def test_get_report_contains_ambiguous_matches(self) -> None:
        """Test that get_report includes ambiguous match counts."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Ambiguous matches (reads):")

    def test_get_report_contains_spacer_info(self) -> None:
        """Test that get_report includes spacer presence information."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report()

        assert_that(report).contains("Matches with spacers present:")

    def test_get_report_without_run_details(self) -> None:
        """Test that get_report can exclude run details section."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report(include_run_details=False)

        assert_that(report).does_not_contain("Carmack version:")
        assert_that(report).does_not_contain("Report generated at:")
        assert_that(report).contains("Overall Barcode Extraction Stats")

    def test_get_report_with_run_details(self) -> None:
        """Test that get_report includes run details when requested."""
        stats = self._make_sample_extraction_stats()
        report = stats.get_report(include_run_details=True)

        assert_that(report).contains("Carmack version:")
        assert_that(report).contains("Report generated at:")

    def test_get_per_barcode_section_format(self) -> None:
        """Test the format of per-barcode section generation."""
        stats = self._make_sample_extraction_stats()
        bc3_stats = stats.per_barcode[0]
        section = stats.get_per_barcode_section(bc3_stats)

        assert_that(section).contains("Matching method: EXACTMATCH")
        assert_that(section).contains("Reads checked: 1000")
        assert_that(section).contains("Successful matches: 950")
        assert_that(section).contains("Failed matches: 50")
        assert_that(section).contains("Ambiguous matches (reads): 5")

    def test_get_run_details_format(self) -> None:
        """Test the format of run details section generation."""
        stats = self._make_sample_extraction_stats()
        details = stats.get_run_details()

        assert_that(details).contains("# Carmack version:")
        assert_that(details).contains("# Report generated at:")
        assert_that(details).matches(r"\d{4}-\d{2}-\d{2}")


class TestExtractionStatsMqcReporting:
    """Tests for the MultiQC custom-content payload methods on ExtractionStats."""

    SAMPLE_PREFIX = "SK123"

    @pytest.fixture
    def sample_extraction_stats(self) -> ExtractionStats:
        """Reuse TestExtractionStats' sample fixture rather than duplicating it."""
        return TestExtractionStats()._make_sample_extraction_stats()

    def make_stats_with_edit_distance_dists(
        self, edit_distance_dist: Counter[int] | None
    ) -> ExtractionStats:
        """Build an ExtractionStats whose per_barcode entries all share one edit_distance_dist."""
        overall = OverallStats(
            total_reads=100,
            perfect=80,
            corrok=15,
            fail=5,
            top_10_barcodes=[],
        )
        per_barcode = [
            PerBarcodeStats(
                bc_name=bc_name,
                method=MatchMethod.EXACTMATCH,
                attempts=100,
                success=95,
                fail=5,
                edit_distance_dist=edit_distance_dist,
                reads_w_ambiguous_match=0,
                spacer_present=0,
            )
            for bc_name in ("BC3", "BC2", "BC1")
        ]
        return ExtractionStats(
            overall=overall,
            per_barcode=per_barcode,
            bc_names=["BC3", "BC2", "BC1"],
        )

    # ===== to_mqc_general_stats =====

    def test_to_mqc_general_stats_has_generalstats_plot_type_and_id(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """to_mqc_general_stats returns a generalstats payload with a non-empty id."""
        payload = sample_extraction_stats.to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("generalstats")
        assert_that(payload["id"]).is_instance_of(str)
        assert_that(payload["id"]).is_not_empty()

    def test_to_mqc_general_stats_computes_percentages(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """to_mqc_general_stats computes 0-100 percentages from the overall and per-barcode counts."""
        payload = sample_extraction_stats.to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_perfect"]).is_equal_to(80.0)
        assert_that(data["pct_corrected"]).is_equal_to(15.0)
        assert_that(data["pct_failed"]).is_equal_to(5.0)
        assert_that(data["pct_ambiguous"]).is_equal_to(2.3)

    def test_to_mqc_general_stats_keys_data_by_prefix(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """The payload's data dict is keyed by exactly the prefix passed in."""
        payload = sample_extraction_stats.to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(list(payload["data"].keys())).is_equal_to([self.SAMPLE_PREFIX])

    def test_to_mqc_general_stats_attributes_its_columns_with_a_namespace(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """to_mqc_general_stats names carmack as its columns' source through ``namespace``."""
        payload = sample_extraction_stats.to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload).contains_entry({"namespace": CARMACK_PARENT_NAME})
        assert_that(payload).does_not_contain_key("parent_id")
        assert_that(payload).does_not_contain_key("parent_name")

    # ===== to_mqc_breakdown =====

    def test_to_mqc_breakdown_has_bargraph_plot_type_and_parent(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """to_mqc_breakdown returns a bargraph payload naming carmack's shared parent section."""
        payload = sample_extraction_stats.to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("bargraph")
        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    def test_to_mqc_breakdown_data_matches_overall_counts(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """The breakdown data holds raw perfect/corrected/failed counts for the prefix."""
        payload = sample_extraction_stats.to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(list(payload["data"].keys())).is_equal_to([self.SAMPLE_PREFIX])
        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(
            {"perfect": 800, "corrected": 150, "failed": 50}
        )

    # ===== to_mqc_edit_distance =====

    def test_to_mqc_edit_distance_has_linegraph_plot_type(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """to_mqc_edit_distance returns a linegraph payload when a distribution exists."""
        payload = sample_extraction_stats.to_mqc_edit_distance(self.SAMPLE_PREFIX)

        assert_that(payload).is_not_none()
        assert_that(payload["plot_type"]).is_equal_to("linegraph")

    def test_to_mqc_edit_distance_combines_per_barcode_counters(
        self, sample_extraction_stats: ExtractionStats
    ) -> None:
        """The edit distance data sums every per-barcode Counter into one combined Counter."""
        payload = sample_extraction_stats.to_mqc_edit_distance(self.SAMPLE_PREFIX)

        assert_that(list(payload["data"].keys())).is_equal_to([self.SAMPLE_PREFIX])
        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to({0: 2770})

    @pytest.mark.parametrize(
        "edit_distance_dist",
        [None, Counter()],
        ids=["none", "empty_counter"],
    )
    def test_to_mqc_edit_distance_returns_none_when_all_empty(
        self, edit_distance_dist: Counter[int] | None
    ) -> None:
        """When every per_barcode entry carries no edit distance data, the method returns None."""
        stats = self.make_stats_with_edit_distance_dists(edit_distance_dist)

        assert_that(stats.to_mqc_edit_distance(self.SAMPLE_PREFIX)).is_none()

    # ===== Shared prefix-keying contract =====

    @pytest.mark.parametrize("prefix", ["SK123", "another_sample_prefix"])
    def test_mqc_payloads_are_keyed_by_the_given_prefix(
        self, sample_extraction_stats: ExtractionStats, prefix: str
    ) -> None:
        """Every payload's data dict is keyed by exactly the prefix supplied, not a
        hard-coded sample name.

        The barcode rank builder is named here explicitly rather than picked up with
        the others: it is a module-level function, not a method, because the full
        barcode counts it plots never reach the finalized stats object. Enumerating
        the methods on a stats instance therefore cannot reach it.
        """
        general_stats_payload = sample_extraction_stats.to_mqc_general_stats(prefix)
        breakdown_payload = sample_extraction_stats.to_mqc_breakdown(prefix)
        edit_distance_payload = sample_extraction_stats.to_mqc_edit_distance(prefix)
        barcode_rank_payload = to_mqc_barcode_rank(prefix, Counter({"ACGT": 9, "TGCA": 4}))

        assert_that(list(general_stats_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(breakdown_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(edit_distance_payload["data"].keys())).is_equal_to([prefix])
        assert_that(list(barcode_rank_payload["data"].keys())).is_equal_to([prefix])


class TestLogSpacedRanks:
    """Tests for log_spaced_ranks, the rank downsampler behind the barcode rank curve."""

    LARGE_N = 1_000_000

    @pytest.fixture
    def large_ranks(self) -> list[int]:
        """Ranks for a barcode count far above the default budget, downsampled once per test."""
        return log_spaced_ranks(self.LARGE_N)

    # ===== Degenerate and small inputs =====

    @pytest.mark.parametrize("n", [0, -1, -1000], ids=["zero", "negative", "very_negative"])
    def test_log_spaced_ranks_returns_no_ranks_for_a_non_positive_n(self, n: int) -> None:
        """Test that a run with nothing to rank produces no ranks at all.

        The caller turns an empty result into a suppressed section, so this has
        to be empty rather than a degenerate one-point curve.
        """
        assert_that(log_spaced_ranks(n)).is_equal_to([])

    def test_log_spaced_ranks_returns_the_only_rank_for_a_single_barcode(self) -> None:
        """Test that a single observed barcode yields exactly rank 1."""
        assert_that(log_spaced_ranks(1)).is_equal_to([1])

    @pytest.mark.parametrize("n", [2, 7, 299, BARCODE_RANK_MAX_POINTS])
    def test_log_spaced_ranks_returns_every_rank_when_n_fits_the_budget(self, n: int) -> None:
        """Test that a barcode count within the point budget is plotted rank by rank.

        Downsampling a curve that already fits would throw away detail for no
        gain, so every rank from 1 to n is kept, the boundary case n ==
        max_points included.
        """
        assert_that(log_spaced_ranks(n)).is_equal_to(list(range(1, n + 1)))

    # ===== Downsampling a curve larger than the budget =====

    def test_log_spaced_ranks_stays_within_the_point_budget(self, large_ranks: list[int]) -> None:
        """Test that a million barcodes are reduced to at most max_points ranks."""
        assert_that(large_ranks).is_not_empty()
        assert_that(len(large_ranks)).is_less_than_or_equal_to(BARCODE_RANK_MAX_POINTS)

    def test_log_spaced_ranks_collapses_below_the_budget_through_deduplication(
        self, large_ranks: list[int]
    ) -> None:
        """Test that the downsampled ranks come out strictly shorter than the budget.

        The shortfall is deduplication at the low end: consecutive log steps
        there are far less than one rank apart, so several of them round to the
        same integer rank and collapse into one. That is why the budget is an
        upper bound and not an exact count. The property is asserted rather than
        an exact length, which would pin one particular spacing algorithm
        instead of the behaviour that matters.
        """
        assert_that(len(large_ranks)).described_as(
            "log-spaced ranks after deduplication"
        ).is_less_than(BARCODE_RANK_MAX_POINTS)

    def test_log_spaced_ranks_are_strictly_ascending_integers(
        self, large_ranks: list[int]
    ) -> None:
        """Test that the ranks ascend strictly, which also proves they are deduplicated.

        A repeated rank would plot the same barcode twice, and a rank out of
        order would draw the curve backwards on a log x-axis.
        """
        for rank in large_ranks:
            assert_that(rank).is_instance_of(int)
        assert_that(large_ranks).is_equal_to(sorted(set(large_ranks)))

    def test_log_spaced_ranks_keep_both_endpoints(self, large_ranks: list[int]) -> None:
        """Test that the most and least abundant barcodes are always plotted.

        The endpoints carry the two numbers a reader takes off this chart: the
        top barcode's depth and the total number of barcodes observed. Log
        spacing must never round either of them away.
        """
        assert_that(large_ranks[0]).is_equal_to(1)
        assert_that(large_ranks[-1]).is_equal_to(self.LARGE_N)

    # ===== max_points =====

    @pytest.mark.parametrize("max_points", [1, 0, -5], ids=["one", "zero", "negative"])
    def test_log_spaced_ranks_rejects_a_budget_below_two_points(self, max_points: int) -> None:
        """Test that a budget too small to hold both endpoints raises.

        Fewer than two points cannot carry rank 1 and rank n at once, so there
        is no curve to draw and a silent truncation would misreport the run.
        """
        with pytest.raises(ValueError):
            log_spaced_ranks(1000, max_points=max_points)

    @pytest.mark.parametrize("max_points", [2, 10, 64])
    def test_log_spaced_ranks_honours_a_non_default_budget(self, max_points: int) -> None:
        """Test that a caller-supplied budget, not the module default, bounds the result."""
        ranks = log_spaced_ranks(10_000, max_points=max_points)

        assert_that(len(ranks)).is_less_than_or_equal_to(max_points)
        assert_that(ranks[0]).is_equal_to(1)
        assert_that(ranks[-1]).is_equal_to(10_000)
        assert_that(ranks).is_equal_to(sorted(set(ranks)))


class TestToMqcBarcodeRank:
    """Tests for to_mqc_barcode_rank, the MultiQC payload for the barcode rank curve."""

    SAMPLE_PREFIX = "SK123"

    @staticmethod
    def make_barcode_counts(n: int) -> Counter[str]:
        """Build n barcodes whose counts strictly decrease, so the rank order is unambiguous."""
        return Counter({f"BC{i:05d}": (n - i) * 10 for i in range(n)})

    # ===== Data shape =====

    def test_to_mqc_barcode_rank_renders_data_as_pairs_rather_than_a_mapping(self) -> None:
        """Test that the series is a list of points, asserted by type, and not a mapping.

        MultiQC's custom-content linegraph path renders an ``{x: y}`` mapping as
        lexically sorted strings, putting '10' between '1' and '2' -- fatal for a
        curve spanning rank 1 to rank one million. A list of ``[x, y]`` pairs is
        a first-class input shape MultiQC builds the mapping from itself, which
        keeps the ranks as integers.
        """
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))
        series = payload["data"][self.SAMPLE_PREFIX]

        assert_that(series).is_instance_of(list)
        assert_that(isinstance(series, dict)).described_as(
            "series rendered as a mapping"
        ).is_false()

    def test_to_mqc_barcode_rank_renders_every_point_as_a_two_element_list(self) -> None:
        """Test that each point is a list of exactly two values, not a tuple.

        MultiQC inspects the first point with ``isinstance(x_to_y[0], list)`` to
        decide it was handed pairs. A tuple fails that check in memory even
        though it would round-trip through JSON as an array, so the payload has
        to hold real lists before it is ever serialised.
        """
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))
        series = payload["data"][self.SAMPLE_PREFIX]

        for point in series:
            assert_that(point).described_as(f"point {point}").is_instance_of(list)
            assert_that(point).described_as(f"point {point}").is_length(2)

    def test_to_mqc_barcode_rank_x_values_are_strictly_ascending_integers(self) -> None:
        """Test that the ranks plotted are integers that ascend strictly."""
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(500))
        ranks = [x for x, _ in payload["data"][self.SAMPLE_PREFIX]]

        for rank in ranks:
            assert_that(rank).is_instance_of(int)
        assert_that(ranks).is_equal_to(sorted(set(ranks)))

    def test_to_mqc_barcode_rank_y_values_never_increase_with_rank(self) -> None:
        """Test that counts fall off monotonically, which is what makes it a rank curve.

        The counts are sorted descending before ranking, so a rise anywhere in
        the series would mean the sort or the rank lookup is misaligned.
        """
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(500))
        counts = [y for _, y in payload["data"][self.SAMPLE_PREFIX]]

        for earlier, later in zip(counts, counts[1:]):
            assert_that(later).is_less_than_or_equal_to(earlier)

    def test_to_mqc_barcode_rank_starts_at_the_most_abundant_barcode(self) -> None:
        """Test that the first point is rank 1 paired with the highest count observed."""
        barcode_counts = self.make_barcode_counts(500)
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts)

        assert_that(payload["data"][self.SAMPLE_PREFIX][0]).is_equal_to(
            [1, max(barcode_counts.values())]
        )

    def test_to_mqc_barcode_rank_ends_at_the_number_of_observed_barcodes(self) -> None:
        """Test that the last rank plotted is the count of barcodes actually seen."""
        barcode_counts = self.make_barcode_counts(500)
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts)
        last_rank, _ = payload["data"][self.SAMPLE_PREFIX][-1]

        assert_that(last_rank).is_equal_to(500)

    def test_to_mqc_barcode_rank_renders_a_single_barcode_as_one_point(self) -> None:
        """Test that one observed barcode yields exactly one point, at rank 1."""
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, Counter({"ACGTACGTAC": 42}))

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to([[1, 42]])

    # ===== Zero counts and empty input =====

    def test_to_mqc_barcode_rank_excludes_zero_count_barcodes(self) -> None:
        """Test that barcodes counted zero times are left out of the curve entirely.

        A Counter can carry explicit zero entries, and a zero-depth barcode was
        never observed: plotting it would extend the tail with barcodes the run
        never saw and drag the last rank past the true total.
        """
        barcode_counts = Counter({"AAAA": 9, "CCCC": 4, "GGGG": 0, "TTTT": 0})

        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts)
        series = payload["data"][self.SAMPLE_PREFIX]

        assert_that(series).is_equal_to([[1, 9], [2, 4]])

    def test_to_mqc_barcode_rank_returns_none_for_an_empty_counter(self) -> None:
        """Test that a run that matched no full barcode renders no payload at all.

        Returning an empty series instead would take down the whole report
        rather than one section: MultiQC indexes the first point unguarded, so
        an empty pair list raises IndexError. The writer's own emptiness check
        does not catch it either, because a dict holding an empty list is
        truthy -- so the suppression has to happen here, as ``None``.
        """
        assert_that(to_mqc_barcode_rank(self.SAMPLE_PREFIX, Counter())).is_none()

    def test_to_mqc_barcode_rank_returns_none_when_every_count_is_zero(self) -> None:
        """Test that a counter holding only zero-count barcodes also renders nothing.

        Nothing survives the non-zero filter, so this reaches the same empty
        series the report cannot survive, by a different route.
        """
        barcode_counts = Counter({"AAAA": 0, "CCCC": 0})

        assert_that(to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts)).is_none()

    # ===== Payload identity and MultiQC config =====

    def test_to_mqc_barcode_rank_has_linegraph_plot_type_and_id(self) -> None:
        """Test that the payload declares the id its file and section are named from."""
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))

        assert_that(payload["id"]).is_equal_to("carmack_extraction_barcode_rank")
        assert_that(payload["plot_type"]).is_equal_to("linegraph")

    def test_to_mqc_barcode_rank_pconfig_names_the_plot_and_axes(self) -> None:
        """Test that the plot config carries its own id, a title and both axis labels."""
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))
        pconfig = payload["pconfig"]

        assert_that(pconfig["id"]).is_equal_to("carmack_extraction_barcode_rank_plot")
        assert_that(pconfig["title"]).is_instance_of(str)
        assert_that(pconfig["title"]).is_not_empty()
        assert_that(pconfig["xlab"]).is_not_empty()
        assert_that(pconfig["ylab"]).is_not_empty()

    def test_to_mqc_barcode_rank_plots_both_axes_logarithmically(self) -> None:
        """Test that both axes are logarithmic, which is what makes the knee readable.

        Rank spans orders of magnitude and so does depth; on linear axes the
        whole curve collapses against the origin.
        """
        pconfig = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))["pconfig"]

        assert_that(pconfig["xlog"]).is_true()
        assert_that(pconfig["ylog"]).is_true()

    @pytest.mark.parametrize("max_points", [None, 5, 12, 1000], ids=["default", "5", "12", "1000"])
    def test_to_mqc_barcode_rank_sets_smooth_points_above_the_budget_in_force(
        self, max_points: int | None
    ) -> None:
        """Test that the re-binning threshold always sits above the points actually sent.

        MultiQC re-bins any series longer than this threshold onto uniform index
        spacing, which would flatten exactly the log spacing this chart exists
        to produce -- and ``smooth_points: null`` does not disable it. Keeping
        the threshold derived from the budget in force, rather than a literal,
        means a caller raising max_points cannot silently walk back into
        re-binning.
        """
        barcode_counts = self.make_barcode_counts(50)
        if max_points is None:
            payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts)
            budget = BARCODE_RANK_MAX_POINTS
        else:
            payload = to_mqc_barcode_rank(
                self.SAMPLE_PREFIX, barcode_counts, max_points=max_points
            )
            budget = max_points

        assert_that(payload["pconfig"]["smooth_points"]).is_greater_than(budget)

    def test_to_mqc_barcode_rank_nests_under_the_shared_carmack_parent(self) -> None:
        """Test that the chart attaches to Carmack's parent section, not a namespace.

        ``parent_id``/``parent_name`` are what nest a chart section under the
        one Carmack heading; ``namespace`` is the generalstats-only key, and is
        inert on this branch of MultiQC's parser.
        """
        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, self.make_barcode_counts(20))

        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)
        assert_that(payload).does_not_contain_key("namespace")
        assert_that(payload["section_name"]).is_not_empty()
        assert_that(payload["description"]).is_not_empty()

    # ===== Downsampling through the payload =====

    def test_to_mqc_barcode_rank_downsamples_more_barcodes_than_the_budget(self) -> None:
        """Test that a curve longer than the budget is thinned but keeps both endpoints.

        Fifty barcodes into a five-point budget proves the payload runs its
        counts through the downsampler rather than emitting one point per
        barcode, while the first and last points stay exact so the chart still
        reports the top barcode's depth and the true barcode total.
        """
        barcode_counts = self.make_barcode_counts(50)

        payload = to_mqc_barcode_rank(self.SAMPLE_PREFIX, barcode_counts, max_points=5)
        series = payload["data"][self.SAMPLE_PREFIX]

        assert_that(len(series)).is_less_than_or_equal_to(5)
        assert_that(len(series)).is_less_than(50)
        assert_that(series[0]).is_equal_to([1, max(barcode_counts.values())])
        assert_that(series[-1]).is_equal_to([50, min(barcode_counts.values())])
