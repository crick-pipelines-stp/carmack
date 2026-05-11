"""
Tests for barcode extraction pipeline: HybridExtractor, dataclasses, and utility functions.
"""

from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from assertpy import assert_that

from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.barcode.barcode_utils import edit_distance, hamming_distance, make_barcode_rank_plot
from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchAttempt,
    BarcodeMatchHistory,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.hybrid_extractor import HybridExtractor
from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponentType


R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"


class TestBarcodeExtractor:
    """Tests for barcode extractor class with Hydrop chemistry."""

    @pytest.fixture(scope="class")
    def barcode_extractor(self) -> BarcodeExtractor:
        """Provide a BarcodeExtractor instance initialized with the test FASTQ and hydrop chemistry."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,  # Use single worker for testing
        )
        return extractor

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
        for comp in hydrop_chemistry.read_structure.get_components_by_type(
            ReadComponentType.BARCODE
        ):
            fixed_matchers[comp.name] = FixedPositionMatcher(
                whitelist=whitelists[comp.name],
                barcode_component=comp,
                chemistry=hydrop_chemistry,
            )
        return {MatchMethod.EXACTMATCH: fixed_matchers}

    # ===== Initialization Tests =====

    def test_init_stores_parameters(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that BarcodeExtractor stores initialization parameters correctly."""
        assert_that(barcode_extractor.chemistry_name).is_equal_to("hydrop")
        assert_that(barcode_extractor.kmer_size).is_equal_to(4)
        assert_that(barcode_extractor.n_workers).is_equal_to(1)

    def test_init_reads_total_reads(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that BarcodeExtractor reads total reads count from FASTQ."""
        assert_that(barcode_extractor.total_reads).is_greater_than(0)

    def test_init_raises_on_empty_fastq(self) -> None:
        """Test that BarcodeExtractor raises ValueError on empty FASTQ file."""
        # Create a temporary empty fastq
        import gzip
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fastq.gz", delete=False) as f:
            with gzip.open(f.name, "wt") as gz:
                gz.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="No reads found"):
                BarcodeExtractor(
                    fastq_file=temp_path,
                    chemistry_name="HyDrop",
                    n_workers=1,
                )
        finally:
            import os

            os.unlink(temp_path)

    def test_init_raises_on_invalid_chemistry(self) -> None:
        """Test that BarcodeExtractor raises ValueError on invalid chemistry name."""
        with pytest.raises(ValueError, match="not supported"):
            BarcodeExtractor(
                fastq_file=R1_PATH,
                chemistry_name="InvalidChemistry",
                n_workers=1,
            )

    def test_init_creates_chemistry_instance(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that BarcodeExtractor creates chemistry instance."""
        from carmack.chemistry.chemistry_hydrop import ChemistryHydrop

        assert_that(barcode_extractor.chemistry).is_instance_of(ChemistryHydrop)

    def test_init_loads_whitelists(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that BarcodeExtractor loads barcode whitelists."""
        assert_that(barcode_extractor.whitelists).is_not_empty()
        assert_that(barcode_extractor.whitelists).contains_key("BC1").contains_key(
            "BC2"
        ).contains_key("BC3")

    def test_init_with_custom_kmer_size(self) -> None:
        """Test initialization with custom kmer size."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            kmer_size=5,
            n_workers=1,
        )
        assert_that(extractor.kmer_size).is_equal_to(5)

    def test_init_with_custom_batch_size(self) -> None:
        """Test initialization with custom batch size."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            batch_size=100,
        )
        assert_that(extractor.batch_size).is_equal_to(100)

    def test_init_with_multiple_workers(self) -> None:
        """Test initialization with multiple workers."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=4,
        )
        assert_that(extractor.n_workers).is_equal_to(4)

    def test_init_with_fast_mode(self) -> None:
        """Test initialization with fast mode enabled."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            fast=True,
        )
        assert_that(extractor.fast).is_true()

    # ===== calc_batch_size Tests =====

    def test_calc_batch_size_respects_min(self) -> None:
        """Test that calc_batch_size respects MIN_READS_PER_BATCH."""
        # Use a chemistry with known small whitelist to speed up
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1000,  # Many workers, few reads per worker
        )
        assert_that(extractor.batch_size).is_greater_than_or_equal_to(10)  # MIN_READS_PER_BATCH

    def test_calc_batch_size_respects_max(self) -> None:
        """Test that calc_batch_size respects MAX_READS_PER_BATCH."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,  # Single worker gets all reads
        )
        assert_that(extractor.batch_size).is_less_than_or_equal_to(2500)  # MAX_READS_PER_BATCH

    @pytest.mark.parametrize(
        "n_workers,expected_batches",
        [
            (1, 1),  # Single worker, single batch (assuming <2500 reads)
            (2, 2),  # Two workers, two batches
            (4, 4),  # Four workers, four batches
        ],
    )
    def test_calc_batch_size_scales_with_workers(
        self, n_workers: int, expected_batches: int
    ) -> None:
        """Test batch size calculation scales with worker count."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=n_workers,
        )
        # For small files, batch size should be adjusted so we get expected_batches
        expected_batch_size = max(10, (extractor.total_reads + n_workers - 1) // n_workers)
        expected_batch_size = min(expected_batch_size, 2500)
        assert_that(extractor.batch_size).is_equal_to(expected_batch_size)

    # ===== init_matchers Tests =====

    def test_init_matchers_creates_all_matchers(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that init_matchers creates all three matcher types."""
        matchers = barcode_extractor.matchers
        assert_that(matchers).contains_key(MatchMethod.EXACTMATCH)
        assert_that(matchers).contains_key(MatchMethod.KMERMATCH)
        assert_that(matchers).contains_key(MatchMethod.ALIGNMATCH)

    def test_init_matchers_creates_matchers_for_all_barcodes(
        self, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that init_matchers creates matchers for all barcode components."""
        for method in [MatchMethod.EXACTMATCH, MatchMethod.KMERMATCH, MatchMethod.ALIGNMATCH]:
            assert_that(barcode_extractor.matchers[method]).contains_key("BC1")
            assert_that(barcode_extractor.matchers[method]).contains_key("BC2")
            assert_that(barcode_extractor.matchers[method]).contains_key("BC3")

    def test_init_matchers_uses_correct_kmer_size(self) -> None:
        """Test that init_matchers uses the configured kmer size."""
        from carmack.barcode.matchers.kmer_matcher import KmerMatcher

        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            kmer_size=5,
            n_workers=1,
        )
        kmer_matcher = extractor.matchers[MatchMethod.KMERMATCH]["BC3"]
        assert_that(kmer_matcher).is_instance_of(KmerMatcher)
        assert_that(getattr(kmer_matcher, "k")).is_equal_to(5)

    def test_init_matchers_matchers_are_different_types(
        self, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that each matcher type creates appropriate matcher instances."""
        from carmack.barcode.matchers.alignment_matcher import AlignmentMatcher
        from carmack.barcode.matchers.fixed_position_matcher import FixedPositionMatcher
        from carmack.barcode.matchers.kmer_matcher import KmerMatcher

        assert_that(barcode_extractor.matchers[MatchMethod.EXACTMATCH]["BC3"]).is_instance_of(
            FixedPositionMatcher
        )
        assert_that(barcode_extractor.matchers[MatchMethod.KMERMATCH]["BC3"]).is_instance_of(
            KmerMatcher
        )
        assert_that(barcode_extractor.matchers[MatchMethod.ALIGNMATCH]["BC3"]).is_instance_of(
            AlignmentMatcher
        )

    def test_init_matchers_skips_alignment_matchers_in_fast_mode(self) -> None:
        """Test that fast mode omits alignment matchers from the pipeline."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            fast=True,
        )

        assert_that(extractor.matchers).contains_key(MatchMethod.EXACTMATCH)
        assert_that(extractor.matchers).contains_key(MatchMethod.KMERMATCH)
        assert_that(extractor.matchers).does_not_contain_key(MatchMethod.ALIGNMATCH)

    # ===== generate_batches Tests =====

    def test_generate_batches_returns_list_of_tuples(
        self, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that generate_batches returns list of (name, seq, qual) tuples."""
        batches = barcode_extractor.generate_batches()
        assert_that(batches).is_instance_of(list)
        if batches:
            assert_that(batches[0]).is_instance_of(list)
            if batches[0]:
                assert_that(batches[0][0]).is_instance_of(tuple)
                assert_that(len(batches[0][0])).is_equal_to(3)  # (name, seq, qual)

    def test_generate_batches_respects_batch_size(self) -> None:
        """Test that generate_batches respects the configured batch size."""
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            batch_size=5,  # Small batch size for testing
        )
        batches = extractor.generate_batches()
        if len(batches) > 1:  # If multiple batches, check they're appropriately sized
            for batch in batches[:-1]:  # All but last should be full
                assert_that(len(batch)).is_less_than_or_equal_to(5)

    def test_generate_batches_contains_all_reads(
        self, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that generate_batches includes all reads from the file."""
        batches = barcode_extractor.generate_batches()
        total_reads_in_batches = sum(len(batch) for batch in batches)
        assert_that(total_reads_in_batches).is_equal_to(barcode_extractor.total_reads)

    def test_generate_batches_empty_last_batch_handled(self) -> None:
        """Test that generate_batches handles cases where reads exactly fill batches."""
        # This is a theoretical test - if we had exactly batch_size reads
        # The implementation should not create an empty trailing batch
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
        )
        batches = extractor.generate_batches()
        # Last batch should not be empty
        if batches:
            assert_that(len(batches[-1])).is_greater_than(0)

    def test_generate_batches_appends_remaining_partial_batch(self) -> None:
        """Test that generate_batches appends remaining reads when they don't fill a batch."""
        # Create extractor with large batch size so all reads fit in one batch
        # This tests line 130: if current_batch:
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            batch_size=5000,  # Much larger than file size
        )
        total_reads = extractor.total_reads

        # Now create extractor with batch_size < total_reads to ensure multiple batches
        # with a remainder that needs to be appended
        batch_size = max(10, total_reads // 3)  # Ensure we'll have remainder
        extractor = BarcodeExtractor(
            fastq_file=R1_PATH,
            chemistry_name="hydrop",
            n_workers=1,
            batch_size=batch_size,
        )
        batches = extractor.generate_batches()

        # Total reads in all batches should equal total_reads
        total_in_batches = sum(len(batch) for batch in batches)
        assert_that(total_in_batches).is_equal_to(total_reads)

        # If we have multiple batches, the last one should be partial (testing line 130)
        if len(batches) > 1:
            assert_that(len(batches[-1])).is_less_than(batch_size)
            assert_that(len(batches[-1])).is_greater_than(0)

    # ===== Edge Cases and Error Handling =====

    @pytest.mark.parametrize(
        "chemistry_name",
        [
            "hydrop",
            "carmack_custom_seq_1_0",
        ],
    )
    def test_init_accepts_supported_chemistries(self, chemistry_name: str) -> None:
        """Test that BarcodeExtractor accepts all supported chemistry names."""
        # Skip if chemistry doesn't exist
        try:
            extractor = BarcodeExtractor(
                fastq_file=R1_PATH,
                chemistry_name=chemistry_name,
                n_workers=1,
            )
            assert_that(extractor.chemistry_name).is_equal_to(chemistry_name)
        except ValueError as e:
            if "not supported" in str(e):
                pytest.skip(f"Chemistry {chemistry_name} not available in this environment")
            raise

    def test_fastq_file_property_accessible(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that the FastqFile object is accessible."""
        assert_that(barcode_extractor.fastq).is_not_none()
        assert_that(barcode_extractor.fastq.filename).contains("hydrop_scatac_1_S1_R1_001")

    def test_batch_size_property_is_int(self, barcode_extractor: BarcodeExtractor) -> None:
        """Test that batch_size property is an integer."""
        assert_that(barcode_extractor.batch_size).is_instance_of(int)
        assert_that(barcode_extractor.batch_size).is_greater_than(0)

    # ===== write_bc_all Tests =====

    def test_write_bc_all_creates_file_with_annotated_readnames(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_all writes annotated read names for all results."""
        output_path = tmp_path / "test_bc_all.txt"

        # Create mock results
        whitelists = hydrop_chemistry.barcode_whitelists
        bc_results = [
            self._make_successful_history("BC3", whitelists["BC3"][0]),
            self._make_successful_history("BC2", whitelists["BC2"][0]),
            self._make_successful_history("BC1", whitelists["BC1"][0]),
        ]
        result = ReadMatchResult(
            read_name="test_read_1",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=bc_results,
        )

        barcode_extractor.write_bc_all(output_path, [result])

        assert_that(output_path.exists()).is_true()
        content = output_path.read_text()
        assert_that(content).contains("test_read_1")
        assert_that(content).contains("SUCCESS")

    def test_write_bc_all_multiple_results(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_all handles multiple results correctly."""
        output_path = tmp_path / "test_bc_all_multi.txt"

        whitelists = hydrop_chemistry.barcode_whitelists
        results = []
        for i in range(3):
            bc_results = [
                self._make_successful_history("BC3", whitelists["BC3"][i]),
                self._make_successful_history("BC2", whitelists["BC2"][i]),
                self._make_successful_history("BC1", whitelists["BC1"][i]),
            ]
            results.append(
                ReadMatchResult(
                    read_name=f"read_{i}",
                    read="A" * 50,
                    qual="I" * 50,
                    chemistry=hydrop_chemistry,
                    bc_results=bc_results,
                )
            )

        barcode_extractor.write_bc_all(output_path, results)

        lines = output_path.read_text().strip().split("\n")
        assert_that(len(lines)).is_equal_to(3)
        for i in range(3):
            assert_that(lines[i]).contains(f"read_{i}")

    def test_write_bc_all_empty_results(
        self, tmp_path, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that write_bc_all handles empty results list."""
        output_path = tmp_path / "test_bc_all_empty.txt"

        barcode_extractor.write_bc_all(output_path, [])

        assert_that(output_path.exists()).is_true()
        assert_that(output_path.read_text()).is_empty()

    # ===== write_bc_valid Tests =====

    def test_write_bc_valid_only_successful_matches(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_valid only writes successful matches."""
        output_path = tmp_path / "test_bc_valid.txt"

        whitelists = hydrop_chemistry.barcode_whitelists
        # Successful result
        success_result = ReadMatchResult(
            read_name="success_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", whitelists["BC3"][0]),
                self._make_successful_history("BC2", whitelists["BC2"][0]),
                self._make_successful_history("BC1", whitelists["BC1"][0]),
            ],
        )
        # Failed result
        fail_result = ReadMatchResult(
            read_name="fail_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_failed_history("BC3", "ZZZZZZZZZZ"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),
                self._make_failed_history("BC1", "ZZZZZZZZZZ"),
            ],
        )

        barcode_extractor.write_bc_valid(output_path, [success_result, fail_result])

        content = output_path.read_text()
        assert_that(content).contains("success_read")
        assert_that(content).does_not_contain("fail_read")

    def test_write_bc_valid_skips_none_full_barcode(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_valid skips results with None full_barcode."""
        output_path = tmp_path / "test_bc_valid_none.txt"

        # Partial success - one barcode failed
        partial_result = ReadMatchResult(
            read_name="partial_read",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", "CAGTGTGGAA"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),  # Failed
                self._make_successful_history("BC1", "GAACAGTAGT"),
            ],
        )

        barcode_extractor.write_bc_valid(output_path, [partial_result])

        # Should be empty since full_barcode is None (BC2 failed)
        assert_that(output_path.read_text()).is_empty()

    def test_write_bc_valid_empty_results(
        self, tmp_path, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that write_bc_valid handles empty results list."""
        output_path = tmp_path / "test_bc_valid_empty.txt"

        barcode_extractor.write_bc_valid(output_path, [])

        assert_that(output_path.exists()).is_true()
        assert_that(output_path.read_text()).is_empty()

    # ===== write_bc_counts Tests =====

    def test_write_bc_counts_aggregates_barcodes(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_counts aggregates and counts unique barcodes."""
        output_path = tmp_path / "test_bc_counts.csv"

        # Create results with duplicate barcodes
        results = []
        for _ in range(3):
            results.append(
                ReadMatchResult(
                    read_name="read",
                    read="A" * 50,
                    qual="I" * 50,
                    chemistry=hydrop_chemistry,
                    bc_results=[
                        self._make_successful_history("BC3", "CAGTGTGGAA"),
                        self._make_successful_history("BC2", "ACGGTGGACT"),
                        self._make_successful_history("BC1", "GAACAGTAGT"),
                    ],
                )
            )
        # Add one different barcode
        results.append(
            ReadMatchResult(
                read_name="read2",
                read="A" * 50,
                qual="I" * 50,
                chemistry=hydrop_chemistry,
                bc_results=[
                    self._make_successful_history("BC3", "TGACCGTACT"),
                    self._make_successful_history("BC2", "TATGCAGTTA"),
                    self._make_successful_history("BC1", "TCTGAGATCG"),
                ],
            )
        )

        barcode_extractor.write_bc_counts(output_path, results)

        lines = output_path.read_text().strip().split("\n")
        assert_that(len(lines)).is_equal_to(2)
        # First line should be the barcode with count 3 (most common first)
        assert_that(lines[0]).contains(",3")
        # Second line should have count 1
        assert_that(lines[1]).contains(",1")

    def test_write_bc_counts_skips_failed_results(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_counts excludes failed results from counting."""
        output_path = tmp_path / "test_bc_counts_skip.csv"

        success_result = ReadMatchResult(
            read_name="success",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", "CAGTGTGGAA"),
                self._make_successful_history("BC2", "ACGGTGGACT"),
                self._make_successful_history("BC1", "GAACAGTAGT"),
            ],
        )
        fail_result = ReadMatchResult(
            read_name="fail",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_failed_history("BC3", "ZZZZZZZZZZ"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),
                self._make_failed_history("BC1", "ZZZZZZZZZZ"),
            ],
        )

        barcode_extractor.write_bc_counts(output_path, [success_result, fail_result])

        lines = output_path.read_text().strip().split("\n")
        assert_that(len(lines)).is_equal_to(1)
        assert_that(lines[0]).starts_with("CAGTGTGGAAACGGTGGACTGAACAGTAGT")

    def test_write_bc_counts_empty_results(
        self, tmp_path, barcode_extractor: BarcodeExtractor
    ) -> None:
        """Test that write_bc_counts handles empty results list."""
        output_path = tmp_path / "test_bc_counts_empty.csv"

        barcode_extractor.write_bc_counts(output_path, [])

        assert_that(output_path.exists()).is_true()
        assert_that(output_path.read_text()).is_empty()

    def test_get_barcode_counts_skips_failed_results(
        self, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that get_barcode_counts only includes successful full barcodes."""
        success_result = ReadMatchResult(
            read_name="success",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", "CAGTGTGGAA"),
                self._make_successful_history("BC2", "ACGGTGGACT"),
                self._make_successful_history("BC1", "GAACAGTAGT"),
            ],
        )
        fail_result = ReadMatchResult(
            read_name="fail",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_failed_history("BC3", "ZZZZZZZZZZ"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),
                self._make_failed_history("BC1", "ZZZZZZZZZZ"),
            ],
        )

        barcode_counts = barcode_extractor.get_barcode_counts([success_result, fail_result])

        assert_that(barcode_counts).is_equal_to({"CAGTGTGGAAACGGTGGACTGAACAGTAGT": 1})

    # ===== write_bc_rank_plot Tests =====

    def test_write_bc_rank_plot_creates_png(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_rank_plot saves a barcode rank plot image."""
        output_path = tmp_path / "test_bc_rank.png"
        results = []
        for _ in range(3):
            results.append(
                ReadMatchResult(
                    read_name="read",
                    read="A" * 50,
                    qual="I" * 50,
                    chemistry=hydrop_chemistry,
                    bc_results=[
                        self._make_successful_history("BC3", "CAGTGTGGAA"),
                        self._make_successful_history("BC2", "ACGGTGGACT"),
                        self._make_successful_history("BC1", "GAACAGTAGT"),
                    ],
                )
            )

        barcode_extractor.write_bc_rank_plot(output_path, results)

        assert_that(output_path.exists()).is_true()
        assert_that(output_path.stat().st_size).is_greater_than(0)

    def test_write_bc_rank_plot_handles_no_valid_barcodes(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_rank_plot still writes a plot when no barcodes matched."""
        output_path = tmp_path / "test_bc_rank_empty.png"
        failed_result = ReadMatchResult(
            read_name="fail",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_failed_history("BC3", "ZZZZZZZZZZ"),
                self._make_failed_history("BC2", "ZZZZZZZZZZ"),
                self._make_failed_history("BC1", "ZZZZZZZZZZ"),
            ],
        )

        barcode_extractor.write_bc_rank_plot(output_path, [failed_result])

        assert_that(output_path.exists()).is_true()
        assert_that(output_path.stat().st_size).is_greater_than(0)

    # ===== write_bc_stats Tests =====

    def test_write_bc_stats_creates_report(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_stats creates a statistics report file."""
        output_path = tmp_path / "test_bc_stats.txt"

        whitelists = hydrop_chemistry.barcode_whitelists
        results = [
            ReadMatchResult(
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
        ]

        barcode_extractor.write_bc_stats(output_path, results, log_stats=False)

        assert_that(output_path.exists()).is_true()
        content = output_path.read_text()
        assert_that(content).contains("Total reads")
        assert_that(content).contains("Perfect matches")
        assert_that(content).contains("BC3")
        assert_that(content).contains("BC2")
        assert_that(content).contains("BC1")

    def test_write_bc_stats_with_log_stats_enabled(
        self, tmp_path, barcode_extractor: BarcodeExtractor, hydrop_chemistry: ChemistryHydrop
    ) -> None:
        """Test that write_bc_stats logs when log_stats=True."""
        output_path = tmp_path / "test_bc_stats_log.txt"

        whitelists = hydrop_chemistry.barcode_whitelists
        results = [
            ReadMatchResult(
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
        ]

        # Should not raise any logging errors
        barcode_extractor.write_bc_stats(output_path, results, log_stats=True)

        assert_that(output_path.exists()).is_true()

    def test_extract_barcodes_writes_rank_plot_in_output_stage(
        self,
        tmp_path,
        monkeypatch,
        barcode_extractor: BarcodeExtractor,
        hydrop_chemistry: ChemistryHydrop,
    ) -> None:
        """Test that extract_barcodes includes the rank plot in the file-writing stage."""
        import carmack.barcode.barcode_extractor as barcode_extractor_module

        result = ReadMatchResult(
            read_name="read1",
            read="A" * 50,
            qual="I" * 50,
            chemistry=hydrop_chemistry,
            bc_results=[
                self._make_successful_history("BC3", "CAGTGTGGAA"),
                self._make_successful_history("BC2", "ACGGTGGACT"),
                self._make_successful_history("BC1", "GAACAGTAGT"),
            ],
        )

        class DummyFuture:
            def result(self):
                return [result], 1

        class DummyExecutor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, func, batch, hybrid_extractor=None):
                return DummyFuture()

        class DummyProgress:
            def __init__(self):
                self.add_task_calls = []
                self.update_calls = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def add_task(self, description, total):
                self.add_task_calls.append((description, total))
                return description

            def update(self, task, advance):
                self.update_calls.append((task, advance))

        read_progress = DummyProgress()
        write_progress = DummyProgress()
        progress_bars = iter([read_progress, write_progress])

        monkeypatch.setattr(
            barcode_extractor, "iter_batches", lambda: iter([[("read1", "A", "I")]])
        )
        monkeypatch.setattr(
            barcode_extractor_module, "ProcessPoolExecutor", lambda max_workers: DummyExecutor()
        )
        monkeypatch.setattr(
            barcode_extractor_module, "wait", lambda fs, return_when: (set(fs), set())
        )
        monkeypatch.setattr(
            barcode_extractor_module, "progress_bar", lambda unit: next(progress_bars)
        )

        barcode_extractor.write_bc_all = mock.Mock()
        barcode_extractor.write_bc_valid = mock.Mock()
        barcode_extractor.write_bc_counts = mock.Mock()
        barcode_extractor.write_bc_rank_plot = mock.Mock()
        barcode_extractor.write_bc_stats = mock.Mock()

        barcode_extractor.extract_barcodes(output_dir=str(tmp_path), prefix="test")

        barcode_extractor.write_bc_rank_plot.assert_called_once_with(
            tmp_path / "test.bc_rank.png", [result]
        )
        assert_that(write_progress.add_task_calls).contains(("Writing output files...", 5))
        assert_that(write_progress.update_calls).is_length(5)

    # ===== process_read_batch Tests =====

    def test_process_read_batch_empty_batch(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_matchers: dict
    ) -> None:
        """Test that process_read_batch handles empty batch."""
        from carmack.barcode.barcode_extractor import process_read_batch
        from carmack.barcode.hybrid_extractor import HybridExtractor

        hybrid_extractor = HybridExtractor(chemistry=hydrop_chemistry, matchers=hydrop_matchers)

        results, batch_len = process_read_batch([], hybrid_extractor)

        assert_that(results).is_empty()
        assert_that(batch_len).is_equal_to(0)

    def test_process_read_batch_single_read(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_matchers: dict
    ) -> None:
        """Test that process_read_batch processes single read correctly."""
        from carmack.barcode.barcode_extractor import process_read_batch
        from carmack.barcode.hybrid_extractor import HybridExtractor

        hybrid_extractor = HybridExtractor(chemistry=hydrop_chemistry, matchers=hydrop_matchers)

        batch = [("test_read", "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "I" * 52)]
        results, batch_len = process_read_batch(batch, hybrid_extractor)

        assert_that(batch_len).is_equal_to(1)
        assert_that(len(results)).is_equal_to(1)
        assert_that(results[0]).is_instance_of(ReadMatchResult)
        assert_that(results[0].read_name).is_equal_to("test_read")

    def test_process_read_batch_multiple_reads(
        self, hydrop_chemistry: ChemistryHydrop, hydrop_matchers: dict
    ) -> None:
        """Test that process_read_batch processes multiple reads."""
        from carmack.barcode.barcode_extractor import process_read_batch
        from carmack.barcode.hybrid_extractor import HybridExtractor

        hybrid_extractor = HybridExtractor(chemistry=hydrop_chemistry, matchers=hydrop_matchers)

        batch = [
            ("read1", "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT", "I" * 52),
            ("read2", "TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA", "I" * 52),
        ]
        results, batch_len = process_read_batch(batch, hybrid_extractor)

        assert_that(batch_len).is_equal_to(2)
        assert_that(len(results)).is_equal_to(2)
        assert_that(results[0].read_name).is_equal_to("read1")
        assert_that(results[1].read_name).is_equal_to("read2")

    # ===== Helper methods =====

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
    """Tests for barcode utility functions: hamming_distance, edit_distance, and plotting."""

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

    # ===== make_barcode_rank_plot =====

    def test_make_barcode_rank_plot_returns_log_scaled_figure(self) -> None:
        """Test that barcode rank plots use log scales and expected labels."""
        fig = make_barcode_rank_plot({"bc1": 10, "bc2": 4, "bc3": 1})
        ax = fig.axes[0]

        assert_that(fig).is_instance_of(plt.Figure)
        assert_that(ax.get_xscale()).is_equal_to("log")
        assert_that(ax.get_yscale()).is_equal_to("log")
        assert_that(ax.get_xlabel()).is_equal_to("Barcode rank")
        assert_that(ax.get_ylabel()).is_equal_to("Reads per barcode")
        assert_that(ax.get_title()).is_equal_to("Barcode Rank Plot")
        assert_that(ax.lines).is_length(1)

        plt.close(fig)

    def test_make_barcode_rank_plot_handles_empty_counts(self) -> None:
        """Test that barcode rank plotting handles empty counts without failing."""
        fig = make_barcode_rank_plot({})
        ax = fig.axes[0]

        assert_that(ax.lines).is_empty()
        assert_that([text.get_text() for text in ax.texts]).contains("No valid barcodes")

        plt.close(fig)


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
        for comp in hydrop_chemistry.read_structure.get_components_by_type(
            ReadComponentType.BARCODE
        ):
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
            if comp.type is ReadComponentType.BARCODE and comp.name != "BC3":
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
