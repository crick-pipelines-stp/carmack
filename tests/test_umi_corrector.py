"""Tests for UMI correction.

The corrector groups extracted UMIs by their full cell barcode, normalises each
raw UMI to a fixed length and collapses directional variants of one molecule
into a single representative (``UB``) via ``umi_tools.UMIClusterer``. These tests
pin the collapse behaviour, the conservative distance-2 non-merge, the raw-``N``
and off-length filters, cell-barcode isolation and the reconciling stats.
"""

import pytest
from assertpy import assert_that

from carmack.umi import umi_corrector
from carmack.umi.umi_corrector import CorrectedUmi, UmiCorrector, UmiRecord


def records_from(specs: list[tuple[str, str, str]]) -> list[UmiRecord]:
    """Build ``UmiRecord`` objects from ``(read_id, barcode, raw_umi)`` specs."""
    return [UmiRecord(read_id=r, barcode=b, raw_umi=u) for r, b, u in specs]


class TestUmiCorrectorClustering:
    """Directional collapse, conservative non-merge and representative choice."""

    def test_directional_variants_collapse_to_highest_count_rep(self) -> None:
        specs = [
            ("p1", "BC", "AAAAAAAA"),
            ("p2", "BC", "AAAAAAAA"),
            ("p3", "BC", "AAAAAAAA"),
            ("v1", "BC", "AAAAAAAT"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that({rid: c.ub for rid, c in mapping.items()}).is_equal_to(
            {"p1": "AAAAAAAA", "p2": "AAAAAAAA", "p3": "AAAAAAAA", "v1": "AAAAAAAA"}
        )
        assert_that(mapping["v1"].ur).is_equal_to("AAAAAAAT")
        assert_that(stats.distinct_corrected_umis).is_equal_to(1)
        assert_that(stats.umi_collapses).is_equal_to(1)
        # Only v1 was reassigned to a different representative.
        assert_that(stats.corrections_applied).is_equal_to(1)
        assert_that(stats.assigned_reads).is_equal_to(4)
        assert_that(stats.num_cell_barcodes).is_equal_to(1)
        assert_that(stats.mean_reads_per_umi).is_equal_to(4.0)

    def test_distance_two_stays_separate(self) -> None:
        specs = [
            ("a1", "BC", "AAAAAAAA"),
            ("a2", "BC", "AAAAAAAA"),
            ("b1", "BC", "AAAAAATT"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that({c.ub for c in mapping.values()}).is_equal_to({"AAAAAAAA", "AAAAAATT"})
        assert_that(stats.distinct_corrected_umis).is_equal_to(2)
        assert_that(stats.umi_collapses).is_equal_to(0)
        # Every read is its own representative; nothing was reassigned.
        assert_that(stats.corrections_applied).is_equal_to(0)


class TestUmiCorrectorFilters:
    """Raw-``N`` and off-length reads are excluded before clustering."""

    def test_raw_n_reads_dropped_before_clustering(self) -> None:
        specs = [
            ("clean", "BC", "AAAAAAAA"),
            ("ncontam", "BC", "AAAANAAA"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(mapping).contains_key("clean")
        assert_that(mapping).does_not_contain_key("ncontam")
        assert_that(stats.dropped_raw_n).is_equal_to(1)

    def test_off_length_reads_dropped(self) -> None:
        specs = [
            ("ok", "BC", "AAAAAAAA"),
            ("short", "BC", "AAAAA"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(mapping).does_not_contain_key("short")
        assert_that(stats.dropped_off_length).is_equal_to(1)

    def test_group_with_only_dropped_reads_yields_empty(self) -> None:
        specs = [("only", "BC", "AANAAAAA")]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(mapping).is_empty()
        assert_that(stats.dropped_raw_n).is_equal_to(1)
        assert_that(stats.assigned_reads).is_equal_to(0)
        # A group whose reads were all dropped is not counted.
        assert_that(stats.num_cell_barcodes).is_equal_to(0)

    def test_swapping_the_sentinel_changes_the_drop_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(umi_corrector, "UMI_PAD_CHAR", "X")
        specs = [
            ("keepN", "BC", "AAAANAAA"),
            ("dropX", "BC", "AAAAXAAA"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(mapping).contains_key("keepN")
        assert_that(mapping).does_not_contain_key("dropX")
        assert_that(stats.dropped_raw_n).is_equal_to(1)


class TestUmiCorrectorBarcodeIsolation:
    """Correction is scoped to each full cell barcode."""

    def test_grouping_isolates_barcodes(self) -> None:
        specs = [
            ("x1", "B1", "AAAAAAAA"),
            ("x2", "B1", "AAAAAAAA"),
            ("y1", "B2", "AAAAAAAA"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(mapping["x1"].barcode).is_equal_to("B1")
        assert_that(mapping["y1"].barcode).is_equal_to("B2")
        assert_that(stats.distinct_corrected_umis).is_equal_to(2)
        assert_that(stats.umi_collapses).is_equal_to(0)
        assert_that(stats.num_cell_barcodes).is_equal_to(2)


class TestUmiCorrectorPaddingAndUr:
    """Padded representatives retain trailing sentinels; ``UR`` stays faithful."""

    def test_padded_representative_retains_trailing_sentinel(self) -> None:
        mapping, stats = UmiCorrector(8, 1).correct(records_from([("s1", "BC", "AAAAAAA")]))

        assert_that(mapping["s1"].ub).is_equal_to("AAAAAAAN")
        assert_that(mapping["s1"].ur).is_equal_to("AAAAAAA")
        # Length-normalisation alone is not a correction: the padded read is its
        # own representative, so nothing was reassigned.
        assert_that(stats.corrections_applied).is_equal_to(0)

    def test_ur_is_faithful_raw_and_ub_is_length_x(self) -> None:
        mapping, _ = UmiCorrector(8, 1).correct(records_from([("t", "BC", "AAAAAAA")]))

        assert_that(mapping["t"].ur).is_equal_to("AAAAAAA")
        assert_that(mapping["t"].ub).is_length(8)


class TestUmiCorrectorStats:
    """The reconciling correction tallies."""

    def test_empty_records_returns_empty(self) -> None:
        mapping, stats = UmiCorrector(8, 1).correct([])

        assert_that(mapping).is_empty()
        assert_that(stats.assigned_reads).is_equal_to(0)
        assert_that(stats.mean_reads_per_umi).is_equal_to(0.0)

    def test_stats_reconcile(self) -> None:
        specs = [
            ("c1", "BC", "AAAAAAAA"),
            ("c2", "BC", "AAAAAAAA"),
            ("nn", "BC", "AANAAAAA"),
            ("sh", "BC", "AAAAA"),
        ]
        mapping, stats = UmiCorrector(8, 1).correct(records_from(specs))

        assert_that(
            stats.assigned_reads + stats.dropped_raw_n + stats.dropped_off_length
        ).is_equal_to(4)
        assert_that(stats.assigned_reads).is_equal_to(len(mapping))
        assert_that(stats.corrections_applied).is_less_than_or_equal_to(stats.assigned_reads)

    def test_corrected_umi_is_immutable(self) -> None:
        corrected = CorrectedUmi(barcode="BC", ur="AAAAAAAA", ub="AAAAAAAA")
        with pytest.raises(Exception):
            corrected.ub = "TTTTTTTT"  # type: ignore[misc]
