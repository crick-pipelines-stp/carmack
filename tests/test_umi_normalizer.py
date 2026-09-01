"""Tests for UMI normalisation (SI-4 of the UMI epic).

The normaliser forces every raw UMI onto a single canonical length ``x`` so that
umi_tools' equal-length clustering can run: raw UMIs inside the length window are
right-padded (too short) or right-trimmed (too long) with a single swappable
sentinel, while raw UMIs outside the window are rejected. These tests pin the
length window, the padding/trimming behaviour and the single-constant sentinel.
"""

import pytest
from assertpy import assert_that

from carmack.umi import umi_normalizer
from carmack.umi.umi_normalizer import UMI_PAD_CHAR, normalize_umi


class TestNormalizeUmi:
    """Length-window rejection, padding, trimming and sentinel behaviour."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ACTACTA", "ACTACTAN"),  # 7 -> right-pad one sentinel
            ("ACTACTAC", "ACTACTAC"),  # 8 -> unchanged
            ("ACTACTACT", "ACTACTAC"),  # 9 -> right-trim
            ("ACTACT", None),  # 6 -> below window, reject
            ("ACTACTACTA", None),  # 10 -> above window, reject
        ],
    )
    def test_length_window(self, raw: str, expected: str | None) -> None:
        assert_that(normalize_umi(raw, 8, 1)).is_equal_to(expected)

    @pytest.mark.parametrize("raw", ["ACTACTA", "ACTACTAC", "ACTACTACT"])
    def test_accepted_umis_are_length_x(self, raw: str) -> None:
        assert_that(normalize_umi(raw, 8, 1)).is_length(8)

    def test_padding_uses_the_sentinel(self) -> None:
        assert_that(normalize_umi("ACTACTA", 8, 1)).ends_with(UMI_PAD_CHAR)

    def test_multi_pad_when_tolerance_allows(self) -> None:
        assert_that(normalize_umi("ACTACT", 8, 2)).is_equal_to("ACTACT" + UMI_PAD_CHAR * 2)

    def test_reject_far_below_window_returns_none(self) -> None:
        assert_that(normalize_umi("ACTAC", 8, 1)).is_none()

    def test_swapping_the_sentinel_changes_padding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(umi_normalizer, "UMI_PAD_CHAR", "X")
        assert_that(umi_normalizer.normalize_umi("ACTACTA", 8, 1)).is_equal_to("ACTACTAX")

    def test_default_sentinel_is_n(self) -> None:
        assert_that(UMI_PAD_CHAR).is_equal_to("N")
