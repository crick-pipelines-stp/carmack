# pylint: disable=missing-function-docstring,missing-class-docstring
import numpy as np
import pytest
from assertpy import assert_that

from carmack.barcode.barcode_utils import hamming_distance, find_anchor_hamming


def dna_to_array(seq: str) -> np.ndarray:
    """Convert a DNA string to a numpy array of single characters."""
    return np.array(list(seq), dtype="U1")


class TestBarcodeUtils:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ("ACGT", "ACGT", 0),
            ("ACGT", "TGCA", 4),
            ("AAAA", "TTTT", 4),
            ("A", "T", 1),
            ("", "", 0),
            ("ACGT", "ACGA", 1),
        ],
    )
    def test_hamming_distance(self, a, b, expected):
        arr_a = dna_to_array(a)
        arr_b = dna_to_array(b)
        result = hamming_distance(arr_a, arr_b)
        assert_that(result).is_equal_to(expected)

    @pytest.mark.parametrize(
        "read, anchor, max_mismatches, expected",
        [
            ("ACGT", "AC", 0, 0),  # Exact match at start
            ("TACGT", "CG", 0, 2),  # Exact match in middle
            ("ACGT", "CG", 1, 1),  # Match with mismatches allowed
            ("ACGT", "TT", 0, -1),  # No match within allowed mismatches
            ("AC", "ACG", 1, -1),  # Anchor longer than read
            ("ACG", "", 0, 0),  # Empty anchor
            ("", "A", 0, -1),  # Empty read
        ],
    )
    def test_find_anchor_hamming(self, read, anchor, max_mismatches, expected):
        arr_read = dna_to_array(read)
        arr_anchor = dna_to_array(anchor)
        result = find_anchor_hamming(arr_read, arr_anchor, max_mismatches)
        assert_that(result).is_equal_to(expected)
