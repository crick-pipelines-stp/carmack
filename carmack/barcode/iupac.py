"""IUPAC nucleotide ambiguity code handling for sequence alignment."""

import numpy as np
from Bio.Align import substitution_matrices


# IUPAC ambiguity codes mapping to possible nucleotides
IUPAC_CODES: dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}


def create_iupac_substitution_matrix(
    match_score: float = 1.0, mismatch_score: float = -1.0
) -> substitution_matrices.Array:
    """
    Create a substitution matrix that handles IUPAC ambiguity codes.

    Two bases are considered a match (full match_score) if their possible
    nucleotide sets have any overlap. For example, R (A or G) matches A
    because A is in {A, G}.

    Args:
        match_score: Score for matching bases (default: 1.0).
        mismatch_score: Score for mismatching bases (default: -1.0).

    Returns:
        A BioPython substitution matrix Array that can be assigned to
        PairwiseAligner.substitution_matrix.
    """
    alphabet = "".join(IUPAC_CODES.keys())
    n = len(alphabet)

    # Create score matrix
    scores = np.zeros((n, n), dtype=np.float64)

    for i, base1 in enumerate(alphabet):
        for j, base2 in enumerate(alphabet):
            # Check if the sets of possible nucleotides overlap
            if IUPAC_CODES[base1] & IUPAC_CODES[base2]:
                scores[i, j] = match_score
            else:
                scores[i, j] = mismatch_score

    # Create BioPython substitution matrix
    matrix = substitution_matrices.Array(alphabet, dims=2, data=scores)

    return matrix
