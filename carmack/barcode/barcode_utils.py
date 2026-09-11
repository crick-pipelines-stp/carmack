from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two arrays.

    Args:
        a (np.ndarray): First array (1D, same length as b).
        b (np.ndarray): Second array (1D, same length as a).

    Returns:
        int: Number of positions where a and b differ.
    """
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Arrays must be 1D.")
    return int(np.count_nonzero(a != b))


def edit_distance(seq1, seq2, n_char="N", n_matches_any=True):
    """
    Calculate the Levenshtein (edit) distance between two sequences.

    Parameters:
    -----------
    seq1 : str
        First sequence
    seq2 : str
        Second sequence
    n_char : str, optional
        Character to treat as wildcard (default: 'N')
    n_matches_any : bool, optional
        If True, n_char matches any character with cost 0 (default: True)
        If False, n_char is treated as a regular character

    Returns:
    --------
    int or float
        The Levenshtein distance between seq1 and seq2

    Examples:
    ---------
    >>> edit_distance("ACGT", "ACNT")
    0  # N matches T with no penalty

    >>> edit_distance("ACGT", "ACNT", n_matches_any=False)
    1  # N is treated as a different character from T

    >>> edit_distance("kitten", "sitting")
    3
    """
    m, n = len(seq1), len(seq2)

    # Create a matrix to store distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Determine substitution cost
            if seq1[i - 1] == seq2[j - 1]:
                substitution_cost = 0
            elif n_matches_any and (seq1[i - 1] == n_char or seq2[j - 1] == n_char):
                # If either character is N and n_matches_any is True, no penalty
                substitution_cost = 0
            else:
                substitution_cost = 1

            # Calculate minimum cost
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + substitution_cost,  # Substitution
            )

    return dp[m][n]


def make_barcode_rank_plot(barcode_counts: Mapping[str, int]) -> "Figure":
    """Create a barcode-rank plot from full-barcode counts.

    matplotlib is imported here rather than at module scope because it is the single heaviest
    import in the package -- around half a second -- and this is the only function in the
    module that needs it. Everything else here is small pure-sequence arithmetic that the
    matchers and the chemistry definitions depend on, and those sit on the import path of
    every command, including the ones that draw nothing.
    """
    import matplotlib.pyplot as plt

    counts = sorted((count for count in barcode_counts.values() if count > 0), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if counts:
        ranks = list(range(1, len(counts) + 1))
        ax.plot(ranks, counts, color="tab:blue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=1)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid barcodes",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlabel("Barcode rank")
    ax.set_ylabel("Reads per barcode")
    ax.set_title("Barcode Rank Plot")
    fig.tight_layout()

    return fig
