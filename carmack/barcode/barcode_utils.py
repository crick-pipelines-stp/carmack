import numba
import numpy as np

@numba.njit
def hamming_distance(a, b):
    """
    Calculate the Hamming distance between two arrays.

    Args:
        a (np.ndarray): First array (1D, same length as b).
        b (np.ndarray): Second array (1D, same length as a).

    Returns:
        int: Number of positions where a and b differ.
    """
    mismatches = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            mismatches += a[i] != b[i]
    return mismatches

@numba.njit
def find_anchor_hamming(read, anchor, max_mismatches):
    """
    Find the start index of the first window in 'read' where the Hamming distance to 'anchor' is <= max_mismatches.

    Args:
        read (np.ndarray): The array to search within.
        anchor (np.ndarray): The anchor array to match.
        max_mismatches (int): Maximum allowed mismatches (Hamming distance).

    Returns:
        int: Start index of the first matching window, or -1 if not found.
    """
    read_len = len(read)
    anchor_len = len(anchor)
    for i in range(read_len - anchor_len + 1):
        window = read[i:i + anchor_len]
        if hamming_distance(window, anchor) <= max_mismatches:
            return i
    return -1
