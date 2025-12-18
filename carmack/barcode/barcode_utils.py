import numba
import numpy as np
from Bio.Align import PairwiseAligner
import logging

log = logging.getLogger(__name__)


@numba.njit
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
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
def find_anchor_hamming(read: np.ndarray, anchor: np.ndarray, max_mismatches: int) -> int:
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
        window = read[i : i + anchor_len]
        if hamming_distance(window, anchor) <= max_mismatches:
            return i
    return -1


def align_sequence(
    seq1: str, seq2: str, max_corrections: int = 0, use_iupac: bool = True
) -> dict[str, int] | None:
    """
    Align seq2 to seq1 using local pairwise alignment.

    Performs local alignment to find all occurrences of seq2 within seq1 that have
    the same optimal alignment score. Useful for finding barcode matches that may
    have mismatches or small gaps.

    Args:
        seq1: Target sequence to search within (typically the read).
        seq2: Query sequence to find (typically a barcode).
        max_corrections: Maximum allowed errors (mismatches + gaps).
                        If 0, returns best alignment(s) regardless of score.
                        Otherwise, only returns if score >= (len(seq2) - max_corrections).
        use_iupac: Whether to use IUPAC-aware substitution matrix for alignment. (e.g. "N" matches any base)

    Returns:
        Dictionary containing:
            - "score": Alignment score (float)
            - "seq1_coords": List of (start, end) tuples for each match in seq1 (sorted)
            - "seq2_coords": List of (start, end) tuples for each match in seq2 (sorted)
        Returns None if alignment score doesn't meet the threshold.

        Coordinates represent the full span of the alignment, including any internal gaps.

    Example:
        >>> result = align_sequence("AGNGTAGGGTAFSDFSDAAGGGTAGGT", "AGGGT")
        >>> print(result["score"])
        5.0
        >>> print(result["seq1_coords"])
        [(0, 5), (6, 11), (22, 27)]

        >>> # Example with gap in barcode
        >>> result = align_sequence("AGGAGT", "AGGGT")
        >>> print(result["seq1_coords"])
        [(0, 6)]  # Full span including the insertion
    """

    # Setup local alignment
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    if use_iupac:
        from carmack.barcode.iupac import create_iupac_substitution_matrix

        aligner.substitution_matrix = create_iupac_substitution_matrix(
            match_score=aligner.match_score, mismatch_score=aligner.mismatch_score
        )

    # Align
    alignments = aligner.align(seq1, seq2)

    # Extract coordinates from all alignments with same best score
    seq1_coords = []
    seq2_coords = []

    for aln in alignments:
        # Get full span of aligned coordinates (includes gaps/insertions)
        seq1_start = int(aln.aligned[0][0][0])  # Start of first segment
        seq1_end = int(aln.aligned[0][-1][1])  # End of last segment
        seq2_start = int(aln.aligned[1][0][0])  # Start of first segment
        seq2_end = int(aln.aligned[1][-1][1])  # End of last segment

        seq1_coords.append((seq1_start, seq1_end))
        seq2_coords.append((seq2_start, seq2_end))

    res = {
        "score": alignments.score,
        "seq1_coords": seq1_coords,
        "seq2_coords": seq2_coords,
        "alignment": alignments,
    }

    max_score = min(len(seq1), len(seq2))
    if alignments.score >= max(0, max_score - max_corrections):
        return res


def get_best_barcode(
    read: str, barcodes: list[str], max_corrections: int = 0
) -> tuple[tuple[int, int] | None, str]:
    """
    Align each candidate barcode to `read` and return the best match.

    The best match is the barcode alignment with the highest alignment score.
    If multiple barcodes tie for best score, behavior depends on the number of ties.

    Args:
        read: Read sequence to search within.
        barcodes: Candidate barcode sequences.
        max_corrections: Passed through to `align_sequence` as the maximum allowed errors
            (mismatches + gaps) for an alignment to be accepted.

    Returns:
        (coords, status) where coords is (start, end) on `read` (end-exclusive) or None.

        Status codes:
            - "BCALNOK": A best barcode alignment was found; coords is not None.
            - "NOTFNDBC": No barcode alignment was accepted; coords is None.
            - "AMBIGBC": Ambiguous (>= 3 equally best barcode alignments); coords is None.

        Tie handling:
            - If exactly 2 barcodes tie for best score, the first encountered best alignment
              (by iteration order) is returned.
    """
    all_alignments = []
    for bc in barcodes:
        alignments = align_sequence(read, bc, max_corrections)

        # Return coords on read seq for first alignments
        if alignments is not None:
            all_alignments.append(alignments)

    # If nothing aligns, return None
    if not all_alignments:
        return None, "NOTFNDBC"  # NO-BC-FOUND

    # Get barcode with highest score
    highest_score = max(aln["score"] for aln in all_alignments)
    best_alignments = [aln for aln in all_alignments if aln["score"] == highest_score]

    # If more than three alignments, return fail
    if len(best_alignments) >= 3:
        log.debug(f"Ambiguous barcode alignments found for read: {read}")
        for aln in best_alignments:
            log.debug(
                f"Alignment score: {aln['score']}, seq1_coords: {aln['seq1_coords']}, seq2_coords: {aln['seq2_coords']}"
            )
            for alignment_obj in aln["alignment"]:
                log.debug(alignment_obj.format())
        return None, "AMBIGBC"  # AMBIGUOUS-BARCODE

    if len(best_alignments) == 2:
        log.debug(
            f"Two equally best barcode alignments found for read: {read}, returning left-most."
        )

    # Return first set of coords for read seq
    return best_alignments[0]["seq1_coords"][0], "BCALNOK"  # BC-ALIGNMENT-OK
