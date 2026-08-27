import logging
from dataclasses import dataclass
from typing import Literal

from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import Array

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_structure import ReadComponent


log = logging.getLogger(__name__)

ALPHABETS = "ACGTN"
VALID_BASES = frozenset(ALPHABETS)
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_OPEN_SCORE = -0.5
GAP_EXTEND_SCORE = -1


@dataclass
class AlignmentContainer:
    """
    Container for alignment results from PairwiseAligner.

    Attributes:
        score: Alignment score for the best alignment
        seq1_coords: List of (start, end) tuples for each match in seq1 (sorted)
        seq2_coords: List of (start, end) tuples for each match in seq2 (sorted)
        bc: Optional field to store the matched barcode sequence
    """

    score: float
    seq1_coords: list[tuple[int, int]]
    seq2_coords: list[tuple[int, int]]
    bc: str | None = None


class AlignmentMatcher(MatcherBase):
    """
    Local alignment barcode matching.

    Uses BioPython's PairwiseAligner for local alignment with a custom substitution
    matrix that treats N bases as wildcards matching any nucleotide. Candidate
    selection and ranking are driven entirely by alignment scores rather than
    edit distance computation.
    """

    def __init__(
        self,
        whitelist: tuple[str, ...],
        barcode_component: ReadComponent,
        chemistry: ChemistryBase,
    ) -> None:
        super().__init__(whitelist, barcode_component, chemistry)
        self.max_errors = chemistry.max_errors.barcode
        self.score_threshold = self.compute_score_threshold()
        self.aligner = self.build_aligner()

        log.debug(
            f"AlignmentMatcher initialized for {self.barcode_component.name} with max_errors_barcode={self.max_errors}, {len(whitelist)} barcodes, score_threshold={self.score_threshold}"
        )

    def build_substitution_matrix(self) -> Array:
        """
        Build a substitution matrix where N matches any base with the full match score.

        Returns:
            BioPython substitution matrix with N-as-wildcard scoring
        """
        mat = Array(ALPHABETS, dims=2, dtype=float)
        for c1 in ALPHABETS:
            for c2 in ALPHABETS:
                if c1 == "N" or c2 == "N":
                    mat[c1, c2] = MATCH_SCORE
                elif c1 == c2:
                    mat[c1, c2] = MATCH_SCORE
                else:
                    mat[c1, c2] = MISMATCH_SCORE
        return mat

    def build_aligner(self) -> PairwiseAligner:
        """
        Build and configure a BioPython PairwiseAligner for local alignment.

        Returns:
            Configured PairwiseAligner instance
        """
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.substitution_matrix = self.build_substitution_matrix()
        aligner.open_gap_score = GAP_OPEN_SCORE
        aligner.extend_gap_score = GAP_EXTEND_SCORE
        return aligner

    def compute_score_threshold(self) -> float:
        """
        Compute the minimum alignment score that could correspond to max_errors edits.

        For local alignment, mismatches are clipped rather than penalised, so each
        substitution effectively costs 1 (the lost match) rather than 2 (lost match + penalty).

        Returns:
            Minimum alignment score to consider (as float)
        """
        bc_len = self.barcode_component.length
        max_errors = self.max_errors

        perfect_score = bc_len * MATCH_SCORE
        # For local alignment, substitutions cost MATCH_SCORE (1 point) because
        # mismatches are clipped. Indels cost GAP_OPEN + GAP_EXTEND.
        # We use the maximum of these as a conservative penalty per error.
        max_penalty_per_error = max(
            MATCH_SCORE,  # substitution: lose 1 match
            abs(GAP_OPEN_SCORE) + abs(GAP_EXTEND_SCORE),  # indel: 0.5 + 1 = 1.5
        )
        return float(perfect_score - max_errors * max_penalty_per_error)

    def sanitise_sequence(self, sequence: str) -> str:
        """
        Replace characters not in the ACGTN alphabet with N.

        BioPython's PairwiseAligner requires all characters to be in the substitution
        matrix alphabet. Any non-standard bases are treated as ambiguous (N).

        Args:
            sequence: Input DNA sequence

        Returns:
            Sanitised sequence with only ACGTN characters
        """
        if all(c in VALID_BASES for c in sequence):
            return sequence
        return "".join(c if c in VALID_BASES else "N" for c in sequence)

    def align_seqs(self, seq1: str, seq2: str) -> AlignmentContainer | None:
        """
        Align two sequences and extract alignment details.

        Args:
            seq1: First sequence (e.g., read segment)
            seq2: Second sequence (e.g., barcode)

        Returns:
            AlignmentContainer with score and coordinates if alignment meets threshold, else None
        """
        log.debug(f"Aligning sequences: '{seq1}' vs '{seq2}'")
        alignments = self.aligner.align(self.sanitise_sequence(seq1), self.sanitise_sequence(seq2))
        log.debug(f"Found {len(alignments)} alignments, {alignments}")

        if alignments.score < self.score_threshold:
            return None

        seq1_coords: list[tuple[int, int]] = []
        seq2_coords: list[tuple[int, int]] = []

        for aln in alignments:
            # Get full span of aligned coordinates (includes gaps/insertions)
            seq1_start = int(aln.aligned[0][0][0])  # Start of first segment
            seq1_end = int(aln.aligned[0][-1][1])  # End of last segment
            seq2_start = int(aln.aligned[1][0][0])  # Start of first segment
            seq2_end = int(aln.aligned[1][-1][1])  # End of last segment

            seq1_coords.append((seq1_start, seq1_end))
            seq2_coords.append((seq2_start, seq2_end))

        return AlignmentContainer(
            score=alignments.score,
            seq1_coords=seq1_coords,
            seq2_coords=seq2_coords,
        )

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Align each candidate barcode to `read` and return the best match.

        The best match is the barcode alignment with the highest alignment score.
        If multiple barcodes tie for best score, return all.

        Args:
            read: The sequencing read to match against.
            start_idx: The index in the read to start matching from (default is 0).
        """
        best_alignments = []

        # Initialize below threshold to ensure only valid alignments are considered
        best_score: float = self.score_threshold - 1

        trimmed_read = self.trim_read(read, start_idx)
        for bc in self.whitelist_set:
            alignments = self.align_seqs(trimmed_read, bc)

            # Return coords on read seq for first alignments
            if alignments is not None:
                alignments.bc = bc  # Store matched barcode in alignment container

                if alignments.score > best_score:
                    best_alignments = [alignments]
                    best_score = alignments.score
                elif alignments.score == best_score:
                    best_alignments.append(alignments)

        # No alignments, return attempt with no match
        if not best_alignments:
            log.debug(f"No valid alignment matches found for read starting at index {start_idx}.")
            return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]

        # Prepare match attempts for best alignments
        # Use only the first coordinate pair from each alignment to avoid duplicates
        # when local alignment finds multiple equivalent paths
        results: list[tuple[BarcodeMatchAttempt, str]] = []
        for aln in best_alignments:
            # Only use the first coordinate - all coords in an alignment refer to the same region
            seq1_coord = aln.seq1_coords[0]
            read_idx = (
                seq1_coord[0] + start_idx,
                seq1_coord[1] + start_idx,
            )  # Convert to absolute read coordinates
            result = BarcodeMatchAttempt(
                method=MatchMethod.ALIGNMATCH,
                candidate=trimmed_read[seq1_coord[0] : seq1_coord[1]],
                match=None,  # We don't assign a single match if multiple barcodes tie
                read_idx=read_idx,
                edit_distance=None,
            )
            results.append((result, aln.bc))

        # Single match, assign the matched barcode to the result
        if len(results) == 1:
            result, bc = results[0]
            log.debug(
                f"Unique best alignment match found: {bc} with score {best_alignments[0].score}"
            )
            ed = edit_distance(result.candidate, bc)
            if ed > self.max_errors:
                log.debug(
                    f"Best alignment candidate '{result.candidate}' failed edit distance check with edit distance {ed} exceeding max_errors {self.max_errors}. Marking as no match."
                )
                return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]
            result.match = bc
            result.edit_distance = ed
            return [result]

        # Multiple best alignments - validate with adjacent spacer sequences
        validated_results: list[
            tuple[BarcodeMatchAttempt, str, dict[Literal["upstream", "downstream"], str | None]]
        ] = []
        for result, bc in results:
            if result.read_idx is not None:
                spacers_check = self.check_spacers(read, result.read_idx)
                result.spacer_upstream = spacers_check["upstream"]
                result.spacer_downstream = spacers_check["downstream"]

                # Validate if at least one adjacent spacer is present
                if any(spacers_check.values()):
                    validated_results.append((result, bc, spacers_check))

        # If only one validated result, assign the matched barcode and return
        if len(validated_results) == 1:
            final_result, bc, _ = validated_results[0]
            log.debug(
                f"Unique best alignment match validated by spacers: {bc} with score {best_alignments[0].score}"
            )
            ed = edit_distance(final_result.candidate, bc)
            if ed > self.max_errors:
                log.debug(
                    f"Best alignment candidate '{final_result.candidate}' failed edit distance check with edit distance {ed} exceeding max_errors {self.max_errors}. Marking as no match."
                )
                return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]
            final_result.match = bc  # Assign the matched barcode
            final_result.edit_distance = ed
            return [final_result]

        # If multiple results validate, check if only one has spacers on both sides
        best_candidates_with_two_spacers = [
            c for c in validated_results if sum(v is not None for v in c[2].values()) == 2
        ]
        if len(validated_results) > 1 and len(best_candidates_with_two_spacers) == 1:
            final_result, bc, _ = best_candidates_with_two_spacers[0]
            log.debug(
                f"Unique best alignment match validated by having both spacers: {bc} with score {best_alignments[0].score}"
            )
            ed = edit_distance(final_result.candidate, bc)
            if ed > self.max_errors:
                log.debug(
                    f"Best alignment candidate '{final_result.candidate}' failed edit distance check with edit distance {ed} exceeding max_errors {self.max_errors}. Marking as no match."
                )
                return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]
            final_result.match = bc  # Assign the matched barcode
            final_result.edit_distance = ed
            return [final_result]

        # If multiple results still remain, we have ambiguity
        # We return all validated results but mark match as None to indicate ambiguity
        for r, _, _ in validated_results:
            log.debug(
                f"Ambiguous alignment match: candidate '{r.candidate}' with score {best_alignments[0].score} has multiple best matches. Will be marked as ambiguous."
            )
            return [r for r, _, _ in validated_results]

        return [r for r, _ in results]
