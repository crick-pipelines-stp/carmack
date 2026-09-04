import logging
from dataclasses import dataclass

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


@dataclass(frozen=True)
class AlignmentContainer:
    """
    Container for the single alignment a match decision is taken from.

    Attributes:
        score: Alignment score of the best alignment between the two sequences.
        seq1_span: (start, end) of the aligned region in seq1, in seq1 coordinates, taken from
            the first optimal path. Equally-scoring paths do not always agree on this span, so
            it is the span of one specific alignment rather than a property of the score.
        bc: The seq2 sequence this alignment was against, carried so that a caller holding
            several tied containers knows which whitelist entry each one came from. It is the
            unsanitised argument, since it is what a caller assigns as the matched barcode.
    """

    score: float
    seq1_span: tuple[int, int]
    bc: str


class AlignmentMatcher(MatcherBase):
    """
    Local alignment barcode matching.

    Uses BioPython's PairwiseAligner for local alignment with a custom substitution
    matrix that treats N bases as wildcards matching any nucleotide. Candidate
    selection and ranking are driven entirely by alignment scores rather than
    edit distance computation.

    This matcher keeps the barcode-only allowed_component_types of MatcherBase, and that
    declaration is the only thing stopping it being pointed at another component type. The
    narrowness is a deliberate default rather than a structural bar: unlike FixedPositionMatcher
    it never reads component.start, so widening it would need only the score threshold, which is
    derived from component.length, re-examined against the new component's length and budget.
    """

    def __init__(
        self,
        whitelist: tuple[str, ...],
        component: ReadComponent,
        chemistry: ChemistryBase,
        max_errors: int,
    ) -> None:
        """
        Initialize the AlignmentMatcher with the given whitelist and barcode component.

        Args:
            whitelist: Tuple of valid barcode sequences.
            component: The ReadComponent defining the barcode's position and length.
            chemistry: The chemistry defining the surrounding read structure.
            max_errors: Maximum edit distance allowed against a whitelist entry. Supplied by the
                caller so the matcher does not reach into the chemistry for a budget that may not
                be the barcode one.
        """
        super().__init__(whitelist, component, chemistry)
        self.max_errors = max_errors
        self.score_threshold = self.compute_score_threshold()
        self.aligner = self.build_aligner()

        log.debug(
            f"AlignmentMatcher initialized for {self.component.name} with max_errors={self.max_errors}, {len(whitelist)} barcodes, score_threshold={self.score_threshold}"
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
        bc_len = self.component.length
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

    def trim_read(self, read: str, start_idx: int) -> str:
        """
        Trim the read to the expected length for the barcode component.

        This is used to ensure that the read segment being matched is of the correct length, which
        can help improve matching accuracy and reduce false positives.

        Args:
            read: The sequencing read to trim.
            start_idx: Index to trim from.

        Returns:
            The read from start_idx onwards, or an empty string if start_idx is beyond the read.
        """
        if start_idx >= len(read):
            log.debug(
                f"Start index {start_idx} is beyond read length {len(read)}. Returning empty string."
            )
            return ""
        return read[start_idx:]

    def align_seqs(self, seq1: str, seq2: str) -> AlignmentContainer | None:
        """
        Align two sequences and extract alignment details.

        Args:
            seq1: First sequence (e.g., read segment)
            seq2: Second sequence (e.g., barcode)

        Returns:
            AlignmentContainer holding the score, the first optimal path's aligned span in seq1,
            and the unsanitised seq2, if the alignment meets the threshold, else None
        """
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Aligning sequences: '{seq1}' vs '{seq2}'")
        alignments = self.aligner.align(self.sanitise_sequence(seq1), self.sanitise_sequence(seq2))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Found {len(alignments)} alignments, {alignments}")

        if alignments.score < self.score_threshold:
            return None

        # Only the first optimal path's span is ever read. Indexing takes one traceback;
        # iterating would force BioPython to enumerate every optimal path, of which there
        # can be combinatorially many.
        alignment = alignments[0]
        return AlignmentContainer(
            score=alignments.score,
            seq1_span=(int(alignment.aligned[0][0][0]), int(alignment.aligned[0][-1][1])),
            bc=seq2,
        )

    def assign_within_budget(
        self, attempt: BarcodeMatchAttempt, bc: str
    ) -> list[BarcodeMatchAttempt]:
        """
        Assign `bc` to `attempt` if the aligned candidate is within the error budget.

        The alignment score gate is necessary but not sufficient: a candidate can clear
        score_threshold and still be further than max_errors from the barcode it aligned to,
        because an indel-bearing local alignment scores better than its edit distance implies.
        The distance is therefore recomputed against the barcode actually being assigned.

        Args:
            attempt: The match attempt to assign to, mutated in place on success.
            bc: The whitelist barcode this attempt's alignment came from.

        Returns:
            A single-element list holding `attempt` with `match` and `edit_distance` populated,
            or a single-element list holding a fresh matchless attempt if the budget is exceeded.
        """
        ed = edit_distance(attempt.candidate, bc)
        if ed > self.max_errors:
            log.debug(
                f"Best alignment candidate '{attempt.candidate}' failed edit distance check with edit distance {ed} exceeding max_errors {self.max_errors}. Marking as no match."
            )
            return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]
        attempt.match = bc
        attempt.edit_distance = ed
        return [attempt]

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Align each candidate barcode to `read` and return the best match.

        The best match is the barcode alignment with the highest alignment score.
        If multiple barcodes tie for best score, return all.

        Args:
            read: The sequencing read to match against.
            start_idx: A hard floor on where a match may begin. The read is trimmed to
                ``read[start_idx:]`` before alignment, so bases before ``start_idx`` are invisible
                to this matcher and no match can begin before it. A window that starts before
                ``start_idx`` is therefore seen only in truncated form, and resolves only while
                the truncation stays within ``max_errors``. Alignment coordinates are shifted by
                ``start_idx`` before being returned, so ``read_idx`` is in original-read
                coordinates.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results.
        """
        best_alignments = []

        # Initialize below threshold to ensure only valid alignments are considered
        best_score: float = self.score_threshold - 1

        trimmed_read = self.trim_read(read, start_idx)
        for bc in self.whitelist_set:
            alignment = self.align_seqs(trimmed_read, bc)

            if alignment is not None:
                if alignment.score > best_score:
                    best_alignments = [alignment]
                    best_score = alignment.score
                elif alignment.score == best_score:
                    best_alignments.append(alignment)

        # No alignments, return attempt with no match
        if not best_alignments:
            log.debug(f"No valid alignment matches found for read starting at index {start_idx}.")
            return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]

        # Prepare match attempts for best alignments
        results: list[tuple[BarcodeMatchAttempt, str]] = []
        for aln in best_alignments:
            seq1_span = aln.seq1_span
            read_idx = (
                seq1_span[0] + start_idx,
                seq1_span[1] + start_idx,
            )  # Convert to absolute read coordinates
            result = BarcodeMatchAttempt(
                method=MatchMethod.ALIGNMATCH,
                candidate=trimmed_read[seq1_span[0] : seq1_span[1]],
                match=None,  # We don't assign a single match if multiple barcodes tie
                read_idx=read_idx,
                edit_distance=None,
            )
            results.append((result, aln.bc))

        # Single match, assign the matched barcode to the result
        if len(results) == 1:
            result, bc = results[0]
            log.debug(f"Unique best alignment match found: {bc} with score {best_score}")
            return self.assign_within_budget(result, bc)

        # Multiple best alignments - validate with adjacent spacer sequences
        validated_results: list[tuple[BarcodeMatchAttempt, str]] = []
        for result, bc in results:
            if result.read_idx is not None:
                spacers_check = self.check_spacers(read, result.read_idx)
                result.spacer_upstream = spacers_check["upstream"]
                result.spacer_downstream = spacers_check["downstream"]

                # Validate if at least one adjacent spacer is present
                if any(spacers_check.values()):
                    validated_results.append((result, bc))

        # If only one validated result, assign the matched barcode and return
        if len(validated_results) == 1:
            final_result, bc = validated_results[0]
            log.debug(
                f"Unique best alignment match validated by spacers: {bc} with score {best_score}"
            )
            return self.assign_within_budget(final_result, bc)

        # If multiple results validate, check if only one has spacers on both sides
        both_spacers = [
            entry
            for entry in validated_results
            if entry[0].spacer_upstream is not None and entry[0].spacer_downstream is not None
        ]
        if len(validated_results) > 1 and len(both_spacers) == 1:
            final_result, bc = both_spacers[0]
            log.debug(
                f"Unique best alignment match validated by having both spacers: {bc} with score {best_score}"
            )
            return self.assign_within_budget(final_result, bc)

        # If multiple results still remain, we have ambiguity
        # We return all validated results but mark match as None to indicate ambiguity
        if validated_results:
            log.debug(
                f"Ambiguous alignment match: candidate '{validated_results[0][0].candidate}' with score {best_score} has multiple best matches. Will be marked as ambiguous."
            )
            return [entry[0] for entry in validated_results]

        log.debug(
            f"Ambiguous alignment match with no spacer evidence: {len(results)} candidates tied at score {best_score}. All will be marked as ambiguous."
        )
        return [entry[0] for entry in results]
