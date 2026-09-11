import logging
from dataclasses import dataclass

from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import Array

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.matcher_base import MatcherBase, best_window
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


@dataclass
class AlignmentCandidate:
    """
    A whitelist entry that survived every gate, held together with the evidence about it.

    Attributes:
        attempt: The match attempt for this candidate, in original-read coordinates. Carries
            no match until resolution picks a winner, so an unresolved candidate reported as
            part of an ambiguity verdict is matchless by construction rather than by being
            reset.
        bc: The whitelist entry this candidate aligned to.
        edit_distance: The candidate's edit distance to ``bc``, recomputed from the aligned
            span. This is what resolution ranks on. The alignment score is deliberately not
            carried: it gates and it does not rank, and a candidate holding its own score is an
            invitation to order candidates by it.
    """

    attempt: BarcodeMatchAttempt
    bc: str
    edit_distance: int


class AlignmentMatcher(MatcherBase):
    """
    Local alignment barcode matching.

    Uses BioPython's PairwiseAligner for local alignment with a custom substitution
    matrix that treats N bases as wildcards matching any nucleotide. The alignment score
    *gates* which candidates are considered; it does not rank them. Ranking is by recomputed
    edit distance, which is the criterion the ambiguity contract is written in and the one
    KmerMatcher already uses. Score cannot rank because it is not monotone in edit distance:
    of two candidates the same number of edits from the read, the one whose error sits at an
    end of the barcode is clipped by local alignment and outscores the one whose error sits in
    the interior. Ranking on that separates candidates the contract says must not be separated.

    This matcher keeps the barcode-only allowed_component_types of MatcherBase, and that
    declaration is the only thing stopping it being pointed at another component type. The
    narrowness is a deliberate default rather than a structural bar: widening it would need the
    score threshold, which is derived from component.length, re-examined against the new
    component's length and budget, and would want a component whose start is resolved, since
    search_bounds falls back to an unbounded search without one.
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
        Compute the lowest alignment score a window within ``max_errors`` edits can score.

        The threshold is a gate, so it has to be at or below the *worst* score an in-budget
        window can produce. Anything higher makes a legitimate candidate invisible rather than
        merely unlikely, and which candidates disappear depends on where in the barcode their
        errors happen to sit -- so a too-high threshold silently biases the whole match.

        The worst case per error is a substitution that local alignment cannot clip, because
        it sits in the barcode's interior with matching bases on both sides. Aligning through
        it forfeits a match and takes the mismatch penalty as well, costing
        ``MATCH_SCORE - MISMATCH_SCORE``. The alternatives are all cheaper, which is why the
        cost is a maximum over them:

        =========================  ==========================================  ====
        Error                      Cost against a perfect score                10bp
        =========================  ==========================================  ====
        interior substitution      ``MATCH_SCORE - MISMATCH_SCORE``            2.0
        deletion                   ``MATCH_SCORE + abs(GAP_OPEN_SCORE)``       1.5
        insertion                  ``abs(GAP_OPEN_SCORE)``                     0.5
        terminal substitution      ``MATCH_SCORE`` (clipped, no penalty paid)  1.0
        =========================  ==========================================  ====

        A 1bp gap pays only ``open_gap_score``; ``extend_gap_score`` starts at the second gap
        position, so it does not enter a single-error cost at all. The previous formula used
        ``max(MATCH_SCORE, abs(GAP_OPEN_SCORE) + abs(GAP_EXTEND_SCORE))`` -- 1.5 -- which
        priced an interior substitution at 1.0 instead of 2.0. At ``max_errors=1`` that put the
        threshold at 8.5 while an interior substitution scores 8.0, so of two candidates one
        edit from the read, the one whose mismatch happened to sit at an end scored 9.0 and was
        declared a unique best match while the other was dropped before the tie was visible.

        Returns:
            Minimum alignment score to consider (as float)
        """
        bc_len = self.component.length
        max_errors = self.max_errors

        perfect_score = bc_len * MATCH_SCORE
        max_penalty_per_error = max(
            MATCH_SCORE - MISMATCH_SCORE,  # interior substitution: lose a match, pay a mismatch
            MATCH_SCORE + abs(GAP_OPEN_SCORE),  # deletion: lose a match, pay to open a gap
            abs(GAP_OPEN_SCORE),  # insertion: keep every match, pay to open a gap
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

    def search_bounds(self, read: str, start_idx: int) -> tuple[int, int]:
        """
        Return the half-open slice of the read a match for this component may be found in.

        This is the component's structural window from ``component_window`` -- its own extent
        widened either side by the drift the declared layout permits plus the budget this
        matcher searches with -- with the caller's ``start_idx`` applied on top as a floor:
        this matcher treats ``start_idx`` as a hard bound on where a match may begin, so the
        two bounds compose rather than replacing one another.

        The window is not a neighbour's territory being kept clear. It is the whole of where
        the layout can have put this component, which is why the last barcode of a chemistry
        is bounded on the right just as tightly as one with a barcode behind it, and why the
        primer separating two barcodes is not room either of them may wander into.

        Args:
            read: The sequencing read being searched.
            start_idx: A floor on where a match may begin, from the caller.

        Returns:
            ``(low, high)``, a half-open slice of ``read``. Empty (``low >= high``) when the
            read is too short to hold the component at all.
        """
        low, high = self.component_window(read, self.max_errors)
        return (max(low, min(start_idx, len(read))), high)

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

    def collect_candidates(self, read: str, start_idx: int = 0) -> list[AlignmentCandidate]:
        """
        Find every whitelist entry that both aligns and could plausibly be this component.

        Each entry is aligned once against the bounded search window. The alignment is used to
        *locate* the entry and nothing more: the span it reports is the span of matching bases,
        which local alignment clips at a terminal mismatch, so it comes back a base short
        exactly when the error sits at the component's edge. Reporting that span would sabotage
        the spacer check and shift the UMI in the same way a first-found k-mer window did, so
        the located entry is re-measured with ``best_window``, the same span-selection rule
        ``KmerMatcher`` uses.

        An entry then survives only if it passes both remaining gates. The span-length gate
        rejects a candidate whose extent in the read differs from the component by more than
        the error budget, since such a span cannot be this component however well it scores.
        The edit-distance gate is the one the contract is written in, and it is necessary
        because the score gate is not sufficient: an indel-bearing local alignment scores
        better than its edit distance implies, so a candidate can clear the score threshold and
        still be out of budget.

        No best-of filter is applied, so every in-budget candidate remains visible to
        resolution. That matters: dropping near misses is precisely how a tie became a "unique
        best alignment" and got assigned without ever being recognised as a tie.

        Args:
            read: The sequencing read to search.
            start_idx: A floor on where a match may begin, passed on to ``search_bounds``.

        Returns:
            Every surviving candidate, each carrying its attempt (in original-read
            coordinates, with no match assigned yet), its whitelist entry and its edit
            distance. Empty if nothing survived.
        """
        lo, hi = self.search_bounds(read, start_idx)
        window = read[lo:hi]

        if not window:
            log.debug(
                f"Search window for {self.component.name} is empty at start_idx {start_idx} "
                f"on a read of length {len(read)}."
            )
            return []

        bc_len = self.component.length
        candidates: list[AlignmentCandidate] = []

        for bc in self.whitelist_set:
            alignment = self.align_seqs(window, bc)
            if alignment is None:
                continue

            # Re-measure the located entry rather than trusting the aligned span. The band
            # best_window searches is centred on the alignment, in original-read coordinates,
            # so a base the aligner clipped is recovered while the search stays local to where
            # the alignment actually landed. It is floored at the window's own start, so
            # recovering that base cannot reach back past where the caller said a match may
            # begin -- this matcher treats start_idx as a hard floor and continues to. There is
            # no matching ceiling: a span may end up to max_errors past the window's end, which
            # is the price of recovering a clipped base from an alignment that landed against
            # that edge, and it is bounded by the span-length and edit-distance gates below.
            is_valid, start, end, ed = best_window(
                read, bc, alignment.seq1_span[0] + lo, self.max_errors, lower_bound=lo
            )
            if not is_valid:
                continue

            if abs(end - start - bc_len) > self.max_errors:
                continue

            candidates.append(
                AlignmentCandidate(
                    attempt=BarcodeMatchAttempt(
                        method=MatchMethod.ALIGNMATCH,
                        candidate=read[start:end],
                        match=None,  # Assigned only once resolution picks a winner
                        read_idx=(start, end),
                        edit_distance=None,
                    ),
                    bc=bc,
                    edit_distance=ed,
                )
            )

        return candidates

    def assign_match(self, candidate: AlignmentCandidate) -> list[BarcodeMatchAttempt]:
        """
        Assign a resolved candidate's whitelist entry onto its attempt.

        Args:
            candidate: The candidate resolution settled on.

        Returns:
            A single-element list holding the attempt with `match` and `edit_distance` set.
        """
        candidate.attempt.match = candidate.bc
        candidate.attempt.edit_distance = candidate.edit_distance
        return [candidate.attempt]

    def resolve_sole_candidate(self, candidate: AlignmentCandidate) -> list[BarcodeMatchAttempt]:
        """
        Decide whether a candidate that arrived alone has shown enough to be assigned.

        There is no tie here, so the spacer rungs have nothing to separate; what is being
        judged is whether the candidate is corroborated at all. The rule is on
        ``requires_spacer_evidence``: a candidate where the read structure predicts this
        component needs nothing further, a candidate somewhere else needs a flanking spacer.
        Assigning a lone candidate unexamined is what produced barcodes read off unrelated
        sequence.

        Args:
            candidate: The only candidate at the minimum edit distance, with its spacer
                evidence already recorded on its attempt.

        Returns:
            A single-element list: the assigned attempt, or a fresh matchless one.
        """
        read_idx = candidate.attempt.read_idx
        has_spacer = (
            candidate.attempt.spacer_upstream is not None
            or candidate.attempt.spacer_downstream is not None
        )

        if (
            not has_spacer
            and read_idx is not None
            and self.requires_spacer_evidence(read_idx, self.max_errors)
        ):
            log.debug(
                f"Sole alignment candidate {candidate.bc} at {read_idx} is away from the "
                f"expected start of {self.component.name} and has no adjacent spacer evidence. "
                "Marking as no match."
            )
            return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]

        log.debug(
            f"Unique best alignment match found: {candidate.bc} with edit distance "
            f"{candidate.edit_distance}"
        )
        return self.assign_match(candidate)

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Find this component's barcode in `read`, or decline to.

        Candidates are collected by ``collect_candidates`` and then resolved here. Resolution
        ranks on recomputed edit distance, never on alignment score, so that this matcher and
        ``KmerMatcher`` agree on what "equidistant" means. Each whitelist entry is aligned once,
        so two tied candidates are always two different entries, and a tie is therefore always
        the ambiguity the contract is about. Candidates tied at the minimum distance are
        separated only by the spacer rungs, in a fixed order: first by requiring at least one
        adjacent spacer, then by preferring the single candidate flanked by two.
        A candidate that arrives alone is no longer waved through unexamined; what it has to
        show is set out on ``requires_spacer_evidence``.

        There are four ways out, and they are the specification: a called match, a matchless
        attempt when nothing survives the gates, a matchless attempt when a lone candidate is
        away from its expected position with no spacer to support it, and several matchless
        attempts when candidates tie and no rung separates them. Only the last is an ambiguity
        verdict, which the caller reads off the number of attempts returned.

        Args:
            read: The sequencing read to match against.
            start_idx: A floor on where a match may begin. It is applied on top of the
                structural bound from ``search_bounds``, so bases before it are invisible to
                this matcher and no match can begin before it. Alignment coordinates are
                shifted back into original-read coordinates before being returned, so
                ``read_idx`` never needs the offset adding back.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results. A single
            attempt when a match is called or when nothing resolves, and one attempt per tied
            candidate, each with no match assigned, when the result is ambiguous.
        """
        candidates = self.collect_candidates(read, start_idx)

        if not candidates:
            log.debug(f"No valid alignment matches found for read starting at index {start_idx}.")
            return [BarcodeMatchAttempt(method=MatchMethod.ALIGNMATCH)]

        # Rank on edit distance, not on score.
        best_ed = min(candidate.edit_distance for candidate in candidates)
        best = [candidate for candidate in candidates if candidate.edit_distance == best_ed]

        # Spacer evidence is recorded for every tied candidate, including a lone one, so the
        # annotation and the stats show what the decision was taken on.
        for candidate in best:
            if candidate.attempt.read_idx is not None:
                spacers_check = self.check_spacers(read, candidate.attempt.read_idx)
                candidate.attempt.spacer_upstream = spacers_check["upstream"]
                candidate.attempt.spacer_downstream = spacers_check["downstream"]

        validated = [
            candidate
            for candidate in best
            if candidate.attempt.spacer_upstream is not None
            or candidate.attempt.spacer_downstream is not None
        ]

        if len(best) == 1:
            return self.resolve_sole_candidate(best[0])

        # Rung one: at least one adjacent spacer.
        if len(validated) == 1:
            log.debug(
                f"Unique best alignment match validated by spacers: {validated[0].bc} with edit "
                f"distance {best_ed}"
            )
            return self.assign_match(validated[0])

        # Rung two: exactly one candidate flanked by two spacers.
        both_spacers = [
            candidate
            for candidate in validated
            if candidate.attempt.spacer_upstream is not None
            and candidate.attempt.spacer_downstream is not None
        ]
        if len(validated) > 1 and len(both_spacers) == 1:
            log.debug(
                f"Unique best alignment match validated by having both spacers: "
                f"{both_spacers[0].bc} with edit distance {best_ed}"
            )
            return self.assign_match(both_spacers[0])

        # No rung separated them, so the read's window is genuinely equidistant from several
        # whitelist entries. Report every tied candidate, all matchless: that shape is what
        # tells the caller this is an ambiguity verdict rather than an empty search.
        if validated:
            log.debug(
                f"Ambiguous alignment match: candidate '{validated[0].attempt.candidate}' at edit "
                f"distance {best_ed} has multiple best matches. Will be marked as ambiguous."
            )
            return [candidate.attempt for candidate in validated]

        log.debug(
            f"Ambiguous alignment match with no spacer evidence: {len(best)} candidates tied at "
            f"edit distance {best_ed}. All will be marked as ambiguous."
        )
        return [candidate.attempt for candidate in best]
