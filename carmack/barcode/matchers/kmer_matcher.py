import logging
from collections import defaultdict
from typing import ClassVar

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_structure import ReadComponent, ReadComponentType

log = logging.getLogger(__name__)

# A whitelist entry verified against a window of the read, in original-read coordinates.
type KmerCandidate = tuple[str, int, int, int]  # (entry, start, end, edit_distance)


class KmerMatcher(MatcherBase):
    """
    Kmer seed-and-extend matching against a whitelist.

    Uses k-mer seeding to identify candidate sequences, then verifies
    with Levenshtein distance. Handles indels within the max_errors budget.

    Beyond barcodes, this matcher also accepts TGIDX components. That widening is safe
    precisely because the matcher never reads ``component.start``: it searches the read for
    seed k-mers instead of indexing a fixed window, so a component whose start is unresolved
    is located just as well as one whose start is known.
    """

    allowed_component_types: ClassVar[frozenset[ReadComponentType]] = frozenset(
        {ReadComponentType.BARCODE, ReadComponentType.TGIDX}
    )

    def __init__(
        self,
        whitelist: tuple[str, ...],
        component: ReadComponent,
        chemistry: ChemistryBase,
        max_errors: int,
        k: int = 4,
    ):
        """
        Initialize the KmerMatcher with the given whitelist and component.

        Args:
            whitelist: Tuple of valid sequences to match against.
            component: The ReadComponent this matcher is matching, defining its length and name.
            chemistry: The chemistry defining the surrounding read structure.
            max_errors: Maximum edit distance allowed against a whitelist entry. Supplied by the
                caller so the matcher does not reach into the chemistry for a budget that may not
                be the barcode one.
            k: Length of k-mers to use for seeding.
        """
        super().__init__(whitelist, component, chemistry)
        self.k = k
        self.max_errors = max_errors

        self.kmer_index = self.build_kmer_index()

        log.debug(
            f"KmerMatcher initialized for {self.component.name} with k={k}, max_errors={self.max_errors}, {len(whitelist)} barcodes"
        )

    def build_kmer_index(self) -> dict[str, list[tuple[str, int]]]:
        """
        Build an index mapping k-mers to (barcode, position) tuples.

        Args:
            whitelist: Tuple of barcode sequences
            k: k-mer size

        Returns:
            Dictionary mapping each k-mer to list of (barcode, start_position) tuples
        """
        log.debug(f"Building k-mer index for {len(self.whitelist_set)} barcodes with k={self.k}")
        index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for bc in self.whitelist_set:
            for i in range(len(bc) - self.k + 1):
                kmer = bc[i : i + self.k]
                index[kmer].append((bc, i))
        log.debug(f"K-mer index built with {len(index)} unique k-mers")
        return dict(index)

    def extend_and_verify(
        self,
        read: str,
        read_kmer_pos: int,
        barcode: str,
        bc_kmer_pos: int,
    ) -> tuple[bool, int, int, int]:
        """
        Extend a seed match and verify the full barcode alignment.

        Args:
            read: The read sequence
            read_kmer_pos: Position of k-mer match in read
            barcode: The barcode sequence
            bc_kmer_pos: Position of k-mer in barcode

        Returns:
            (is_valid, start_pos, end_pos, edit_distance) tuple
        """
        bc_len = len(barcode)
        expected_start = read_kmer_pos - bc_kmer_pos

        # Search within a small band around expected start
        min_start = max(0, expected_start - self.max_errors)
        max_start = min(len(read) - 1, expected_start + self.max_errors)

        # Only consider window sizes where |win_len - bc_len| <= max_errors
        # This is the key pruning: window length alone must allow <= max_errors
        min_len = max(1, bc_len - self.max_errors)
        max_len = min(len(read), bc_len + self.max_errors)

        best_dist = self.max_errors + 1
        best_span = (-1, -1)

        for start in range(min_start, max_start + 1):
            max_len_at_start = min(max_len, len(read) - start)
            if max_len_at_start < min_len:
                continue

            # Only iterate window lengths within feasible range
            for win_len in range(min_len, max_len_at_start + 1):
                read_window = read[start : start + win_len]

                dist = edit_distance(read_window, barcode, "N", True)

                if dist < best_dist:
                    best_dist = dist
                    best_span = (start, start + win_len)

                    # Early exit: perfect match found
                    if best_dist == 0:
                        return (True, best_span[0], best_span[1], best_dist)

        if best_dist <= self.max_errors:
            return (True, best_span[0], best_span[1], best_dist)

        return (False, -1, -1, best_dist)

    def collect_candidates(self, read: str, start_idx: int = 0) -> list[KmerCandidate]:
        """
        Find every whitelist entry that verifies somewhere in the read.

        This is the collection half of ``match``: it scans the read for seed k-mers, extends each
        distinct seed and keeps the ones that verify. It deliberately does not choose between
        them — no best-score filter is applied here — so the near misses that resolution is about
        to discard remain visible to a caller that wants them.

        Args:
            read: The sequencing read to search.
            start_idx: A floor on where a seed k-mer may begin, not on where a candidate may
                begin. The read is never trimmed, so a candidate seeded at or after the floor can
                still span back before it. Defaults to 0, which imposes no bound.

        Returns:
            Every verified candidate, in the order it was verified. Empty if nothing verified.

        Raises:
            ValueError: If the read is shorter than the seed k-mer length.
        """
        if len(read) < self.k:
            raise ValueError(
                f"Read segment too short for k-mer matching: read length {len(read)}, k={self.k}"
            )

        candidates: list[KmerCandidate] = []
        # To avoid redundant verification of same (barcode, position)
        candidates_seen: set[tuple[str, int]] = set()

        # Scan read for seed k-mers
        for i in range(start_idx, len(read) - self.k + 1):
            read_kmer = read[i : i + self.k]

            if read_kmer in self.kmer_index:
                for bc, bc_kmer_pos in self.kmer_index[read_kmer]:
                    expected_start = i - bc_kmer_pos
                    candidate_key = (bc, expected_start)

                    if candidate_key in candidates_seen:
                        continue
                    candidates_seen.add(candidate_key)

                    is_valid, start, end, edit_dist = self.extend_and_verify(
                        read, i, bc, bc_kmer_pos
                    )

                    if is_valid:
                        candidates.append((bc, start, end, edit_dist))

        return candidates

    def resolve_candidates(
        self, read: str, candidates: list[KmerCandidate]
    ) -> list[BarcodeMatchAttempt]:
        """
        Choose between collected candidates and report the outcome as match attempts.

        This is the resolution half of ``match``. Candidates are first filtered to the best edit
        distance, since choosing between them is what resolution is for. A lone survivor is
        reported directly with no spacer validation at all, because spacers serve only as a
        tie-break and there is no tie to break. A tie is broken first by requiring at least one
        adjacent spacer, and then, if several candidates still stand, by preferring the single
        candidate flanked by two. If neither rung separates them the matcher declines to guess.

        Args:
            read: The sequencing read the candidates were collected from, used both to slice out
                each candidate sequence and to inspect the flanking spacers.
            candidates: Verified candidates, as returned by ``collect_candidates``. They are
                assumed to already be within the matcher's error budget, as
                ``collect_candidates`` guarantees; a candidate outside that budget would be
                resolved rather than rejected here.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results. A single attempt
            when a match is called or when nothing resolves, and one attempt per validated
            candidate, each with no match assigned, when the result is ambiguous.
        """
        # Intit a match attempt with method KMER
        result = BarcodeMatchAttempt(method=MatchMethod.KMERMATCH)

        # Filter candidates to those with best score. Seeding the minimum one past the error
        # budget keeps a candidate outside the budget from becoming the best score on its own.
        best_score = min([c[3] for c in candidates] + [self.max_errors + 1])
        best_candidates = [c for c in candidates if c[3] == best_score]

        if len(best_candidates) == 0:
            return [result]  # No valid candidates found

        if len(best_candidates) == 1:
            best_bc, start, end, edit_dist = best_candidates[0]
            result.match = best_bc
            result.candidate = read[start:end]
            result.read_idx = (start, end)
            result.edit_distance = edit_dist
            log.debug(
                f"Kmer match found: {best_bc} at position {start}-{end} with edit distance {edit_dist}"
            )
        else:
            # Check candidates for ambiguity based on spacer presence
            validated_candidates: list[tuple[str, int, int, int, dict]] = []

            for candidate in best_candidates:
                bc, start, end, edit_dist = candidate
                spacers_check = self.check_spacers(read, (start, end))

                # Validate candidates if any adjacent spacer is present
                if any(spacers_check.values()):
                    validated_candidates.append((bc, start, end, edit_dist, spacers_check))

            if len(validated_candidates) == 0:
                log.debug("No validated candidates could be found.")
                return [result]
            elif len(validated_candidates) == 1:
                best_bc, start, end, edit_dist, spacers_check = validated_candidates[0]
            else:
                # If still ambiguous, check if only one of the validated candidate has two spacers present, which would make it more likely to be correct
                # If more than one candidate has two adjacent spacers, then we have to consider it ambiguous and cannot confidently call a single best match
                best_candidates_with_two_spacers = [
                    vc
                    for vc in validated_candidates
                    if sum(v is not None for v in vc[4].values()) == 2
                ]

                if len(best_candidates_with_two_spacers) == 1:
                    best_bc, start, end, edit_dist, spacers_check = (
                        best_candidates_with_two_spacers[0]
                    )
                else:
                    log.debug(
                        f"Ambiguous kmer matches found: {validated_candidates}. Multiple candidates with two adjacent spacers."
                    )
                    ambiguous_attempts: list[BarcodeMatchAttempt] = []
                    for vc in validated_candidates:
                        ambiguous_attempts.append(
                            BarcodeMatchAttempt(
                                method=MatchMethod.KMERMATCH,
                                candidate=read[vc[1] : vc[2]],
                                match=None,
                                read_idx=(vc[1], vc[2]),
                                edit_distance=vc[3],
                                spacer_upstream=vc[4].get("upstream"),
                                spacer_downstream=vc[4].get("downstream"),
                            )
                        )
                    return ambiguous_attempts

            result.match = best_bc
            result.candidate = read[start:end]
            result.read_idx = (start, end)
            result.edit_distance = edit_dist
            result.spacer_upstream = spacers_check.get("upstream")
            result.spacer_downstream = spacers_check.get("downstream")
            log.debug(
                f"Kmer match found with spacer check: {best_bc} at position {start}-{end} with edit distance {edit_dist}, spacers: {spacers_check}"
            )

        return [result]

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read based on kmer matching.

        Args:
            read: The sequencing read to match against.
            start_idx: A floor on where a seed k-mer may begin, not on where a match may begin.
                The read is never trimmed. Seeds are scanned from ``start_idx`` onwards and each
                seed is then extended in both directions from its implied start, clamped only at
                0, so a match seeded at or after ``start_idx`` can begin before it — as early as
                ``max(0, start_idx - (len(entry) - k) - max_errors)``. Returned ``read_idx``
                values are already in original-read coordinates because no trimming occurred.

        Returns:
            List of BarcodeMatchAttempt objects representing the match results.

        Raises:
            ValueError: If the read is shorter than the seed k-mer length.
        """
        candidates = self.collect_candidates(read, start_idx)
        return self.resolve_candidates(read, candidates)
