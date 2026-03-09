import logging
from collections import defaultdict

from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.read_structure import ReadComponent


log = logging.getLogger(__name__)


class KmerMatcher(MatcherBase):
    """
    Kmer seed-and-extend barcode matching.

    Uses k-mer seeding to identify candidate barcodes, then verifies
    with Levenshtein distance. Handles indels within max_errors.
    """

    def __init__(
        self,
        whitelist: tuple[str, ...],
        barcode_component: ReadComponent,
        chemistry: ChemistryBase,
        k: int = 4,
    ):
        """
        Initialize the KmerMatcher with the given whitelist and barcode component.

        Args:
            whitelist: Tuple of valid barcode sequences.
            barcode_component: The ReadComponent defining the barcode's position and length.
            k: Length of k-mers to use for seeding.
        """
        super().__init__(whitelist, barcode_component, chemistry)
        self.k = k
        self.max_errors = chemistry.max_errors.barcode

        self.kmer_index = self.build_kmer_index()

        log.debug(
            f"KmerMatcher initialized for {self.barcode_component.name} with k={k}, max_errors_barcode={self.max_errors}, {len(whitelist)} barcodes"
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

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read based on kmer matching.

        Args:
            read: The sequencing read to match against.
            start_idx: The index in the read to start matching from (default is 0).
        """
        # Intit a match attempt with method KMER
        result = BarcodeMatchAttempt(method=MatchMethod.KMERMATCH)

        if len(read) < self.k:
            raise ValueError(
                f"Read segment too short for k-mer matching: read length {len(read)}, k={self.k}"
            )

        # Track best match and all candidates
        best_score = self.max_errors + 1
        candidates: list[tuple[str, int, int, int]] = (
            []
        )  # (barcode, start_pos, end_pos, edit_distance)
        candidates_seen: set[tuple[str, int]] = (
            set()
        )  # To avoid redundant verification of same (barcode, position)

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

                        if edit_dist < best_score:
                            best_score = edit_dist

        # Filter candidates to those with best score
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
                    c
                    for c in validated_candidates
                    if sum(v is not None for v in c[4].values()) == 2
                ]

                if len(best_candidates_with_two_spacers) == 1:
                    best_bc, start, end, edit_dist, spacers_check = (
                        best_candidates_with_two_spacers[0]
                    )
                else:
                    log.debug(
                        f"Ambiguous kmer matches found: {validated_candidates}. Multiple candidates with two adjacent spacers."
                    )
                    result = []
                    for c in validated_candidates:
                        result.append(
                            BarcodeMatchAttempt(
                                method=MatchMethod.KMERMATCH,
                                candidate=read[c[1] : c[2]],
                                match=None,
                                read_idx=(c[1], c[2]),
                                edit_distance=c[3],
                                spacer_upstream=c[4].get("upstream"),
                                spacer_downstream=c[4].get("downstream"),
                            )
                        )
                    return result

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
