"""
Data structures for barcode extraction results.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from carmack.chemistry.chemistry_base import ChemistryBase


class MatchMethod(Enum):
    """Enumeration of barcode matching methods."""

    EXACTMATCH = "EXACTMATCH"
    KMERMATCH = "KMERMATCH"
    ALIGNMATCH = "ALIGNMATCH"


@dataclass
class BarcodeMatchAttempt:
    """
    Records the outcome of a single matching stage attempt.

    Attributes:
        candidate: The candidate barcode sequence from the read that was attempted to be matched.
        match: The corresponding whitelist barcode sequence that the candidate matched to, if any. None if no match was found.
        read_idx: The start and end indices in the read where the candidate was found. None if not applicable.
        method: The matching method used in this attempt
        edit_distance: The edit distance of the candidate to the closest whitelist entry, if applicable. None if not applicable, such as for exact matches.
        spacer_upstream: Optional spacer name of the upstream to the match, if searched for.
        spacer_downstream: Optional spacer name of the downstream to the match, if searched for.
    """

    candidate: str
    method: MatchMethod
    match: str | None = None
    read_idx: tuple[int, int] | None = None
    edit_distance: int | None = None
    spacer_upstream: str | None = None
    spacer_downstream: str | None = None


@dataclass
class BarcodeMatchHistory:
    """
    Tracks all matching attempts for a single barcode component across pipeline stages.

    Attributes:
        bc_name: Barcode component name (e.g., "BC1")
        attempts: Ordered list of failed attempts, from first stage to last
        success: If the final attempt was successful, this holds the successful result. Otherwise False.
    """

    bc_name: str
    attempts: list[BarcodeMatchAttempt] = field(default_factory=list)
    success: bool = False

    def record_attempt(self, attempt: BarcodeMatchAttempt, success: bool = False) -> None:
        """Append a attempt to the history."""
        self.attempts.append(attempt)

        if success:
            self.success = True

    @property
    def succeeded_at(self) -> MatchMethod | None:
        """Return the method at which matching succeeded, or None."""
        if self.success and self.attempts:
            return self.attempts[-1].method
        return None

    @property
    def ambiguous_matches(self) -> dict[MatchMethod, bool]:
        """
        Return a dict indicating whether each method had ambiguous matches that failed tiebreaking.

        We only record ambigous BarcodeMatchAttempts if spacer validation failed.
        """
        methods = [a.method for a in self.attempts if a.match is not None]

        counts = Counter(methods)

        return {method: count > 1 for method, count in counts.items()}

    def to_status_string(self) -> str:
        """
        Generate status string for this barcode component. Useful for read name annotations.

        Format: BCX:METHODA-STATUS:METHODB-STATUS-...
        """
        status = f"{self.bc_name}"
        for attempt in self.attempts:
            if attempt.match is not None:
                status += f":{attempt.method.value}"

                if attempt.edit_distance is not None:
                    status += f"-ED{attempt.edit_distance}"
                if attempt.spacer_upstream:
                    status += "-spUp"
                if attempt.spacer_downstream:
                    status += "-spDown"
            else:
                status += ":NOMATCH"

        return status


@dataclass
class ReadMatchResult:
    """
    Complete result of barcode extraction for a single read.

    Attributes:
        read_name: Original read name
        bc_results: List of BarcodeMatchHistory for each barcode component, in the order defined by the read structure.
        chemistry: The chemistry used for matching, which defines the read structure and whitelists.
        full_barcode: Concatenated full barcode if all components matched, else None
    """

    read_name: str
    read: str
    qual: str
    chemistry: ChemistryBase
    bc_results: list[BarcodeMatchHistory] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether all barcode components were successfully matched."""
        return all(bc.success for bc in self.bc_results)

    @property
    def is_perfect(self) -> bool:
        """Whether all barcode components were matched with exact matches."""
        return self.success and all(
            bc.succeeded_at == MatchMethod.EXACTMATCH for bc in self.bc_results
        )

    @property
    def full_barcode(self) -> str | None:
        """Concatenate the matched barcode components into a full barcode string, if all succeeded. Otherwise None."""
        if not self.success:
            return None

        result_dict = {
            bc.bc_name: bc.attempts[-1].match
            for bc in self.bc_results
            if bc.attempts and bc.attempts[-1].match is not None
        }
        return self.chemistry.construct_full_barcode(result_dict)

    def get_annotated_readname(self) -> str:
        """Generate an annotated read name with matching status of each barcode component."""
        status_parts = [bc.to_status_string() for bc in self.bc_results]
        main_status = "SUCCESS" if self.success else "FAIL"
        if self.success:
            main_status += ":PERFECT" if self.is_perfect else ":CORROK"  # Correction-OK
            main_status += f":{self.full_barcode}"
        return f"{self.read_name}|{main_status}|{'|'.join(status_parts)}"
