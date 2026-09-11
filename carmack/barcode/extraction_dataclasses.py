"""
Data structures for barcode extraction results.
"""

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

    method: MatchMethod
    candidate: str | None = None
    match: str | None = None
    read_idx: tuple[int, int] | None = None
    edit_distance: int | None = None
    spacer_upstream: str | None = None
    spacer_downstream: str | None = None


@dataclass
class BarcodeMatchHistory:
    """
    Tracks all matching attempts for a single barcode component across pipeline stages.

    A component ends in one of three states, and they are deliberately three rather than two.
    "Matched", "no candidate was found" and "candidates were found and could not be separated"
    are different answers, and the last one is a *verdict*: the read's barcode window is
    genuinely equidistant from two whitelist entries, so no amount of further searching can
    honestly resolve it. Storing that as plain failure, as this used to, let a later matcher
    run on precisely the reads an earlier one had declared unresolvable and pick one of them.

    Attributes:
        bc_name: Barcode component name (e.g., "BC1")
        attempts: Ordered list of attempts, from first stage to last
        success: Whether any attempt called a barcode.
        ambiguous_at: The method that declared this component unresolvable, or None if no
            method did. Latches on the first such verdict, since ambiguity is terminal.
    """

    bc_name: str
    attempts: list[BarcodeMatchAttempt] = field(default_factory=list)
    success: bool = False
    ambiguous_at: MatchMethod | None = None

    def record_attempt(
        self, attempt: BarcodeMatchAttempt, success: bool = False, ambiguous: bool = False
    ) -> None:
        """
        Append an attempt to the history.

        Args:
            attempt: The attempt to record.
            success: Whether this attempt called a barcode.
            ambiguous: Whether this attempt is one of several equally close candidates the
                matcher could not separate. Recorded on the history rather than the attempt
                because ambiguity is a property of the *set* of attempts a matcher returned,
                not of any one of them.
        """
        self.attempts.append(attempt)

        if success:
            self.success = True

        if ambiguous and self.ambiguous_at is None:
            self.ambiguous_at = attempt.method

    @property
    def is_ambiguous(self) -> bool:
        """Whether some matcher found candidates it could not separate."""
        return self.ambiguous_at is not None

    @property
    def is_terminal(self) -> bool:
        """
        Whether this component's verdict is final, so no later matcher should be run on it.

        A called barcode is final for the obvious reason. An ambiguity verdict is final
        because it is an answer, not an absence: a later matcher ranking by a different
        criterion would separate candidates the contract says must not be separated.
        """
        return self.success or self.is_ambiguous

    @property
    def succeeded_at(self) -> MatchMethod | None:
        """Return the method at which matching succeeded, or None."""
        if self.success and self.attempts:
            return self.attempts[-1].method
        return None

    def to_status_string(self) -> str:
        """
        Generate status string for this barcode component. Useful for read name annotations.

        Format: BCX:METHODA-STATUS:METHODB-STATUS-...:VERDICT

        A component that resolved carries no verdict token; the per-method detail is the
        answer. A component that did not resolve carries one of two verdict tokens, and they
        mean different things. NOMATCH is "no candidate was ever found". METHOD-AMBIG is
        "that method found candidates and could not separate them", which is a read the
        pipeline declined to guess at rather than one it lost. Rendering both as NOMATCH, as
        this used to, made the two indistinguishable to anything reading a read name.
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

        if not self.success:
            status += f":{self.ambiguous_at.value}-AMBIG" if self.is_ambiguous else ":NOMATCH"

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

    def get_attempts(self, bc_name: str, method: MatchMethod) -> list[BarcodeMatchAttempt]:
        """Get all attempts for a given barcode component and method."""
        for bc_history in self.bc_results:
            if bc_history.bc_name == bc_name:
                return [a for a in bc_history.attempts if a.method == method]
        return []

    def get_annotated_readname(self) -> str:
        """Generate an annotated read name with matching status of each barcode component."""
        status_parts = [bc.to_status_string() for bc in self.bc_results]
        main_status = "SUCCESS" if self.success else "FAIL"
        if self.success:
            main_status += ":PERFECT" if self.is_perfect else ":CORROK"  # Correction-OK
            main_status += f":{self.full_barcode}"
        return f"{self.read_name}|{main_status}|{'|'.join(status_parts)}"
