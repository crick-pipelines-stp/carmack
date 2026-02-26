import logging

from carmack.barcode.extraction_dataclasses import BarcodeMatchAttempt, MatchMethod
from carmack.barcode.matchers.matcher_base import MatcherBase


log = logging.getLogger(__name__)


class FixedPositionMatcher(MatcherBase):
    """
    Fixed position barcode matching.

    Attempts to match barcodes at their expected positions in the read.
    This is the fastest method but requires no indels in the read.
    """

    def match(self, read: str, start_idx: int = 0) -> list[BarcodeMatchAttempt]:
        """
        Attempt to match barcodes in the given read based on fixed positions.

        Args:
            read: The sequencing read to match against.
            start_idx: The index in the read to start matching from (default is 0). Only used if the
            barcode component does not have a defined start position.
        """
        start = self.barcode_component.start or start_idx
        end = start + self.barcode_component.length
        candidate = read[start:end]

        # Init a match attempt
        result = BarcodeMatchAttempt(candidate=candidate, method=MatchMethod.EXACTMATCH)

        # Check if read is long enough
        if not self.check_read_len(read):
            log.debug(
                f"Read too short for fixed position matching: read length {len(read)}, required {end}"
            )
            return [result]

        if candidate in self.whitelist_set:
            log.debug(f"Fixed position match found: {candidate} at position {start}-{end}")
            # No ambiguity possible for exact matches, so we can directly record the match
            result.match = candidate
            result.read_idx = (start, end)
        else:
            log.debug(f"No fixed position match: {candidate} not in whitelist")

        return [result]
