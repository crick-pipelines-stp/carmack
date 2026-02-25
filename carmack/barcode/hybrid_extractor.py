import logging

from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchHistory,
    ChemistryBase,
    ReadMatchResult,
)
from carmack.barcode.matchers.matcher_base import MatcherBase


log = logging.getLogger(__name__)


class HybridExtractor:
    def __init__(
        self,
        chemistry: ChemistryBase,
        matchers: dict[str, dict[str, MatcherBase]],
    ):
        self.chemistry = chemistry
        self.matchers = matchers

    def trim_read(self, read: str, start: int) -> str:
        """
        Trim the read to the expected length for the barcode component.

        This is used to ensure that the read segment being matched is of the correct length, which can
        help improve matching accuracy and reduce false positives.
        """
        if start >= len(read):
            log.debug(
                f"Start index {start} is beyond read length {len(read)}. Returning empty string."
            )
            return ""
        return read[start:]

    def process_read(self, read_name: str, read: str, qual: str) -> ReadMatchResult:
        """
        Process a single read to extract barcodes.

        This function applies the defined matchers, and records the
        matching history for each barcode component. The final result includes the matched barcodes
        and their statuses.
        """
        barcode_components = [
            comp for comp in self.chemistry.read_structure.components if comp.is_barcode
        ]

        search_start_idx: dict[str, int] = {
            barcode_component.name: 0 for barcode_component in barcode_components
        }
        barcode_match_tracker: dict[str, BarcodeMatchHistory] = {
            barcode_component.name: BarcodeMatchHistory(barcode_component.name)
            for barcode_component in barcode_components
        }

        for matcher_name, matcher in self.matchers.items():
            for barcode_component in barcode_components:
                bc_name: str = barcode_component.name
                if bc_name not in matcher:
                    raise ValueError(
                        f"Matcher '{matcher_name}' does not have a matcher for barcode component '{barcode_component.name}'"
                    )

                if matcher_name == "fixed":
                    # No ambiguous barcode handling needed for fixed position matcher, so we can directly record the match attempts
                    result = matcher[bc_name].match(read)
                else:
                    # Skip other matchers if barcode extraction succedeed for this barcode component
                    if barcode_match_tracker[bc_name].success:
                        continue

                    # Get trimmed read for matching, if applicable
                    trimmed_read = self.trim_read(read, search_start_idx[bc_name])

                    # Handle other matcher types if any
                    result = matcher[bc_name].match(trimmed_read)

                    # Need to handle ambigous matchers here
                    ...

                success = result.match is not None
                barcode_match_tracker[bc_name].record_attempt(result, success=success)

                # Update search start index with the end of the matched barcode for the next matcher
                if success and result.read_idx:
                    search_start_idx[barcode_component.name] = result.read_idx[1]

        # Compile final result for the read
        read_result = ReadMatchResult(
            read_name=read_name,
            read=read,
            qual=qual,
            chemistry=self.chemistry,
            bc_results=list(barcode_match_tracker.values()),
        )
        return read_result
