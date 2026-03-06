import logging
from collections.abc import Mapping

from carmack.barcode.extraction_dataclasses import (
    BarcodeMatchHistory,
    ChemistryBase,
    MatchMethod,
    ReadMatchResult,
)
from carmack.barcode.matchers.matcher_base import MatcherBase


log = logging.getLogger(__name__)


class HybridExtractor:
    """
    Perform barcode matching using defined matchers for each barcode component.

    Returns:
        ReadMatchResult containing the matched barcodes and their statuses.
    """

    def __init__(
        self,
        chemistry: ChemistryBase,
        matchers: dict[MatchMethod, Mapping[str, MatcherBase]],
    ):
        self.chemistry = chemistry
        self.matchers = matchers

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

        search_start_idx_tracker: dict[str, int] = {
            barcode_component.name: 0 for barcode_component in barcode_components
        }
        barcode_match_tracker: dict[str, BarcodeMatchHistory] = {
            barcode_component.name: BarcodeMatchHistory(barcode_component.name)
            for barcode_component in barcode_components
        }

        for matcher_method, matcher in self.matchers.items():
            # Validate matcher method
            if matcher_method not in MatchMethod:
                raise ValueError(f"Unknown matcher method '{matcher_method}'")

            # Iterate through barcode components in the order defined by the read structure
            for barcode_component in barcode_components:
                bc_name: str = barcode_component.name

                # Determine the end of the previous matched barcode
                # This will be used to trim the read to reduce search space
                prev_bc = self.chemistry.read_structure.get_previous(
                    barcode_component, bc_only=True
                )
                prev_bc_name = prev_bc.name if prev_bc else None
                search_start_idx = search_start_idx_tracker[prev_bc_name] if prev_bc_name else 0

                if search_start_idx >= len(read) - barcode_component.length:
                    search_start_idx = 0  # Reset to start if we've gone beyond the read length

                if bc_name not in matcher:
                    raise ValueError(
                        f"Matcher '{matcher_method}' does not have a matcher for barcode component '{barcode_component.name}'"
                    )

                # Start matching
                if matcher_method == MatchMethod.EXACTMATCH:
                    # No ambiguous barcode handling needed for fixed position matcher, so we can directly record the match attempts
                    results = matcher[bc_name].match(read)
                else:
                    # Skip other matchers if barcode extraction succedeed for this barcode component
                    if barcode_match_tracker[bc_name].success:
                        continue

                    # Handle other matcher types if any
                    results = matcher[bc_name].match(read, start_idx=search_start_idx)

                if len(results) == 1 and results[0].match is not None:
                    result = results[0]
                    barcode_match_tracker[bc_name].record_attempt(result, success=True)

                    # Update search start index with the end of the matched barcode for the next matcher
                    if result.read_idx:
                        search_start_idx_tracker[bc_name] = result.read_idx[1]
                else:
                    # If multiple attempts, we have ambiguity. We record all attempts but mark success as False.
                    for r in results:
                        barcode_match_tracker[bc_name].record_attempt(r, success=False)

        # Compile final result for the read
        read_result = ReadMatchResult(
            read_name=read_name,
            read=read,
            qual=qual,
            chemistry=self.chemistry,
            bc_results=list(barcode_match_tracker.values()),
        )
        return read_result
