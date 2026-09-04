"""Guard tests for the canonical cell-barcode SAM tag convention.

The single-cell canonical cell-barcode tag is ``CB`` (with raw ``CR``). No
production module may read or write the legacy ``BC`` SAM tag.
"""

import re
from pathlib import Path

from assertpy import assert_that

# Matches the two-letter ``BC`` SAM tag used with a pysam tag accessor, e.g.
# ``set_tag("BC", ...)`` / ``get_tag('BC')`` / ``has_tag("BC")``. It deliberately
# does not match barcode component names (BC1/BC2/BC3) or ``bc_dict`` variables.
BC_SAM_TAG = re.compile(r"""(?:set_tag|get_tag|has_tag)\(\s*["']BC["']""")

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent / "carmack"


class TestBarcodeTagConvention:
    def test_no_production_module_uses_the_bc_sam_tag(self) -> None:
        """No production source file reads or writes the legacy ``BC`` SAM tag."""
        offenders = [
            str(path.relative_to(PRODUCTION_ROOT.parent))
            for path in PRODUCTION_ROOT.rglob("*.py")
            if BC_SAM_TAG.search(path.read_text())
        ]
        assert_that(offenders).described_as("modules still using the BC SAM tag").is_empty()
