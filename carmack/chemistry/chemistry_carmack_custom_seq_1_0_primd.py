"""
Carmack Custom Sequencing 1.0 chemistry definition with PRIMER_D.

Read structure (5' to 3'):
PRIMER_D (22bp) -> BC3 (10bp) -> PRIMER_C (22bp) -> BC2 (10bp) -> PRIMER_A (22bp) -> BC1 (10bp)
-> ...

Identical to ChemistryCarmackCustomSeq10 except for the leading PRIMER_D
component, whose sequence is unknown so only its length is used as an anchor.
"""

from functools import cached_property

from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (
    PRIMER_C,
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType


class ChemistryCarmackCustomSeq10PrimD(ChemistryCarmackCustomSeq10):
    """
    Carmack Custom Sequencing 1.0 chemistry with an additional PRIMER_D
    component at the 5' end.
    """

    @cached_property
    def name(self) -> str:
        """Return the unique identifier for this chemistry."""
        return "carmack_custom_seq_1_0_primd"

    def _build_components(self) -> list[ReadComponent]:
        return [
            ReadComponent(
                name="PRIMER_D", type=ReadComponentType.PRIMER, length=len(PRIMER_C)
            ),
            *super()._build_components(),
        ]


ChemistryFactory.register(ChemistryCarmackCustomSeq10PrimD)
