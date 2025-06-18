from .chemistry_base import ChemistryBase
from .chemistry_hydrop import ChemistryHydrop
from .chemistry_carmack_custom_seq_1_0 import ChemistryCarmackCustomSeq10


class ChemistryFactory:
    """Factory for chemistry objects."""

    def __init__(self) -> None:
        raise TypeError("ChemistryFactory is not meant to be instantiated.")

    @staticmethod
    def get_chemistry(chemistry_name: str) -> ChemistryBase:
        """Return chemistry object."""
        if chemistry_name == "hydrop":
            return ChemistryHydrop()
        elif chemistry_name == "carmack_custom_seq_1_0":
            return ChemistryCarmackCustomSeq10()
        else:
            raise ValueError(f"Chemistry {chemistry_name} not supported.")
