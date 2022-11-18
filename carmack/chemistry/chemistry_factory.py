from .chemistry_hydrop import ChemistryHydrop

class ChemistryFactory:
    """Factory for chemistry objects."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_chemistry(chemistry_name: str):
        """Return chemistry object."""
        if chemistry_name == 'hydrop':
            return ChemistryHydrop()
        # elif chemistry_name == 'v2':
        #     return ChemistryV2()
        # elif chemistry_name == 'v1':
        #     return ChemistryV1()
        else:
            raise ValueError(f'Chemistry {chemistry_name} not supported.')
