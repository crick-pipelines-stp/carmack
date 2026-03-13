"""
Factory class for creating chemistry instances.
"""

from carmack.chemistry.chemistry_base import ChemistryBase


class ChemistryFactory:
    """
    Factory for creating chemistry instances by name.

    This class is not meant to be instantiated. Use the static method
    get_chemistry() to obtain chemistry instances.
    """

    registry: dict[str, type[ChemistryBase]] = {}

    def __init__(self):
        raise TypeError("ChemistryFactory is not meant to be instantiated.")

    @classmethod
    def register(cls, chemistry_class: type[ChemistryBase]) -> None:
        """
        Register a chemistry class with the factory.

        Raises:
            TypeError: If chemistry_class is not a subclass of ChemistryBase
        """
        if not (isinstance(chemistry_class, type) and issubclass(chemistry_class, ChemistryBase)):
            raise TypeError(
                f"chemistry_class must be a subclass of ChemistryBase, got {chemistry_class}"
            )
        cls.registry[chemistry_class().name] = chemistry_class

    @classmethod
    def get_chemistry(cls, chemistry_name: str) -> ChemistryBase:
        """
        Create and return a chemistry instance by name.

        Args:
            chemistry_name: The unique identifier for the chemistry

        Returns:
            An instance of the requested chemistry

        Raises:
            ValueError: If the chemistry name is not recognized
        """
        if chemistry_name not in cls.registry:
            available = ", ".join(cls.registry.keys())
            raise ValueError(
                f"Chemistry '{chemistry_name}' not supported. Available chemistries: {available}"
            )
        return cls.registry[chemistry_name]()

    @classmethod
    def list_chemistries(cls) -> list[str]:
        """
        Return a list of all registered chemistry names.

        Returns:
            List of available chemistry names
        """
        return list(cls.registry.keys())
