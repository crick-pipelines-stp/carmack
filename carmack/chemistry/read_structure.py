"""
ReadStructure class for managing read component sequences with easy navigation.
"""

from typing import Iterator

from carmack.chemistry.read_component import ReadComponent


class ReadStructure:
    """
    A wrapper around a list of ReadComponents that provides easy access to next and previous components and other useful methods.
    """

    def __init__(self, components: list[ReadComponent]):
        """
        Initialize the ReadStructure with a list of components.

        Args:
            components: List of ReadComponent instances.
        """
        self.components = components
        self._index_map = {comp.name: i for i, comp in enumerate(components)}

        self.compute_start_positions()

    def compute_start_positions(self) -> None:
        """
        Compute and set the start positions for each component based on their lengths.
        The first component starts at position 0, and subsequent components start immediately after the previous one.
        """
        current_position = 0
        for comp in self.components:
            comp.start = current_position
            current_position += comp.length

    def get_component_by_name(self, name: str) -> ReadComponent:
        """
        Get a component by its name.

        Args:
            name: The name of the component to retrieve.

        Returns:
            The ReadComponent with the specified name.
        """
        idx = self._index_map.get(name)

        if idx is None:
            raise ValueError(f"Component with name '{name}' not found in read structure")

        return self.components[idx]

    def get_known_sequences(self) -> dict[str, str]:
        """
        Get a dictionary of known sequences for components that have them defined.
        """
        known_sequences = {}
        for comp in self.components:
            if comp.sequence is not None:
                known_sequences[comp.name] = comp.sequence
        return known_sequences

    def get_next(self, component: ReadComponent) -> ReadComponent | None:
        """
        Get the next component after the given component.

        Args:
            component: The component to find the next one for.

        Returns:
            The next ReadComponent or None if it's the last one or not found.
        """
        idx = self._index_map.get(component.name)
        if idx is not None and idx + 1 < len(self.components):
            return self.components[idx + 1]
        return None

    def get_previous(self, component: ReadComponent) -> ReadComponent | None:
        """
        Get the previous component before the given component.

        Args:
            component: The component to find the previous one for.

        Returns:
            The previous ReadComponent or None if it's the first one or not found.
        """
        idx = self._index_map.get(component.name)
        if idx is not None and idx - 1 >= 0:
            return self.components[idx - 1]
        return None

    def __iter__(self) -> Iterator[ReadComponent]:
        """Iterate over the components."""
        return iter(self.components)

    def __getitem__(self, index: int) -> ReadComponent:
        """Get a component by index."""
        return self.components[index]

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self.components)
