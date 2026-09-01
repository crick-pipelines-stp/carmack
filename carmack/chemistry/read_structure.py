"""
ReadStructure class for managing read component sequences with easy navigation.
"""

from typing import Iterator

from carmack.chemistry.read_component import ReadComponent, ReadComponentType


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

        The walk keeps a running position starting at 0. Fixed-length components
        retain exact starts, and the first variable-length (or unknown-length)
        component still receives its concrete start. Every component after that
        first variable component is left with ``start = None`` because its
        position can only be resolved per-read.
        """
        current_position: int | None = 0
        for comp in self.components:
            comp.start = current_position
            if current_position is None:
                continue
            if comp.is_variable_length or comp.length is None:
                current_position = None
            else:
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

    def get_components_by_type(self, component_type: ReadComponentType) -> list[ReadComponent]:
        if not isinstance(component_type, ReadComponentType):
            raise TypeError(
                f"component_type must be a ReadComponentType enum member, got {type(component_type).__name__}"
            )

        matches = [comp for comp in self.components if comp.type is component_type]
        if not matches:
            seen: list[ReadComponentType] = []
            for comp in self.components:
                if comp.type not in seen:
                    seen.append(comp.type)
            available = ", ".join(str(t) for t in seen) if seen else "none"
            raise ValueError(
                f"ReadStructure does not contain any components of type '{component_type}'. Available types in read order: {available}"
            )

        return matches

    def get_next(self, component: ReadComponent, bc_only: bool = False) -> ReadComponent | None:
        """
        Get the next component after the given component.

        Args:
            component: The component to find the next one for.
            bc_only: If True, only consider barcode components when looking for the next component.

        Returns:
            The next ReadComponent or None if it's the last one or not found.
        """
        idx = self._index_map.get(component.name)
        if idx is not None and idx + 1 < len(self.components):
            next_comp = self.components[idx + 1]
            if bc_only and next_comp.type is not ReadComponentType.BARCODE:
                # If bc_only is True, skip non-barcode components
                return self.get_next(next_comp, bc_only=True)
            return next_comp
        return None

    def get_previous(
        self, component: ReadComponent, bc_only: bool = False
    ) -> ReadComponent | None:
        """
        Get the previous component before the given component.

        Args:
            component: The component to find the previous one for.
            bc_only: If True, only consider barcode components when looking for the previous component.

        Returns:
            The previous ReadComponent or None if it's the first one or not found.
        """
        idx = self._index_map.get(component.name)
        if idx is not None and idx - 1 >= 0:
            prev_comp = self.components[idx - 1]
            if bc_only and prev_comp.type is not ReadComponentType.BARCODE:
                # If bc_only is True, skip non-barcode components
                return self.get_previous(prev_comp, bc_only=True)
            return prev_comp
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
