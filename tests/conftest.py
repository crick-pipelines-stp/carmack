"""
Shared pytest fixtures for the barcode matcher test classes.

The matcher implementations share a constructor signature, so the HyDrop chemistry,
whitelist and per-component matcher fixtures are defined once here. A test class opts in
by overriding the `matcher_class` fixture with the matcher it exercises.
"""

from typing import Any, Callable

import pytest

from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop


@pytest.fixture
def hydrop_chemistry() -> ChemistryHydrop:
    """
    Provide a HyDrop chemistry instance.

    Returns:
        A freshly constructed ChemistryHydrop.
    """
    return ChemistryHydrop()


@pytest.fixture
def hydrop_whitelists(hydrop_chemistry: ChemistryHydrop) -> dict[str, tuple[str, ...]]:
    """
    Provide stripped HyDrop whitelists (10bp variable regions).

    Args:
        hydrop_chemistry: The HyDrop chemistry supplying the whitelists.

    Returns:
        Mapping of barcode component name to its whitelist sequences.
    """
    return hydrop_chemistry.barcode_whitelists


@pytest.fixture
def matcher_class() -> type[MatcherBase]:
    """
    Provide the matcher class used to build the shared HyDrop matcher fixtures.

    This is a guard implementation: a test class that consumes any of the shared matcher
    fixtures is responsible for overriding it with a concrete matcher.

    Returns:
        The MatcherBase subclass under test.
    """
    pytest.fail(
        "Test classes using the shared matcher fixtures must override the 'matcher_class' fixture"
    )


@pytest.fixture
def hydrop_matcher(
    matcher_class: type[MatcherBase],
    hydrop_chemistry: ChemistryHydrop,
    hydrop_whitelists: dict[str, tuple[str, ...]],
) -> Callable[..., MatcherBase]:
    """
    Provide a factory that builds a matcher for a named HyDrop barcode component.

    Args:
        matcher_class: The matcher class to instantiate, from the consuming test class.
        hydrop_chemistry: The HyDrop chemistry to configure the matcher with.
        hydrop_whitelists: Mapping of barcode component name to whitelist sequences.

    Returns:
        Callable taking a HyDrop barcode component name plus any matcher-specific keyword
        arguments, and returning the configured matcher.
    """

    def build(bc_name: str, **kwargs: Any) -> MatcherBase:
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        return matcher_class(
            whitelist=hydrop_whitelists[bc_name],
            barcode_component=comp,
            chemistry=hydrop_chemistry,
            **kwargs,
        )

    return build


@pytest.fixture
def bc3_matcher(hydrop_matcher: Callable[..., MatcherBase]) -> MatcherBase:
    """
    Provide a matcher for HyDrop BC3 (start=0, length=10).

    Args:
        hydrop_matcher: Factory building a matcher for a named barcode component.

    Returns:
        The matcher configured for BC3.
    """
    return hydrop_matcher("BC3")


@pytest.fixture
def bc2_matcher(hydrop_matcher: Callable[..., MatcherBase]) -> MatcherBase:
    """
    Provide a matcher for HyDrop BC2 (start=20, length=10).

    Args:
        hydrop_matcher: Factory building a matcher for a named barcode component.

    Returns:
        The matcher configured for BC2.
    """
    return hydrop_matcher("BC2")


@pytest.fixture
def bc1_matcher(hydrop_matcher: Callable[..., MatcherBase]) -> MatcherBase:
    """
    Provide a matcher for HyDrop BC1 (start=40, length=10).

    Args:
        hydrop_matcher: Factory building a matcher for a named barcode component.

    Returns:
        The matcher configured for BC1.
    """
    return hydrop_matcher("BC1")


@pytest.fixture
def small_whitelist() -> tuple[str, ...]:
    """
    Provide a small, deterministic whitelist for isolated tests.

    Returns:
        Four 10bp barcode sequences.
    """
    return ("ACGTACGTAC", "TGCATGCATG", "GGGGGGGGGG", "CCCCCCCCCC")


@pytest.fixture
def small_matcher(
    matcher_class: type[MatcherBase], small_whitelist: tuple[str, ...]
) -> MatcherBase:
    """
    Provide a matcher with a small whitelist and simple chemistry (HyDrop BC3).

    Args:
        matcher_class: The matcher class to instantiate, from the consuming test class.
        small_whitelist: The small, deterministic whitelist to match against.

    Returns:
        The matcher configured for BC3 over the small whitelist.
    """
    chemistry = ChemistryHydrop()
    comp = chemistry.read_structure.get_component_by_name("BC3")
    return matcher_class(whitelist=small_whitelist, barcode_component=comp, chemistry=chemistry)
