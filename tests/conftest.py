"""
Shared pytest fixtures for the barcode matcher test classes.

The matcher implementations share a constructor signature, so the HyDrop chemistry,
whitelist and per-component matcher fixtures are defined once here. A test class opts in
by overriding the `matcher_class` fixture with the matcher it exercises, and supplies any
constructor arguments unique to that matcher by overriding the `matcher_kwargs` fixture.
"""

from functools import cached_property
from typing import Any, Callable

import pytest

from carmack.barcode.matchers.matcher_base import MatcherBase
from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure

# Spacer sequences for the wide-spaced test chemistry below.
WIDE_SPACER_UPSTREAM = "AGGGTACTCG"
WIDE_SPACER_DOWNSTREAM = "GCAGTAGCTG"


class WideSpacedChemistry(ChemistryBase):
    """
    A chemistry whose middle barcode has an unusually wide search window.

    A matcher bounds its search to the region the declared layout could have moved a component
    into: the component's own extent widened by the errors everything ahead of it may carry,
    plus the budget it is matched with. Every shipped chemistry keeps that region far too
    tight for two complete spacer-barcode-spacer arrangements to fit inside it, which is the
    point of the bound and is exactly what the misassignment it prevents relied on. It does
    leave the spacer tie-break rungs -- which by definition need two candidates inside one
    component's window -- with nowhere to be exercised.

    This double buys that room by declaring a spacer budget far larger than any real
    chemistry would: a wide window is a claim that the layout is very uncertain about where a
    component sits, and a component's error budget is where that claim is made. Widening the
    read layout instead does nothing, because drift accumulates over the budgets of the
    components ahead of a barcode rather than over the bases between them -- the same room
    could be bought by declaring many more components upstream, but one large budget says what
    the fixture is for far more plainly than a wall of filler would.

    The 10bp spacers immediately flanking BC2 are kept because the tie-break rungs read them.
    The 60bp gap behind BC2 is kept only so the layout still has three barcodes to space out;
    it sits 3' of BC2 and so contributes nothing to BC2's own window, and no test currently
    matches against this double's BC1.

    It is deliberately not registered with ``ChemistryFactory``: it is a test double describing
    a read layout, not a chemistry anything is run with.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier for this test chemistry."""
        return "test_wide_spaced"

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Define BC3, BC2 flanked by two spacers, a long gap, then BC1."""
        return ReadStructure(
            [
                ReadComponent(name="BC3", type=ReadComponentType.BARCODE, length=10),
                ReadComponent(
                    name="SPACER_1",
                    type=ReadComponentType.OTHER,
                    length=len(WIDE_SPACER_UPSTREAM),
                    sequence=WIDE_SPACER_UPSTREAM,
                ),
                ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=10),
                ReadComponent(
                    name="SPACER_2",
                    type=ReadComponentType.OTHER,
                    length=len(WIDE_SPACER_DOWNSTREAM),
                    sequence=WIDE_SPACER_DOWNSTREAM,
                ),
                ReadComponent(name="GAP", type=ReadComponentType.OTHER, length=60),
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
            ]
        )

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return HyDrop's barcode budget over a deliberately enormous spacer budget.

        The barcode budget stays at HyDrop's two so the tie-break tests keep the tolerances
        they were written against. The spacer budget is what widens BC2's window, and it is
        set well past anything a real chemistry would declare so that two spacer-flanked
        candidates fit inside that window with room to spare.
        """
        return MatchErrors(barcode=2, spacer=30)

    @cached_property
    def whitelists(self) -> dict[str, tuple[str, ...]]:
        """Return placeholder whitelists.

        Construction rejects a barcode component with no whitelist, but the tie-break tests
        hand their matcher an explicit whitelist, so these only have to exist and to be spread
        widely enough to satisfy the distance bound.
        """
        return {
            "BC3": ("ACGTACGTAC", "TTTTTGGGGG"),
            "BC2": ("TGCATGCATG", "AAAAACCCCC"),
            "BC1": ("GGGGGGGGGG", "CCCCCCCCCC"),
        }


@pytest.fixture
def wide_spaced_chemistry() -> WideSpacedChemistry:
    """
    Provide the wide-spaced test chemistry.

    Returns:
        A freshly constructed WideSpacedChemistry.
    """
    return WideSpacedChemistry()


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
def matcher_kwargs() -> dict[str, Any]:
    """
    Provide the extra constructor arguments for the matcher under test.

    Matchers differ in the arguments they require beyond whitelist, component and chemistry,
    so a test class whose matcher takes any extras overrides this fixture with them.

    Returns:
        Mapping of keyword argument name to value; empty when the matcher takes no extras.
    """
    return {}


@pytest.fixture
def hydrop_matcher(
    matcher_class: type[MatcherBase],
    matcher_kwargs: dict[str, Any],
    hydrop_chemistry: ChemistryHydrop,
    hydrop_whitelists: dict[str, tuple[str, ...]],
) -> Callable[..., MatcherBase]:
    """
    Provide a factory that builds a matcher for a named HyDrop barcode component.

    Args:
        matcher_class: The matcher class to instantiate, from the consuming test class.
        matcher_kwargs: Extra constructor arguments required by that matcher class.
        hydrop_chemistry: The HyDrop chemistry to configure the matcher with.
        hydrop_whitelists: Mapping of barcode component name to whitelist sequences.

    Returns:
        Callable taking a HyDrop barcode component name plus any matcher-specific keyword
        arguments, and returning the configured matcher. Per-call keyword arguments take
        precedence over `matcher_kwargs`.
    """

    def build(bc_name: str, **kwargs: Any) -> MatcherBase:
        comp = hydrop_chemistry.read_structure.get_component_by_name(bc_name)
        return matcher_class(
            whitelist=hydrop_whitelists[bc_name],
            component=comp,
            chemistry=hydrop_chemistry,
            **{**matcher_kwargs, **kwargs},
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
    matcher_class: type[MatcherBase],
    matcher_kwargs: dict[str, Any],
    small_whitelist: tuple[str, ...],
) -> MatcherBase:
    """
    Provide a matcher with a small whitelist and simple chemistry (HyDrop BC3).

    Args:
        matcher_class: The matcher class to instantiate, from the consuming test class.
        matcher_kwargs: Extra constructor arguments required by that matcher class.
        small_whitelist: The small, deterministic whitelist to match against.

    Returns:
        The matcher configured for BC3 over the small whitelist.
    """
    chemistry = ChemistryHydrop()
    comp = chemistry.read_structure.get_component_by_name("BC3")
    return matcher_class(
        whitelist=small_whitelist, component=comp, chemistry=chemistry, **matcher_kwargs
    )
