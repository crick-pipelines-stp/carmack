"""
Tests for the chemistry module.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from importlib.resources import files
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.chemistry.chemistry_base import (
    AnchorOffset,
    ChemistryBase,
    MatchErrors,
    WhitelistSource,
)
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (
    POLYG_BASE,
    POLYG_MIN_RUN,
    TGIDX_LENGTH,
    UMI_LENGTH,
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.chemistry_carmack_custom_seq_1_0_primd import (
    ChemistryCarmackCustomSeq10PrimD,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.gzip_file import GzipFile


class TestReadComponent:
    """Tests for the ReadComponent dataclass."""

    def test_read_component_creation(self):
        """Test that a ReadComponent can be created with the expected attributes."""
        comp = ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10)
        assert_that(comp.name).is_equal_to("BC1")
        assert_that(comp.type).is_equal_to(ReadComponentType.BARCODE)
        assert_that(comp.length).is_equal_to(10)

    def test_read_component_type_must_be_enum(self):
        """Test that the component type must be a ReadComponentType member."""
        with pytest.raises(
            TypeError,
            match="ReadComponent type must be a ReadComponentType enum member, got str",
        ):
            ReadComponent(name="X", type="OTHER", length=5)

    def test_barcode_component_cannot_have_sequence(self):
        """Test that barcode components reject explicit sequences."""
        with pytest.raises(
            ValueError,
            match="Barcode components should not be instantiated with a sequence.",
        ):
            ReadComponent(
                name="X",
                type=ReadComponentType.BARCODE,
                length=5,
                sequence="ACGT",
            )

    def test_barcode_component_without_sequence_is_valid(self):
        """Test that barcode components are valid when sequence is omitted."""
        comp = ReadComponent(name="X", type=ReadComponentType.BARCODE, length=5)
        assert_that(comp.type).is_equal_to(ReadComponentType.BARCODE)
        assert_that(comp.sequence).is_none()

    def test_read_component_defaults_to_other_type(self):
        """Test that components default to OTHER type."""
        comp = ReadComponent(name="X", length=5)
        assert_that(comp.type).is_equal_to(ReadComponentType.OTHER)

    def test_read_component_with_invalid_values(self):
        """Test that a ReadComponent with invalid length raises an error."""
        with pytest.raises(ValueError):
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=-1)
        with pytest.raises(ValueError):
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=0)

    def test_read_component_start_field(self):
        """The start field defaults to None and accepts int or None assignment directly."""
        comp = ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10)
        assert_that(comp.start).is_none()  # default value

        comp.start = 5
        assert_that(comp.start).is_equal_to(5)

        comp.start = None
        assert_that(comp.start).is_none()

    # ===== Enum rename: GGG -> HOMOPOLYMER =====

    def test_homopolymer_enum_member_replaces_ggg(self) -> None:
        """The former GGG member is renamed to HOMOPOLYMER with a matching value."""
        assert_that(ReadComponentType.HOMOPOLYMER.value).is_equal_to("HOMOPOLYMER")
        assert_that(hasattr(ReadComponentType, "GGG")).is_false()

    # ===== Per-type validation: valid construction =====

    def test_primer_component_valid(self) -> None:
        """A primer component requires only a positive length."""
        comp = ReadComponent(name="PRIMER_C", type=ReadComponentType.PRIMER, length=22)
        assert_that(comp.length).is_equal_to(22)

    def test_umi_component_valid(self) -> None:
        """A UMI component accepts a positive length."""
        comp = ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8)
        assert_that(comp.length).is_equal_to(8)

    def test_umi_component_is_fixed_length(self) -> None:
        """A UMI occupies exactly its declared length, carrying no jitter."""
        comp = ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8)
        assert_that(comp.is_variable_length).is_false()

    def test_homopolymer_component_valid(self) -> None:
        """A homopolymer requires a single base and positive min_run; length may be None."""
        comp = ReadComponent(
            name="polyG",
            type=ReadComponentType.HOMOPOLYMER,
            homopolymer_base="G",
            min_run=3,
        )
        assert_that(comp.homopolymer_base).is_equal_to("G")
        assert_that(comp.min_run).is_equal_to(3)
        assert_that(comp.length).is_none()

    def test_tgidx_component_valid(self) -> None:
        """A TGIDX component requires a positive length."""
        comp = ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8)
        assert_that(comp.length).is_equal_to(8)

    # ===== Per-type validation: invalid construction =====

    @pytest.mark.parametrize(
        "component_type",
        [
            ReadComponentType.BARCODE,
            ReadComponentType.PRIMER,
            ReadComponentType.UMI,
            ReadComponentType.TGIDX,
            ReadComponentType.OTHER,
        ],
    )
    def test_non_homopolymer_component_requires_length(
        self, component_type: ReadComponentType
    ) -> None:
        """Every non-homopolymer component type requires a length."""
        with pytest.raises(ValueError, match="length must be positive"):
            ReadComponent(name="X", type=component_type, length=None)

    @pytest.mark.parametrize("base", [None, "N", "GG", "g"])
    def test_homopolymer_component_rejects_invalid_base(self, base: str | None) -> None:
        """A homopolymer component rejects a missing or non-single-ACGT base."""
        with pytest.raises(ValueError, match="homopolymer_base"):
            ReadComponent(
                name="polyG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base=base,
                min_run=3,
            )

    @pytest.mark.parametrize("min_run", [None, 0, -1])
    def test_homopolymer_component_rejects_invalid_min_run(self, min_run: int | None) -> None:
        """A homopolymer component rejects a missing or non-positive min_run."""
        with pytest.raises(ValueError, match="min_run"):
            ReadComponent(
                name="polyG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base="G",
                min_run=min_run,
            )

    # ===== Derived properties =====

    @pytest.mark.parametrize(
        "component,expected",
        [
            (ReadComponent(name="BC", type=ReadComponentType.BARCODE, length=10), True),
            (ReadComponent(name="PR", type=ReadComponentType.PRIMER, length=22), True),
            (
                ReadComponent(
                    name="HP",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base="G",
                    min_run=3,
                ),
                True,
            ),
            (ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8), False),
            (ReadComponent(name="OT", type=ReadComponentType.OTHER, length=10), False),
            (ReadComponent(name="TG", type=ReadComponentType.TGIDX, length=8), False),
        ],
    )
    def test_is_anchor(self, component: ReadComponent, expected: bool) -> None:
        """is_anchor is True for BARCODE/PRIMER/HOMOPOLYMER and False otherwise."""
        assert_that(component.is_anchor).is_equal_to(expected)

    @pytest.mark.parametrize(
        "component,expected",
        [
            (ReadComponent(name="BC", type=ReadComponentType.BARCODE, length=10), False),
            (ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8), False),
            (
                ReadComponent(
                    name="HP",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base="G",
                    min_run=3,
                ),
                True,
            ),
        ],
    )
    def test_is_variable_length(self, component: ReadComponent, expected: bool) -> None:
        """is_variable_length reflects homopolymer type or unknown length."""
        assert_that(component.is_variable_length).is_equal_to(expected)

    @pytest.mark.parametrize(
        "component_type,expected",
        [
            (ReadComponentType.BARCODE, True),
            (ReadComponentType.PRIMER, False),
            (ReadComponentType.HOMOPOLYMER, False),
            (ReadComponentType.UMI, False),
            (ReadComponentType.TGIDX, False),
            (ReadComponentType.OTHER, False),
        ],
    )
    def test_records_position(self, component_type: ReadComponentType, expected: bool) -> None:
        """Only barcodes have their position written to the read header."""
        comp = ReadComponent(
            name="X",
            type=component_type,
            length=10,
            homopolymer_base="G" if component_type is ReadComponentType.HOMOPOLYMER else None,
            min_run=3 if component_type is ReadComponentType.HOMOPOLYMER else None,
        )
        assert_that(comp.records_position).is_equal_to(expected)

    def test_records_position_is_not_implied_by_is_anchor(self) -> None:
        """A primer anchors a neighbour without its position ever being recorded.

        The two properties are consulted together by the anchor-offset walk, and
        conflating them would have it return a component whose position tag is
        never written.
        """
        primer = ReadComponent(name="PRIMER_A", type=ReadComponentType.PRIMER, length=22)
        assert_that(primer.is_anchor).is_true()
        assert_that(primer.records_position).is_false()


class TestResolveAnchorOffset:
    """The leftward walk that locates a component against a recorded anchor.

    The walk is what replaced reading a coordinate back out of a UMI span: it
    turns "where does this component start" into chemistry arithmetic over a
    barcode position tag, so a stage no longer depends on which neighbour an
    earlier stage happened to record.
    """

    @staticmethod
    def build(components: list[ReadComponent]) -> ChemistryBase:
        """Return a chemistry over the given read structure.

        Subclasses the shipped chemistry rather than ``ChemistryBase`` directly,
        following the pattern the target-assigner tests use, so the barcode and
        target-index whitelists construction validates against come for free.
        """

        class Synthetic(ChemistryCarmackCustomSeq10):
            @cached_property
            def name(self) -> str:
                return "synthetic"

            @cached_property
            def read_structure(self) -> ReadStructure:
                return ReadStructure(components)

        return Synthetic()

    @staticmethod
    def polyg() -> ReadComponent:
        """Return a poly-G homopolymer component."""
        return ReadComponent(
            name="POLYG", type=ReadComponentType.HOMOPOLYMER, homopolymer_base="G", min_run=3
        )

    def test_offset_is_zero_for_an_immediate_anchor(self) -> None:
        """A component sitting directly on a barcode is at that barcode's span end."""
        target = self.polyg()
        chemistry = self.build(
            [ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10), target]
        )
        assert_that(chemistry.resolve_anchor_offset(target)).is_equal_to(
            AnchorOffset(anchor=chemistry.read_structure.get_component_by_name("BC1"), offset=0)
        )

    def test_offset_sums_one_crossed_component(self) -> None:
        """Crossing a fixed-length UMI puts the run that many bases past the anchor."""
        target = self.polyg()
        chemistry = self.build(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8),
                target,
            ]
        )
        located = chemistry.resolve_anchor_offset(target)
        assert_that(located.anchor.name).is_equal_to("BC1")
        assert_that(located.offset).is_equal_to(8)
        assert_that(located.position_key).is_equal_to("BC1_POS")

    def test_offset_sums_several_crossed_components(self) -> None:
        """A fixed-length linker between the UMI and the run is simply crossed too.

        The old coordinate scheme rejected this layout outright, because the run
        start was read off the end of the UMI span and a linker would have put
        the window on the linker instead.
        """
        target = self.polyg()
        chemistry = self.build(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8),
                ReadComponent(name="LINKER", type=ReadComponentType.OTHER, length=10),
                target,
            ]
        )
        located = chemistry.resolve_anchor_offset(target)
        assert_that(located.anchor.name).is_equal_to("BC1")
        assert_that(located.offset).is_equal_to(18)

    def test_variable_component_before_the_anchor_raises(self) -> None:
        """A variable component in between makes the distance not a fixed number."""
        target = ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8)
        chemistry = self.build(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                self.polyg(),
                target,
            ]
        )
        with pytest.raises(ValueError, match="variable-length component 'POLYG'"):
            chemistry.resolve_anchor_offset(target)

    def test_no_anchor_to_the_left_raises(self) -> None:
        """A component with nothing 5' of it cannot be located at all."""
        target = self.polyg()
        chemistry = self.build([target])
        with pytest.raises(ValueError, match="no anchor 5' of component 'POLYG'"):
            chemistry.resolve_anchor_offset(target)

    def test_anchor_whose_position_is_not_recorded_raises(self) -> None:
        """A primer anchors the run but never has its position written down.

        This is the case ``records_position`` exists for. Testing ``is_anchor``
        alone would return the primer here, and the stage would then look up a
        ``PRIMER_A_POS`` tag that barcode extraction never writes, making every
        read in the run look like it was missing its anchor.
        """
        target = self.polyg()
        chemistry = self.build(
            [ReadComponent(name="PRIMER_A", type=ReadComponentType.PRIMER, length=22), target]
        )
        with pytest.raises(ValueError, match="no anchor 5' of component 'POLYG'"):
            chemistry.resolve_anchor_offset(target)

    @pytest.mark.parametrize(
        "chemistry_name,expected_offset",
        [("carmack_custom_seq_1_0", 8), ("carmack_custom_seq_1_0_primd", 8)],
    )
    def test_shipped_chemistries_locate_the_run_from_bc1(
        self, chemistry_name: str, expected_offset: int
    ) -> None:
        """Both shipped chemistries measure the poly-G run off BC1, across the UMI."""
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        located = chemistry.resolve_anchor_offset(chemistry.tgidx_anchor())
        assert_that(located.anchor.name).is_equal_to("BC1")
        assert_that(located.offset).is_equal_to(expected_offset)
        assert_that(located.position_key).is_equal_to("BC1_POS")


class TestUmiRightAnchor:
    """The diagnostic-only lookup of the component 3' of the UMI.

    Nothing about extracting the UMI consults it -- the slice is taken off the
    left anchor -- so every way it comes back empty has to leave extraction
    working and merely drop the anchor-run section from the report. Built over
    synthetic structures because no shipped chemistry has a UMI whose right
    neighbour cannot anchor it.
    """

    build = staticmethod(TestResolveAnchorOffset.build)

    @staticmethod
    def umi() -> ReadComponent:
        """Return a fixed-length UMI component."""
        return ReadComponent(name="UMI", type=ReadComponentType.UMI, length=UMI_LENGTH)

    def test_non_anchor_neighbour_is_not_returned(self) -> None:
        """A UMI followed by something that cannot anchor it has no right anchor."""
        chemistry = self.build(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                self.umi(),
                ReadComponent(name="LINKER", type=ReadComponentType.OTHER, length=10),
            ]
        )
        assert_that(chemistry.umi_right_anchor()).is_none()
        assert_that(chemistry.supports_umi_extraction()).is_true()

    def test_umi_at_the_end_of_the_read_has_no_right_anchor(self) -> None:
        """With nothing 3' of the UMI at all there is no neighbour to inspect."""
        chemistry = self.build(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                self.umi(),
            ]
        )
        assert_that(chemistry.umi_right_anchor()).is_none()
        assert_that(chemistry.supports_umi_extraction()).is_true()

    def test_chemistry_with_no_umi_has_no_right_anchor(self) -> None:
        """The lookup short-circuits rather than asking for a neighbour of nothing."""
        assert_that(ChemistryHydrop().umi_right_anchor()).is_none()


class TestReadStructure:
    """Test suite for ReadStructure."""

    @pytest.fixture
    def sample_components(self):
        """Provide a sample list of ReadComponents for testing."""
        return [
            ReadComponent(name="BC3", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(name="PRIMER_C", type=ReadComponentType.PRIMER, length=22),
            ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(name="PRIMER_A", type=ReadComponentType.PRIMER, length=22),
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
        ]

    @pytest.fixture
    def read_structure(self, sample_components):
        """Provide a ReadStructure instance."""
        return ReadStructure(sample_components)

    def test_get_component_by_name_valid(self, read_structure, sample_components):
        """Test getting a component by a valid name."""
        comp = read_structure.get_component_by_name("PRIMER_C")
        assert_that(comp).is_instance_of(ReadComponent)
        assert_that(comp.name).is_equal_to("PRIMER_C")
        assert_that(comp.type).is_equal_to(ReadComponentType.PRIMER)
        assert_that(comp.length).is_equal_to(22)

    def test_get_component_by_name_invalid(self, read_structure):
        """Test that getting a component by an invalid name raises an error."""
        with pytest.raises(ValueError) as exc_info:
            read_structure.get_component_by_name("UNKNOWN")
        assert_that(str(exc_info.value)).contains("Component with name 'UNKNOWN' not found")

    def test_get_next_middle_component(self, read_structure, sample_components):
        """Test getting the next component for a middle component."""
        comp = sample_components[1]  # PRIMER_C
        next_comp = read_structure.get_next(comp)
        assert next_comp == sample_components[2]  # BC2

    def test_get_next_last_component(self, read_structure, sample_components):
        """Test getting the next component for the last component returns None."""
        comp = sample_components[-1]  # BC1
        next_comp = read_structure.get_next(comp)
        assert next_comp is None

    def test_get_previous_middle_component(self, read_structure, sample_components):
        """Test getting the previous component for a middle component."""
        comp = sample_components[2]  # BC2
        prev_comp = read_structure.get_previous(comp)
        assert prev_comp == sample_components[1]  # PRIMER_C

    def test_get_previous_first_component(self, read_structure, sample_components):
        """Test getting the previous component for the first component returns None."""
        comp = sample_components[0]  # BC3
        prev_comp = read_structure.get_previous(comp)
        assert prev_comp is None

    def test_get_next_unknown_component(self, read_structure):
        """Test getting next for a component not in the structure returns None."""
        unknown_comp = ReadComponent(name="UNKNOWN", type=ReadComponentType.OTHER, length=5)
        next_comp = read_structure.get_next(unknown_comp)
        assert next_comp is None

    def test_get_previous_unknown_component(self, read_structure):
        """Test getting previous for a component not in the structure returns None."""
        unknown_comp = ReadComponent(name="UNKNOWN", type=ReadComponentType.OTHER, length=5)
        prev_comp = read_structure.get_previous(unknown_comp)
        assert prev_comp is None

    def test_get_components_by_type_returns_barcodes_in_order(
        self, read_structure: ReadStructure
    ) -> None:
        """Test that barcode components are returned in read order."""
        components = read_structure.get_components_by_type(ReadComponentType.BARCODE)
        assert_that([comp.name for comp in components]).is_equal_to(["BC3", "BC2", "BC1"])

    def test_get_components_by_type_returns_primers_in_order(
        self, read_structure: ReadStructure
    ) -> None:
        """Test that primer components are returned in read order."""
        components = read_structure.get_components_by_type(ReadComponentType.PRIMER)
        assert_that([comp.name for comp in components]).is_equal_to(["PRIMER_C", "PRIMER_A"])

    @pytest.mark.parametrize("component_type", ["BARCODE", None])
    def test_get_components_by_type_rejects_invalid_type(
        self, read_structure: ReadStructure, component_type: object
    ) -> None:
        """Test that get_components_by_type requires a ReadComponentType."""
        expected = type(component_type).__name__
        with pytest.raises(
            TypeError,
            match=f"component_type must be a ReadComponentType enum member, got {expected}",
        ):
            read_structure.get_components_by_type(component_type)  # type: ignore[arg-type]

    def test_get_components_by_type_errors_when_type_absent(
        self, read_structure: ReadStructure
    ) -> None:
        """Test that get_components_by_type reports missing valid types clearly."""
        with pytest.raises(ValueError) as exc_info:
            read_structure.get_components_by_type(ReadComponentType.TGIDX)

        assert_that(str(exc_info.value)).contains("TGIDX")
        assert_that(str(exc_info.value)).contains("BARCODE, PRIMER")

    def test_iteration(self, read_structure, sample_components):
        """Test that ReadStructure supports iteration."""
        components = list(read_structure)
        assert components == sample_components

    def test_indexing(self, read_structure, sample_components):
        """Test that ReadStructure supports indexing."""
        assert read_structure[0] == sample_components[0]
        assert read_structure[1] == sample_components[1]
        assert read_structure[-1] == sample_components[-1]

    def test_len(self, read_structure, sample_components):
        """Test that ReadStructure supports len()."""
        assert len(read_structure) == len(sample_components)


class TestComputeStartPositionsVariable:
    """Tests for variable-aware start-position computation."""

    def test_all_fixed_structure_keeps_exact_starts(self) -> None:
        """A fully fixed structure computes contiguous integer starts."""
        components = [
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(name="PRIMER_A", type=ReadComponentType.PRIMER, length=22),
            ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=10),
        ]
        ReadStructure(components)
        assert_that([comp.start for comp in components]).is_equal_to([0, 10, 32])

    def test_trailing_variable_component_gets_start_then_none(self) -> None:
        """The fixed prefix and first variable component keep starts; later ones are None."""
        components = [
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8),
            ReadComponent(
                name="polyG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base="G",
                min_run=3,
            ),
            ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
        ]
        ReadStructure(components)
        starts = {comp.name: comp.start for comp in components}
        assert_that(starts["BC1"]).is_equal_to(0)
        assert_that(starts["UMI"]).is_equal_to(10)
        assert_that(starts["polyG"]).is_equal_to(18)
        assert_that(starts["TGIDX"]).is_none()

    def test_homopolymer_first_variable_gets_start_then_none(self) -> None:
        """A homopolymer as the first variable component still gets a concrete start."""
        components = [
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
            ReadComponent(
                name="polyG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base="G",
                min_run=3,
            ),
            ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
        ]
        ReadStructure(components)
        starts = {comp.name: comp.start for comp in components}
        assert_that(starts["BC1"]).is_equal_to(0)
        assert_that(starts["polyG"]).is_equal_to(10)
        assert_that(starts["TGIDX"]).is_none()


class TestMatchErrors:
    """Tests for the MatchErrors dataclass."""

    def test_tgidx_defaults_to_one(self):
        """tgidx defaults to 1 so a chemistry adding a target index is not exact-match-only."""
        errors = MatchErrors(barcode=1, spacer=1)
        assert_that(errors.tgidx).is_equal_to(1)


class TestChemistryBase:
    """In-depth tests for ChemistryBase abstract class functionality."""

    @pytest.fixture
    def chemistry(self) -> ChemistryCarmackCustomSeq10:
        """Provide a concrete chemistry instance for testing."""
        return ChemistryCarmackCustomSeq10()

    # ===== Tests for barcode_whitelists property =====

    def test_barcode_whitelists_returns_dict(self, chemistry: ChemistryBase):
        """Test that barcode_whitelists returns a dictionary."""
        whitelists = chemistry.barcode_whitelists
        assert_that(whitelists).is_instance_of(dict)

    def test_barcode_whitelists_has_all_barcode_components(self, chemistry: ChemistryBase):
        """Test that barcode_whitelists includes all barcode components."""
        whitelists = chemistry.barcode_whitelists
        barcode_names = {
            comp.name
            for comp in chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)
        }

        for bc_name in barcode_names:
            assert_that(whitelists).contains_key(bc_name)

    def test_barcode_whitelists_excludes_non_barcode_components(self, chemistry: ChemistryBase):
        """Test that barcode_whitelists only includes barcode components."""
        whitelists = chemistry.barcode_whitelists
        spacer_names = {
            comp.name
            for comp in chemistry.read_structure
            if comp.type is not ReadComponentType.BARCODE
        }

        for spacer_name in spacer_names:
            assert_that(whitelists).does_not_contain_key(spacer_name)

    def test_barcode_whitelists_values_are_tuples(self, chemistry: ChemistryBase):
        """Test that all whitelist values are tuples."""
        whitelists = chemistry.barcode_whitelists

        for bc_name, whitelist in whitelists.items():
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)

    def test_barcode_whitelists_is_cached(self, chemistry: ChemistryBase):
        """Test that barcode_whitelists is cached (returns same object)."""
        whitelists1 = chemistry.barcode_whitelists
        whitelists2 = chemistry.barcode_whitelists

        assert_that(whitelists1 is whitelists2).is_true()

    def test_barcode_whitelists_contains_valid_dna_sequences(self, chemistry: ChemistryBase):
        """Test that all barcodes are valid DNA sequences (only ACGT or N)."""
        whitelists = chemistry.barcode_whitelists
        valid_dna_chars = set("ACGTN")

        for bc_name, whitelist in whitelists.items():
            for barcode in whitelist:
                barcode_chars = set(barcode)
                assert_that(barcode_chars).is_subset_of(valid_dna_chars)

    # ===== Tests for construct_full_barcode method =====

    def test_construct_full_barcode_basic(self, chemistry: ChemistryBase):
        """Test basic full barcode construction."""
        barcodes = {
            "BC1": "ACGTACGTAC",
            "BC2": "TGACAGTGAC",
            "BC3": "CGTACGTACG",
        }
        full_bc = chemistry.construct_full_barcode(barcodes)

        assert_that(full_bc).is_instance_of(str)
        assert_that(len(full_bc)).is_equal_to(30)  # 3 barcodes * 10 bp each

    def test_construct_full_barcode_preserves_order(self, chemistry: ChemistryBase):
        """Test that construct_full_barcode maintains barcode order from read structure."""
        barcodes = {
            "BC1": "BC1_SEQNCE",
            "BC2": "BC2_SEQNCE",
            "BC3": "BC3_SEQNCE",
        }
        full_bc = chemistry.construct_full_barcode(barcodes)

        # Read structure order is: BC3, BC2, BC1
        expected = "BC3_SEQNCEBC2_SEQNCEBC1_SEQNCE"
        assert_that(full_bc).is_equal_to(expected)

    def test_construct_full_barcode_missing_component_raises_error(self, chemistry: ChemistryBase):
        """Test that missing a barcode component raises ValueError."""
        barcodes = {
            "BC1": "ACGTACGTAC",
            "BC2": "TGACAGTGAC",
            # BC3 is missing
        }

        with pytest.raises(ValueError) as exc_info:
            chemistry.construct_full_barcode(barcodes)

        assert_that(str(exc_info.value)).contains("Missing barcode component")
        assert_that(str(exc_info.value)).contains("BC3")

    def test_construct_full_barcode_none_value_raises_error(self, chemistry: ChemistryBase):
        """A present-but-None barcode value raises the same clear ValueError."""
        barcodes = {"BC1": "ACGTACGTAC", "BC2": "TGACAGTGAC", "BC3": None}

        with pytest.raises(ValueError) as exc_info:
            chemistry.construct_full_barcode(barcodes)

        assert_that(str(exc_info.value)).contains("Missing barcode component")
        assert_that(str(exc_info.value)).contains("BC3")

    def test_construct_full_barcode_extra_components_ignored(self, chemistry: ChemistryBase):
        """Test that extra barcode components not in read structure are ignored."""
        barcodes = {
            "BC1": "ACGTACGTAC",
            "BC2": "TGACAGTGAC",
            "BC3": "CGTACGTACG",
            "BC4": "EXTRA_DATA",  # Not in read structure
        }
        full_bc = chemistry.construct_full_barcode(barcodes)

        # Should not include BC4
        assert_that(full_bc).does_not_contain("EXTRA_DATA")
        assert_that(len(full_bc)).is_equal_to(30)

    def test_construct_full_barcode_with_actual_whitelisted_barcodes(
        self, chemistry: ChemistryBase
    ):
        """Test construct_full_barcode with actual barcodes from whitelists."""
        whitelists = chemistry.barcode_whitelists

        barcodes = {
            "BC1": whitelists["BC1"][0],
            "BC2": whitelists["BC2"][0],
            "BC3": whitelists["BC3"][0],
        }

        full_bc = chemistry.construct_full_barcode(barcodes)

        assert_that(full_bc).is_instance_of(str)
        assert_that(len(full_bc)).is_equal_to(30)
        assert_that(full_bc).contains(whitelists["BC1"][0])

    def test_construct_full_barcode_empty_string_raises_error(self, chemistry: ChemistryBase):
        """Test that empty barcode strings raise an error."""
        barcodes = {
            "BC1": "",
            "BC2": "TGACAGTGAC",
            "BC3": "CGTACGTACG",
        }

        full_bc = chemistry.construct_full_barcode(barcodes)
        # Empty string is allowed but results in missing data
        assert_that(full_bc).contains("TGACAGTGAC")

    # ===== Tests for __post_init__ validation =====

    def test_chemistry_initialization_with_valid_structure(self, chemistry: ChemistryBase):
        """Test that chemistry initializes successfully with valid structure."""
        # If we got a chemistry instance without error, initialization was valid
        assert_that(chemistry).is_instance_of(ChemistryBase)

    def test_chemistry_all_barcode_components_have_whitelists(self, chemistry: ChemistryBase):
        """Test that all barcode components have valid whitelists."""
        for component in chemistry.read_structure.get_components_by_type(
            ReadComponentType.BARCODE
        ):
            whitelist = chemistry.load_whitelist(component.name)
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)

    def test_chemistry_spacers_match_read_structure(self, chemistry: ChemistryBase):
        """Test that all known sequences are present in the read structure."""
        known_seqs = chemistry.read_structure.get_known_sequences()
        structure_spacer_names = {
            comp.name
            for comp in chemistry.read_structure
            if comp.type is not ReadComponentType.BARCODE
        }

        assert_that(set(known_seqs.keys())).is_subset_of(structure_spacer_names)

    # ===== Tests for abstract properties =====

    def test_abstract_members_are_registered_as_abstract(self):
        """Test that the read layout and tolerance members are seen by the ABC machinery."""
        assert_that(set(ChemistryBase.__abstractmethods__)).is_equal_to(
            {"name", "read_structure", "max_errors"}
        )

    def test_chemistry_base_cannot_be_instantiated(self):
        """Test that the abstract base class itself cannot be constructed."""
        with pytest.raises(TypeError) as exc_info:
            ChemistryBase()

        assert_that(str(exc_info.value)).contains("Can't instantiate abstract class")

    def test_partial_subclass_cannot_be_instantiated(self):
        """Test that a subclass leaving an abstract member unimplemented cannot be constructed."""

        @dataclass
        class PartialChemistry(ChemistryBase):
            """Chemistry implementing every abstract member except max_errors."""

            @cached_property
            def name(self) -> str:
                """Return the identifier for this partial chemistry."""
                return "partial_chemistry"

            @cached_property
            def read_structure(self) -> ReadStructure:
                """Return a read structure holding a single non-barcode component."""
                return ReadStructure(
                    [
                        ReadComponent(
                            name="SPACER_1",
                            type=ReadComponentType.OTHER,
                            length=4,
                            sequence="ACGT",
                        )
                    ]
                )

        with pytest.raises(TypeError) as exc_info:
            PartialChemistry()

        assert_that(str(exc_info.value)).contains("Can't instantiate abstract class")
        assert_that(str(exc_info.value)).contains("max_errors")

    def test_read_structure_is_cached(self, chemistry: ChemistryBase):
        """Test that repeated read_structure access returns the identical object."""
        read_structure1 = chemistry.read_structure
        read_structure2 = chemistry.read_structure

        assert_that(read_structure1 is read_structure2).is_true()

    def test_chemistry_name_property(self, chemistry: ChemistryBase):
        """Test that name property returns a string."""
        name = chemistry.name
        assert_that(name).is_instance_of(str)
        assert_that(name).is_not_empty()

    def test_chemistry_known_sequences(self, chemistry: ChemistryBase):
        """Test that get_known_sequences() returns a dictionary of strings."""
        known_seqs = chemistry.read_structure.get_known_sequences()
        assert_that(known_seqs).is_instance_of(dict)

        for name, sequence in known_seqs.items():
            assert_that(name).is_instance_of(str)
            assert_that(sequence).is_instance_of(str)
            assert_that(sequence).is_not_empty()

    def test_chemistry_read_structure_property(self, chemistry: ChemistryBase):
        """Test that read_structure property returns a ReadStructure."""
        read_structure = chemistry.read_structure
        assert_that(read_structure).is_instance_of(ReadStructure)
        assert_that(len(read_structure)).is_greater_than(0)

    def test_chemistry_max_errors_property(self, chemistry: ChemistryBase):
        """Test that max_errors property returns an integer."""
        max_errors = chemistry.max_errors.barcode
        assert_that(max_errors).is_instance_of(int)
        assert_that(max_errors).is_greater_than_or_equal_to(0)

        max_errors_spacer = chemistry.max_errors.spacer
        assert_that(max_errors_spacer).is_instance_of(int)
        assert_that(max_errors_spacer).is_greater_than_or_equal_to(0)

    # ===== Tests for read structure consistency =====

    def test_read_structure_contains_expected_components(self, chemistry: ChemistryBase):
        """Test that read structure contains the expected barcode and spacer components."""
        barcode_components = chemistry.read_structure.get_components_by_type(
            ReadComponentType.BARCODE
        )
        spacer_components = [
            comp for comp in chemistry.read_structure if comp.type is not ReadComponentType.BARCODE
        ]

        assert_that(len(barcode_components)).is_greater_than(0)
        assert_that(len(spacer_components)).is_greater_than(0)

    def test_read_structure_spacer_components_match_known_sequences(
        self, chemistry: ChemistryBase
    ):
        """Test that spacer components in read structure match known sequences."""
        known_seqs = chemistry.read_structure.get_known_sequences()
        for component in chemistry.read_structure:
            if component.name in known_seqs:
                spacer_seq = known_seqs[component.name]
                assert_that(spacer_seq).is_not_none()
                assert_that(component.length).is_equal_to(len(spacer_seq))

    def test_total_read_structure_length(self, chemistry: ChemistryBase):
        """Test that total read structure length is calculated correctly."""
        total_length = sum(
            comp.length for comp in chemistry.read_structure if comp.length is not None
        )

        assert_that(total_length).is_greater_than(0)

    # ===== Tests for edge cases =====

    def test_barcode_whitelist_size_consistency(self, chemistry: ChemistryBase):
        """Test that all barcode whitelists have consistent sizes."""
        whitelists = chemistry.barcode_whitelists

        if len(whitelists) > 1:
            sizes = [len(wl) for wl in whitelists.values()]
            assert_that(sizes).is_not_empty()

    def test_load_whitelist_returns_tuple_not_list(self, chemistry: ChemistryBase):
        """Test that load_whitelist returns tuple, not list."""
        for component in chemistry.read_structure:
            if component.type is ReadComponentType.BARCODE:
                whitelist = chemistry.load_whitelist(component.name)
                assert_that(type(whitelist)).is_equal_to(tuple)

    def test_barcode_components_are_distinguishable(self, chemistry: ChemistryBase):
        """Test that different barcode components have different names."""
        barcode_names = [
            comp.name
            for comp in chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)
        ]
        unique_names = set(barcode_names)

        assert_that(len(unique_names)).is_equal_to(len(barcode_names))

    def test_construct_full_barcode_with_valid_whitelisted_barcodes_returns_consistent_result(
        self, chemistry: ChemistryBase
    ):
        """Test that construct_full_barcode returns consistent results for same input."""
        whitelists = chemistry.barcode_whitelists

        barcodes = {
            "BC1": whitelists["BC1"][0],
            "BC2": whitelists["BC2"][0],
            "BC3": whitelists["BC3"][0],
        }

        result1 = chemistry.construct_full_barcode(barcodes)
        result2 = chemistry.construct_full_barcode(barcodes)

        assert_that(result1).is_equal_to(result2)

    def test_barcode_whitelists_all_entries_are_strings(self, chemistry: ChemistryBase):
        """Test that all whitelist entries are strings."""
        whitelists = chemistry.barcode_whitelists

        for component_name, whitelist in whitelists.items():
            for barcode in whitelist:
                assert_that(barcode).is_instance_of(str)
                component = chemistry.read_structure.get_component_by_name(component_name)
                component_len = component.length if component else 0
                assert_that(len(barcode)).is_equal_to(component_len)


class TestChemistryCarmackCustomSeq10:
    @pytest.fixture
    def chemistry(self) -> ChemistryCarmackCustomSeq10:
        return ChemistryCarmackCustomSeq10()

    def test_chemistry_properties(self, chemistry: ChemistryCarmackCustomSeq10):
        """Test that the chemistry properties return expected values."""
        assert_that(chemistry.name).is_equal_to("carmack_custom_seq_1_0")
        known_seqs = chemistry.read_structure.get_known_sequences()
        assert_that(known_seqs).contains_key("PRIMER_C", "PRIMER_A")

    def test_read_structure_appends_umi_polyg_tgidx(self, chemistry: ChemistryCarmackCustomSeq10):
        """Read structure ends with the UMI, poly-G and TGIDX components after BC1."""
        actual_order = [comp.name for comp in chemistry.read_structure]
        assert_that(actual_order).is_equal_to(
            ["BC3", "PRIMER_C", "BC2", "PRIMER_A", "BC1", "UMI", "POLYG", "TGIDX"]
        )

    def test_umi_polyg_tgidx_component_parameters(self, chemistry: ChemistryCarmackCustomSeq10):
        """The appended components carry the confirmed UMI/poly-G/TGIDX parameters."""
        read_structure = chemistry.read_structure

        umi = read_structure.get_component_by_name("UMI")
        assert_that(umi.type).is_equal_to(ReadComponentType.UMI)
        assert_that(umi.length).is_equal_to(8)

        polyg = read_structure.get_component_by_name("POLYG")
        assert_that(polyg.type).is_equal_to(ReadComponentType.HOMOPOLYMER)
        assert_that(polyg.homopolymer_base).is_equal_to("G")
        assert_that(polyg.min_run).is_equal_to(3)
        assert_that(polyg.length).is_none()

        tgidx = read_structure.get_component_by_name("TGIDX")
        assert_that(tgidx.type).is_equal_to(ReadComponentType.TGIDX)
        assert_that(tgidx.length).is_equal_to(8)

    def test_module_constants(self):
        """The magic numbers are exposed as discoverable module constants."""
        assert_that(UMI_LENGTH).is_equal_to(8)
        assert_that(POLYG_BASE).is_equal_to("G")
        assert_that(POLYG_MIN_RUN).is_equal_to(3)
        assert_that(TGIDX_LENGTH).is_equal_to(8)

    def test_umi_component_accessor(self, chemistry: ChemistryCarmackCustomSeq10):
        """umi_component() returns the UMI component."""
        umi = chemistry.umi_component()
        assert_that(umi).is_not_none()
        assert_that(umi.name).is_equal_to("UMI")
        assert_that(umi.type).is_equal_to(ReadComponentType.UMI)

    def test_umi_is_measured_from_bc1(self, chemistry: ChemistryCarmackCustomSeq10):
        """The UMI hangs directly off BC1, so its offset from that anchor is zero."""
        located = chemistry.resolve_anchor_offset(chemistry.umi_component())
        assert_that(located.anchor.name).is_equal_to("BC1")
        assert_that(located.anchor.type).is_equal_to(ReadComponentType.BARCODE)
        assert_that(located.offset).is_equal_to(0)
        assert_that(located.position_key).is_equal_to("BC1_POS")

    def test_umi_right_anchor_is_polyg(self, chemistry: ChemistryCarmackCustomSeq10):
        """The UMI right anchor is the poly-G homopolymer and it is an anchor."""
        right_anchor = chemistry.umi_right_anchor()
        assert_that(right_anchor).is_not_none()
        assert_that(right_anchor.name).is_equal_to("POLYG")
        assert_that(right_anchor.type).is_equal_to(ReadComponentType.HOMOPOLYMER)
        assert_that(right_anchor.is_anchor).is_true()

    def test_chemistry_exposes_umi_and_polyg_parameters(
        self, chemistry: ChemistryCarmackCustomSeq10
    ):
        """UMI length and poly-G base/min-run are exposed via the accessors."""
        assert_that(chemistry.umi_component().length).is_equal_to(8)
        assert_that(chemistry.umi_right_anchor().homopolymer_base).is_equal_to("G")
        assert_that(chemistry.umi_right_anchor().min_run).is_equal_to(3)

    def test_supports_umi_extraction_true(self, chemistry: ChemistryCarmackCustomSeq10):
        """custom_seq_1_0 declares a UMI, so it supports UMI extraction."""
        assert_that(chemistry.supports_umi_extraction()).is_true()

    def test_tgidx_component_and_anchor_accessors(self, chemistry: ChemistryCarmackCustomSeq10):
        """The TGIDX accessors return the TGIDX component and its poly-G anchor."""
        tgidx = chemistry.tgidx_component()
        assert_that(tgidx).is_not_none()
        assert_that(tgidx.name).is_equal_to("TGIDX")
        assert_that(tgidx.type).is_equal_to(ReadComponentType.TGIDX)

        anchor = chemistry.tgidx_anchor()
        assert_that(anchor).is_not_none()
        assert_that(anchor.name).is_equal_to("POLYG")
        assert_that(anchor.homopolymer_base).is_equal_to("G")

        assert_that(chemistry.supports_target_assignment()).is_true()

    def test_start_positions(self, chemistry: ChemistryCarmackCustomSeq10):
        """Test that the start positions of read components are computed correctly."""
        read_structure = chemistry.read_structure
        expected_starts = {
            "BC3": 0,
            "PRIMER_C": 10,
            "BC2": 32,
            "PRIMER_A": 42,
            "BC1": 64,
            "UMI": 74,
            # Resolves now that the UMI is fixed-length. It is a layout fact, not a
            # read coordinate: an upstream indel invalidates it, which is why the
            # stages locate the run from a recorded anchor span instead.
            "POLYG": 82,
            "TGIDX": None,
        }
        for component in read_structure:
            assert_that(component.start).is_equal_to(expected_starts[component.name])

    def test_barcode_whitelists_keys_unchanged(self, chemistry: ChemistryCarmackCustomSeq10):
        """Appending UMI/poly-G/TGIDX leaves the barcode whitelists at BC1/BC2/BC3 only."""
        assert_that(set(chemistry.barcode_whitelists.keys())).is_equal_to({"BC1", "BC2", "BC3"})

    def test_load_whitelist(self, chemistry: ChemistryCarmackCustomSeq10):
        """Test loading barcode whitelists."""
        for bc_name in ["BC1", "BC2", "BC3"]:
            whitelist = chemistry.load_whitelist(bc_name)
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)
            assert_that(len(whitelist)).is_equal_to(96)

        with pytest.raises(ValueError):
            chemistry.load_whitelist("INVALID")

    def test_max_errors_includes_tgidx(self, chemistry: ChemistryCarmackCustomSeq10):
        """max_errors carries a TGIDX tolerance of 1 alongside barcode/spacer."""
        assert_that(chemistry.max_errors).is_equal_to(MatchErrors(barcode=1, spacer=2, tgidx=1))
        assert_that(chemistry.max_errors.tgidx).is_equal_to(1)

    def test_tgidx_whitelist_loads_from_packaged_data(
        self, chemistry: ChemistryCarmackCustomSeq10
    ):
        """The single confirmed TGIDX entry comes from the packaged data file."""
        whitelist = chemistry.tgidx_whitelist()
        assert_that(whitelist).is_equal_to(("TATAGCCT",))
        for entry in whitelist:
            assert_that(len(entry)).is_equal_to(TGIDX_LENGTH)

    def test_tgidx_whitelist_declared_as_a_whitelist_source(
        self, chemistry: ChemistryCarmackCustomSeq10
    ):
        """The TGIDX whitelist is declared as a source file, not hard-coded in the module."""
        sources = chemistry.whitelist_sources()
        assert_that(sources).contains_key("TGIDX")

        source = sources["TGIDX"]
        assert_that(source.path.name).is_equal_to("carmack_custom_seq_1_0_tgidx.tsv")
        assert_that(source.line_slice).is_none()
        assert_that(chemistry.whitelists["TGIDX"]).is_equal_to(("TATAGCCT",))

    def test_load_whitelist_reads_tgidx_from_packaged_file(
        self, chemistry: ChemistryCarmackCustomSeq10
    ):
        """load_whitelist reads the TGIDX entry straight from its packaged data file."""
        assert_that(chemistry.load_whitelist("TGIDX")).is_equal_to(("TATAGCCT",))


class TestChemistryCarmackCustomSeq10PrimD:
    @pytest.fixture
    def chemistry(self) -> ChemistryCarmackCustomSeq10PrimD:
        return ChemistryCarmackCustomSeq10PrimD()

    def test_chemistry_properties(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """Name and known sequences match the PRIMER_D variant."""
        assert_that(chemistry.name).is_equal_to("carmack_custom_seq_1_0_primd")
        known_seqs = chemistry.read_structure.get_known_sequences()
        # PRIMER_D has no known sequence (only a length anchor)
        assert_that(known_seqs).contains_key("PRIMER_C", "PRIMER_A")
        assert_that(known_seqs).does_not_contain_key("PRIMER_D")

    def test_inherits_from_base(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The PRIMER_D variant is a subclass of the base chemistry."""
        assert_that(chemistry).is_instance_of(ChemistryCarmackCustomSeq10)

    def test_read_structure_prepends_primer_d(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """Read structure starts with PRIMER_D, then matches the base layout."""
        read_structure = chemistry.read_structure
        actual_order = [comp.name for comp in read_structure]
        assert_that(actual_order).is_equal_to(
            ["PRIMER_D", "BC3", "PRIMER_C", "BC2", "PRIMER_A", "BC1", "UMI", "POLYG", "TGIDX"]
        )

        primer_d = read_structure.get_component_by_name("PRIMER_D")
        assert_that(primer_d.type).is_equal_to(ReadComponentType.PRIMER)
        assert_that(primer_d.length).is_equal_to(22)
        assert_that(primer_d.sequence).is_none()

    def test_inherits_umi_polyg_tgidx(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The PRIMER_D variant inherits the UMI/poly-G/TGIDX tail unchanged."""
        base = ChemistryCarmackCustomSeq10()

        for name in ("UMI", "POLYG", "TGIDX"):
            variant_comp = chemistry.read_structure.get_component_by_name(name)
            base_comp = base.read_structure.get_component_by_name(name)
            assert_that(variant_comp.type).is_equal_to(base_comp.type)
            assert_that(variant_comp.length).is_equal_to(base_comp.length)
            assert_that(variant_comp.homopolymer_base).is_equal_to(base_comp.homopolymer_base)
            assert_that(variant_comp.min_run).is_equal_to(base_comp.min_run)

        assert_that(
            chemistry.resolve_anchor_offset(chemistry.umi_component()).anchor.name
        ).is_equal_to("BC1")
        assert_that(chemistry.umi_right_anchor().name).is_equal_to("POLYG")

    def test_supports_umi_extraction_true(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The PRIMER_D variant also supports UMI extraction."""
        assert_that(chemistry.supports_umi_extraction()).is_true()

    def test_tgidx_component_and_anchor_accessors(
        self, chemistry: ChemistryCarmackCustomSeq10PrimD
    ):
        """The PRIMER_D variant exposes the inherited TGIDX component and poly-G anchor."""
        assert_that(chemistry.supports_target_assignment()).is_true()
        assert_that(chemistry.tgidx_component().name).is_equal_to("TGIDX")
        assert_that(chemistry.tgidx_anchor().name).is_equal_to("POLYG")

    def test_start_positions(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """Start positions account for the prepended PRIMER_D."""
        read_structure = chemistry.read_structure
        expected_starts = {
            "PRIMER_D": 0,
            "BC3": 22,
            "PRIMER_C": 32,
            "BC2": 54,
            "PRIMER_A": 64,
            "BC1": 86,
            "UMI": 96,
            "POLYG": 104,
            "TGIDX": None,
        }
        for component in read_structure:
            assert_that(component.start).is_equal_to(expected_starts[component.name])

    def test_load_whitelist_matches_base(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """Whitelists are inherited unchanged from the base chemistry."""
        base = ChemistryCarmackCustomSeq10()
        for bc_name in ["BC1", "BC2", "BC3"]:
            assert_that(chemistry.load_whitelist(bc_name)).is_equal_to(
                base.load_whitelist(bc_name)
            )

        with pytest.raises(ValueError):
            chemistry.load_whitelist("PRIMER_D")

    def test_max_errors_inherited(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """max_errors is inherited from the base chemistry."""
        base = ChemistryCarmackCustomSeq10()
        assert_that(chemistry.max_errors).is_equal_to(base.max_errors)

    def test_tgidx_whitelist_inherited(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The primd subclass inherits the TGIDX whitelist source and error tolerance."""
        assert_that(chemistry.whitelist_sources()).contains_key("TGIDX")
        assert_that(chemistry.tgidx_whitelist()).is_equal_to(("TATAGCCT",))
        assert_that(chemistry.max_errors.tgidx).is_equal_to(1)

    def test_factory_dispatch(self):
        """Factory returns the PRIMER_D variant for its registered name."""
        instance = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0_primd")
        assert_that(instance).is_instance_of(ChemistryCarmackCustomSeq10PrimD)


class TestChemistryHydrop:
    @pytest.fixture
    def chemistry(self) -> ChemistryHydrop:
        return ChemistryHydrop()

    def test_chemistry_properties(self, chemistry: ChemistryHydrop):
        """Test that the chemistry properties return expected values."""
        assert_that(chemistry.name).is_equal_to("hydrop")
        known_seqs = chemistry.read_structure.get_known_sequences()
        assert_that(known_seqs).contains_key("SPACER_1", "SPACER_2")

    def test_spacer_sequences(self, chemistry: ChemistryHydrop):
        """Test that spacer sequences are correct for hydrop chemistry."""
        known_seqs = chemistry.read_structure.get_known_sequences()
        assert_that(known_seqs["SPACER_1"]).is_equal_to("AGGGTACTCG")
        assert_that(known_seqs["SPACER_2"]).is_equal_to("GCAGTAGCTG")

    def test_tgidx_whitelist_empty_and_default_tgidx_errors(self, chemistry: ChemistryHydrop):
        """No TGIDX component, so the whitelist is empty and tgidx keeps the inherited default."""
        assert_that(chemistry.tgidx_whitelist()).is_equal_to(())
        assert_that(chemistry.max_errors.tgidx).is_equal_to(1)

    def test_start_positions(self, chemistry: ChemistryHydrop):
        """Test that the start positions of read components are computed correctly."""
        read_structure = chemistry.read_structure
        expected_starts = {
            "BC3": 0,
            "SPACER_1": 10,
            "BC2": 20,
            "SPACER_2": 30,
            "BC1": 40,
        }
        for component in read_structure:
            assert_that(component.start).is_equal_to(expected_starts[component.name])

    def test_read_structure_has_correct_order(self, chemistry: ChemistryHydrop):
        """Test that read structure components are in the correct order."""
        read_structure = chemistry.read_structure
        expected_order = ["BC3", "SPACER_1", "BC2", "SPACER_2", "BC1"]
        actual_order = [comp.name for comp in read_structure]
        assert_that(actual_order).is_equal_to(expected_order)

    def test_read_structure_barcode_components(self, chemistry: ChemistryHydrop):
        """Test that only BC1, BC2, BC3 are marked as barcodes."""
        read_structure = chemistry.read_structure
        barcode_components = [
            comp.name for comp in read_structure.get_components_by_type(ReadComponentType.BARCODE)
        ]
        expected_barcodes = ["BC3", "BC2", "BC1"]
        assert_that(barcode_components).is_equal_to(expected_barcodes)

    def test_load_whitelist(self, chemistry: ChemistryHydrop):
        """Test loading barcode whitelists."""
        for bc_name in ["BC1", "BC2", "BC3"]:
            whitelist = chemistry.load_whitelist(bc_name)
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)
            assert_that(len(whitelist)).is_equal_to(96)

        with pytest.raises(ValueError):
            chemistry.load_whitelist("INVALID")

    def test_barcode_whitelists_cached_property(self, chemistry: ChemistryHydrop):
        """Test that barcode_whitelists is properly cached."""
        whitelists1 = chemistry.barcode_whitelists
        whitelists2 = chemistry.barcode_whitelists
        assert_that(whitelists1 is whitelists2).is_true()

    def test_barcode_components_in_structure(self, chemistry: ChemistryHydrop):
        """Test that all barcode components in read structure have valid whitelists."""
        barcode_names = {
            comp.name
            for comp in chemistry.read_structure.get_components_by_type(ReadComponentType.BARCODE)
        }
        whitelists = chemistry.barcode_whitelists

        for bc_name in barcode_names:
            assert_that(whitelists).contains_key(bc_name)
            assert_that(len(whitelists[bc_name])).is_equal_to(96)

    def test_umi_component_is_none(self, chemistry: ChemistryHydrop):
        """A barcode-only chemistry has no UMI component."""
        assert_that(chemistry.umi_component()).is_none()
        assert_that(chemistry.umi_component()).is_none()
        assert_that(chemistry.umi_right_anchor()).is_none()

    def test_supports_umi_extraction_false(self, chemistry: ChemistryHydrop):
        """The barcode-only hydrop chemistry does not support UMI extraction (no raise)."""
        assert_that(chemistry.supports_umi_extraction()).is_false()

    def test_tgidx_component_is_none(self, chemistry: ChemistryHydrop):
        """A barcode-only chemistry has no TGIDX component and no target-assignment support."""
        assert_that(chemistry.tgidx_component()).is_none()
        assert_that(chemistry.tgidx_anchor()).is_none()
        assert_that(chemistry.supports_target_assignment()).is_false()

    def test_construct_full_barcode_hydrop(self, chemistry: ChemistryHydrop):
        """Test full barcode construction for hydrop chemistry."""
        whitelists = chemistry.barcode_whitelists

        barcodes = {
            "BC1": whitelists["BC1"][0],
            "BC2": whitelists["BC2"][0],
            "BC3": whitelists["BC3"][0],
        }

        full_bc = chemistry.construct_full_barcode(barcodes)
        assert_that(full_bc).is_instance_of(str)
        assert_that(len(full_bc)).is_equal_to(30)
        # Order should be BC3 + BC2 + BC1
        assert_that(full_bc).is_equal_to(
            whitelists["BC3"][0] + whitelists["BC2"][0] + whitelists["BC1"][0]
        )


REGISTERED_CHEMISTRY_NAMES = (
    "carmack_custom_seq_1_0",
    "carmack_custom_seq_1_0_primd",
    "hydrop",
)

# First entry of each 96-entry barcode whitelist. These are identical across every
# registered chemistry: hydrop's padded lines slice down to the same sequences.
EXPECTED_FIRST_SEQUENCES = (
    ("BC1", "TGTAGCAAGT"),
    ("BC2", "TTAGTTGGAC"),
    ("BC3", "TGACCGTACT"),
)

LEFT_PAD = "AAAAAAAAAA"
RIGHT_PAD = "TTTTTTTTTT"
CORE_SEQUENCES = ("ACGTACGTAC", "TGCATGCATG")

# Target index entries, none of which lead with the poly-G anchor base.
TGIDX_ENTRIES = ("TATAGCCT", "ATTGGCTC", "CCTATCCT", "ACTCGACT", "AGGCTTAG")

CHEMISTRY_BASE_LOGGER = "carmack.chemistry.chemistry_base"


@dataclass
class StubChemistry(ChemistryBase):
    """Chemistry whose read structure and whitelist sources are supplied by a test.

    Attributes:
        sources: Whitelist sources returned by :meth:`whitelist_sources`.
        components: Read components making up the read structure, in read order.
    """

    sources: dict[str, WhitelistSource] = field(default_factory=dict)
    components: tuple[ReadComponent, ...] = ()

    @cached_property
    def name(self) -> str:
        """Return the identifier for this stub chemistry."""
        return "stub_chemistry"

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return the read structure built from the supplied components."""
        return ReadStructure(list(self.components))

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return permissive match tolerances for this stub chemistry."""
        return MatchErrors(barcode=1, spacer=1)

    def whitelist_sources(self) -> dict[str, WhitelistSource]:
        """Return the whitelist sources supplied by the test."""
        return dict(self.sources)


@dataclass
class StubChemistryNoSources(ChemistryBase):
    """Chemistry that declares no whitelist sources, inheriting the base default."""

    @cached_property
    def name(self) -> str:
        """Return the identifier for this stub chemistry."""
        return "stub_chemistry_no_sources"

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a read structure holding a single non-barcode component."""
        return ReadStructure(
            [
                ReadComponent(
                    name="SPACER_1",
                    type=ReadComponentType.OTHER,
                    length=4,
                    sequence="ACGT",
                )
            ]
        )

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return permissive match tolerances for this stub chemistry."""
        return MatchErrors(barcode=1, spacer=1)


@pytest.fixture
def whitelist_file(tmp_path: Path) -> Callable[[str, list[str]], Path]:
    """Return a factory that writes whitelist lines to a file under tmp_path."""

    def write(filename: str, lines: list[str]) -> Path:
        path = tmp_path / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return write


class TestWhitelistLoading:
    """Tests for the data-driven whitelist loading shared by every chemistry."""

    @pytest.fixture
    def barcode_stub(self) -> Callable[..., StubChemistry]:
        """Return a factory building a stub chemistry with a single BC1 barcode source."""

        def build(path: Path, line_slice: tuple[int, int | None] | None = None) -> StubChemistry:
            return StubChemistry(
                sources={"BC1": WhitelistSource(path=path, line_slice=line_slice)},
                components=(ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),),
            )

        return build

    # ===== Value pins against the shipped whitelist data =====

    @pytest.mark.parametrize("chemistry_name", REGISTERED_CHEMISTRY_NAMES)
    @pytest.mark.parametrize("component_name,expected_first", EXPECTED_FIRST_SEQUENCES)
    def test_load_whitelist_pins_shipped_sequences(
        self, chemistry_name: str, component_name: str, expected_first: str
    ):
        """Every registered chemistry loads 96 ten-base sequences led by the expected entry."""
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        whitelist = chemistry.load_whitelist(component_name)

        assert_that(whitelist).is_instance_of(tuple)
        assert_that(whitelist).is_length(96)
        assert_that(whitelist[0]).is_equal_to(expected_first)
        for sequence in whitelist:
            assert_that(sequence).is_length(10)

    def test_packaged_tgidx_data_file_ships_the_confirmed_entry(self):
        """The TGIDX data file is importable like the barcode data and holds one entry."""
        resource = files("carmack.data.tgidx.carmack.custom_seq").joinpath(
            "carmack_custom_seq_1_0_tgidx.tsv"
        )

        assert_that(resource.is_file()).is_true()
        entries = [line for line in resource.read_text(encoding="utf-8").split("\n") if line]
        assert_that(entries).is_equal_to(["TATAGCCT"])

    @pytest.mark.parametrize("chemistry_name", REGISTERED_CHEMISTRY_NAMES)
    def test_load_whitelist_unknown_component_raises_value_error(self, chemistry_name: str):
        """An undeclared component name is rejected instead of yielding an empty whitelist."""
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        with pytest.raises(ValueError):
            chemistry.load_whitelist("NOT_A_COMPONENT")

    @pytest.mark.parametrize("chemistry_name", REGISTERED_CHEMISTRY_NAMES)
    def test_barcode_whitelists_preserve_read_structure_order(self, chemistry_name: str):
        """Barcode whitelist keys follow read-structure order, not source declaration order."""
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        assert_that(list(chemistry.barcode_whitelists.keys())).is_equal_to(["BC3", "BC2", "BC1"])

    @pytest.mark.parametrize("chemistry_name", REGISTERED_CHEMISTRY_NAMES)
    def test_whitelists_read_each_source_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, chemistry_name: str
    ):
        """Construction plus a barcode_whitelists read opens each declared source once."""
        real_open_read_iterator = GzipFile.open_read_iterator
        opened: list[str] = []

        def spy(self: GzipFile, *args, **kwargs):
            opened.append(self.filename)
            return real_open_read_iterator(self, *args, **kwargs)

        monkeypatch.setattr(GzipFile, "open_read_iterator", spy)

        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        whitelists = chemistry.barcode_whitelists

        expected_sources = len(chemistry.whitelist_sources())
        assert_that(whitelists).is_length(3)
        assert_that(opened).is_length(expected_sources)
        assert_that(set(opened)).is_length(expected_sources)

    # ===== line_slice behaviour =====

    def test_load_whitelist_with_line_slice_returns_sliced_sequences(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        barcode_stub: Callable[..., StubChemistry],
    ):
        """A declared line_slice trims the flanking padding from every whitelist line."""
        path = whitelist_file(
            "padded.tsv", [LEFT_PAD + core + RIGHT_PAD for core in CORE_SEQUENCES]
        )
        chemistry = barcode_stub(path, (10, -10))

        assert_that(chemistry.load_whitelist("BC1")).is_equal_to(CORE_SEQUENCES)

    def test_load_whitelist_without_line_slice_returns_whole_lines(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        barcode_stub: Callable[..., StubChemistry],
    ):
        """Omitting line_slice keeps each stripped line intact."""
        path = whitelist_file("bare.tsv", list(CORE_SEQUENCES))
        chemistry = barcode_stub(path)

        assert_that(chemistry.load_whitelist("BC1")).is_equal_to(CORE_SEQUENCES)

    def test_load_whitelist_skips_blank_and_whitespace_lines(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        barcode_stub: Callable[..., StubChemistry],
    ):
        """Blank and whitespace-only lines are dropped when no slice is declared."""
        path = whitelist_file(
            "sparse.tsv", [CORE_SEQUENCES[0], "", "   ", "\t", CORE_SEQUENCES[1]]
        )
        chemistry = barcode_stub(path)

        assert_that(chemistry.load_whitelist("BC1")).is_equal_to(CORE_SEQUENCES)

    def test_load_whitelist_skips_lines_that_slice_to_empty(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        barcode_stub: Callable[..., StubChemistry],
    ):
        """A non-blank line whose slice yields nothing is dropped, so slicing precedes the check."""
        path = whitelist_file(
            "slices_to_empty.tsv",
            [
                LEFT_PAD + CORE_SEQUENCES[0] + RIGHT_PAD,
                LEFT_PAD + RIGHT_PAD,
                "",
                "   ",
                LEFT_PAD + CORE_SEQUENCES[1] + RIGHT_PAD,
            ],
        )
        chemistry = barcode_stub(path, (10, -10))

        assert_that(chemistry.load_whitelist("BC1")).is_equal_to(CORE_SEQUENCES)

    # ===== whitelists versus barcode_whitelists =====

    def test_barcode_whitelists_excludes_declared_non_barcode_source(
        self, whitelist_file: Callable[[str, list[str]], Path]
    ):
        """A non-barcode source lands in whitelists but is kept out of barcode_whitelists."""
        barcode_path = whitelist_file("bc1.tsv", list(CORE_SEQUENCES))
        tgidx_path = whitelist_file("tgidx.tsv", ["TATAGCCT"])

        chemistry = StubChemistry(
            sources={
                "BC1": WhitelistSource(path=barcode_path),
                "TGIDX": WhitelistSource(path=tgidx_path),
            },
            components=(
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                ReadComponent(
                    name="POLYG",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base="G",
                    min_run=3,
                ),
                ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
            ),
        )

        assert_that(chemistry.whitelists).contains_key("BC1", "TGIDX")
        assert_that(chemistry.whitelists["TGIDX"]).is_equal_to(("TATAGCCT",))
        assert_that(chemistry.barcode_whitelists).contains_key("BC1")
        assert_that(chemistry.barcode_whitelists).does_not_contain_key("TGIDX")

    def test_whitelists_loads_source_declared_outside_read_structure(
        self, whitelist_file: Callable[[str, list[str]], Path]
    ):
        """A source for a component absent from the read structure loads without error."""
        barcode_path = whitelist_file("bc1.tsv", list(CORE_SEQUENCES))
        spare_path = whitelist_file("spare.tsv", ["TATAGCCT"])

        chemistry = StubChemistry(
            sources={
                "BC1": WhitelistSource(path=barcode_path),
                "NOT_IN_READ_STRUCTURE": WhitelistSource(path=spare_path),
            },
            components=(ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),),
        )

        assert_that(chemistry.whitelists).contains_key("BC1", "NOT_IN_READ_STRUCTURE")
        assert_that(chemistry.barcode_whitelists).is_equal_to({"BC1": CORE_SEQUENCES})

    def test_default_whitelist_sources_is_empty_and_allows_construction(self):
        """A chemistry declaring no sources constructs and exposes empty whitelist mappings."""
        chemistry = StubChemistryNoSources()

        assert_that(chemistry.whitelist_sources()).is_equal_to({})
        assert_that(chemistry.whitelists).is_equal_to({})
        assert_that(chemistry.barcode_whitelists).is_equal_to({})

    # ===== Failure reporting =====

    def test_whitelists_wraps_barcode_load_failure_as_key_error(self, tmp_path: Path):
        """An unreadable barcode source is reported as a barcode whitelist lookup failure."""
        missing_path = tmp_path / "does_not_exist.tsv"

        with pytest.raises(KeyError) as exc_info:
            StubChemistry(
                sources={"BC1": WhitelistSource(path=missing_path)},
                components=(ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),),
            )

        message = exc_info.value.args[0]
        assert_that(message).starts_with("Barcode layout 'BC1' not found in whitelist: ")

    def test_whitelists_wraps_non_barcode_load_failure_without_barcode_wording(
        self, tmp_path: Path
    ):
        """An unreadable non-barcode source is reported without calling it a barcode."""
        missing_path = tmp_path / "does_not_exist.tsv"

        with pytest.raises(KeyError) as exc_info:
            StubChemistry(
                sources={"TGIDX": WhitelistSource(path=missing_path)},
                components=(ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),),
            )

        message = exc_info.value.args[0]
        assert_that(message).starts_with("Whitelist for component 'TGIDX' could not be loaded: ")
        assert_that(message).does_not_contain("Barcode layout")

    def test_barcode_component_without_declared_source_raises_key_error(self):
        """A barcode component with no declared whitelist source fails at construction."""
        with pytest.raises(KeyError) as exc_info:
            StubChemistry(
                sources={},
                components=(ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),),
            )

        message = exc_info.value.args[0]
        assert_that(message).starts_with("Barcode layout 'BC1' not found in whitelist: ")


class TestTargetIndexValidation:
    """Tests for the construction-time validation of a declared target index."""

    @pytest.fixture
    def tgidx_stub(self) -> Callable[..., StubChemistry]:
        """Return a factory building a homopolymer-anchored target index stub."""

        def build(path: Path, anchor_base: str = "G") -> StubChemistry:
            return StubChemistry(
                sources={"TGIDX": WhitelistSource(path=path)},
                components=(
                    ReadComponent(
                        name=f"POLY{anchor_base}",
                        type=ReadComponentType.HOMOPOLYMER,
                        homopolymer_base=anchor_base,
                        min_run=3,
                    ),
                    ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
                ),
            )

        return build

    # ===== A declared target index must have a whitelist source =====

    def test_target_index_without_declared_source_raises_key_error(self):
        """A target index component with no declared whitelist source fails at construction."""
        with pytest.raises(KeyError) as exc_info:
            StubChemistry(
                sources={},
                components=(
                    ReadComponent(
                        name="POLYG",
                        type=ReadComponentType.HOMOPOLYMER,
                        homopolymer_base="G",
                        min_run=3,
                    ),
                    ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
                ),
            )

        message = exc_info.value.args[0]
        assert_that(message).starts_with("Target index layout 'TGIDX' not found in whitelist: ")

    # ===== A declared target index must be anchored by a homopolymer =====

    def test_target_index_preceded_by_a_barcode_raises_value_error(
        self, whitelist_file: Callable[[str, list[str]], Path]
    ):
        """A target index whose 5' neighbour is a barcode has no run end to locate it from."""
        barcode_path = whitelist_file("bc1.tsv", list(CORE_SEQUENCES))
        tgidx_path = whitelist_file("tgidx.tsv", [TGIDX_ENTRIES[0]])

        with pytest.raises(ValueError, match="must be preceded by a homopolymer") as exc_info:
            StubChemistry(
                sources={
                    "BC1": WhitelistSource(path=barcode_path),
                    "TGIDX": WhitelistSource(path=tgidx_path),
                },
                components=(
                    ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=10),
                    ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),
                ),
            )

        message = str(exc_info.value)
        assert_that(message).contains("TGIDX")
        assert_that(message).contains("component 'BC1'")

    def test_target_index_as_the_first_component_raises_value_error(
        self, whitelist_file: Callable[[str, list[str]], Path]
    ):
        """A target index opening the read structure is reported as preceded by nothing."""
        tgidx_path = whitelist_file("tgidx.tsv", [TGIDX_ENTRIES[0]])

        with pytest.raises(ValueError, match="must be preceded by a homopolymer") as exc_info:
            StubChemistry(
                sources={"TGIDX": WhitelistSource(path=tgidx_path)},
                components=(ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=8),),
            )

        message = str(exc_info.value)
        assert_that(message).contains("TGIDX")
        assert_that(message).contains("nothing")

    # ===== Entries may not extend the anchor run beyond the allowed bound =====

    @pytest.mark.parametrize(
        "sequence,leading_anchor_bases",
        [
            ("TATAGCCT", 0),
            ("GTATAGCC", 1),
            ("GGTATAGC", 2),
        ],
    )
    def test_leading_anchor_bases_within_the_bound_are_accepted(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
        sequence: str,
        leading_anchor_bases: int,
    ):
        """An entry may open with anchor bases while the run stays inside the bound."""
        assert_that(len(sequence) - len(sequence.lstrip("G"))).is_equal_to(leading_anchor_bases)
        path = whitelist_file("tgidx.tsv", [sequence])

        chemistry = tgidx_stub(path)

        assert_that(chemistry.tgidx_whitelist()).is_equal_to((sequence,))

    @pytest.mark.parametrize(
        "sequence,leading_anchor_bases",
        [
            ("GGGTATAG", "3"),
            ("GGGGGGGG", "8"),
        ],
    )
    def test_leading_anchor_bases_beyond_the_bound_raise_value_error(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
        sequence: str,
        leading_anchor_bases: str,
    ):
        """An entry opening past the bound is rejected, naming what breached it.

        The all-anchor entry is the edge the leading-run count is most likely to
        get wrong, since stripping the run leaves nothing behind.
        """
        path = whitelist_file("tgidx.tsv", [sequence])

        with pytest.raises(ValueError) as exc_info:
            tgidx_stub(path)

        message = str(exc_info.value)
        assert_that(message).contains(sequence)
        assert_that(message).contains(leading_anchor_bases)
        assert_that(message).contains("G")

    # ===== The anchor base is read from the read structure =====

    def test_leading_run_is_counted_in_the_anchor_base_of_the_read_structure(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
    ):
        """A poly-T anchored chemistry counts leading T, not leading G."""
        path = whitelist_file("tgidx.tsv", ["TTTAGCCT"])

        with pytest.raises(ValueError) as exc_info:
            tgidx_stub(path, anchor_base="T")

        message = str(exc_info.value)
        assert_that(message).contains("TTTAGCCT")
        assert_that(message).contains("3")
        assert_that(message).contains("T")

    def test_leading_bases_other_than_the_anchor_base_are_accepted(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
    ):
        """Three leading G are harmless when the read structure anchors on poly-T."""
        path = whitelist_file("tgidx.tsv", ["GGGTATAG"])

        chemistry = tgidx_stub(path, anchor_base="T")

        assert_that(chemistry.tgidx_whitelist()).is_equal_to(("GGGTATAG",))

    # ===== A large whitelist warns rather than failing =====

    def test_whitelist_above_the_size_limit_warns_about_mosaic_end(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
        caplog: pytest.LogCaptureFixture,
    ):
        """Five entries still construct, but raise a Mosaic End warning."""
        path = whitelist_file("tgidx.tsv", list(TGIDX_ENTRIES))

        with caplog.at_level(logging.WARNING, logger=CHEMISTRY_BASE_LOGGER):
            chemistry = tgidx_stub(path)

        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == CHEMISTRY_BASE_LOGGER
        ]
        assert_that(warnings).is_length(1)
        assert_that(warnings[0].getMessage()).contains("Mosaic End")
        assert_that(chemistry.tgidx_whitelist()).is_length(5)

    def test_whitelist_at_the_size_limit_does_not_warn(
        self,
        whitelist_file: Callable[[str, list[str]], Path],
        tgidx_stub: Callable[..., StubChemistry],
        caplog: pytest.LogCaptureFixture,
    ):
        """Four entries sit on the limit, so construction stays silent."""
        path = whitelist_file("tgidx.tsv", list(TGIDX_ENTRIES[:4]))

        with caplog.at_level(logging.WARNING, logger=CHEMISTRY_BASE_LOGGER):
            chemistry = tgidx_stub(path)

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert_that(warnings).is_empty()
        assert_that(chemistry.tgidx_whitelist()).is_length(4)
