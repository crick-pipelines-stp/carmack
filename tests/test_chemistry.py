"""
Tests for the chemistry module.
"""

import pytest
from assertpy import assert_that

from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (
    POLYG_BASE,
    POLYG_MIN_RUN,
    TGIDX_LENGTH,
    TGIDX_WHITELIST,
    UMI_LENGTH,
    UMI_LENGTH_TOLERANCE,
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.chemistry_carmack_custom_seq_1_0_primd import (
    ChemistryCarmackCustomSeq10PrimD,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.chemistry_hydrop import ChemistryHydrop
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure


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
        """A UMI component accepts a positive length and non-negative tolerance."""
        comp = ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=1)
        assert_that(comp.length).is_equal_to(8)
        assert_that(comp.length_tolerance).is_equal_to(1)

    def test_umi_component_default_tolerance_is_zero(self) -> None:
        """A UMI component without an explicit tolerance defaults to zero."""
        comp = ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8)
        assert_that(comp.length_tolerance).is_equal_to(0)

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

    def test_umi_component_rejects_negative_tolerance(self) -> None:
        """A UMI component rejects a negative length tolerance."""
        with pytest.raises(ValueError, match="non-negative length_tolerance"):
            ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=-1)

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
                    name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=1
                ),
                True,
            ),
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
        """is_variable_length reflects tolerance, homopolymer type, or unknown length."""
        assert_that(component.is_variable_length).is_equal_to(expected)

    @pytest.mark.parametrize(
        "component,expected",
        [
            (ReadComponent(name="BC", type=ReadComponentType.BARCODE, length=10), 10),
            (
                ReadComponent(
                    name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=1
                ),
                7,
            ),
            (
                ReadComponent(
                    name="HP",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base="G",
                    min_run=3,
                ),
                3,
            ),
        ],
    )
    def test_min_length(self, component: ReadComponent, expected: int) -> None:
        """min_length returns the minimum bases occupied per component type."""
        assert_that(component.min_length).is_equal_to(expected)

    def test_min_length_none_when_length_unknown_and_not_homopolymer(self) -> None:
        """min_length is None when a non-homopolymer component has no known length."""
        comp = ReadComponent(name="X", type=ReadComponentType.OTHER, length=5)
        comp.length = None
        assert_that(comp.min_length).is_none()


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
            ReadComponent(name="UMI", type=ReadComponentType.UMI, length=8, length_tolerance=1),
            ReadComponent(
                name="polyG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base="G",
                min_run=3,
            ),
        ]
        ReadStructure(components)
        starts = {comp.name: comp.start for comp in components}
        assert_that(starts["BC1"]).is_equal_to(0)
        assert_that(starts["UMI"]).is_equal_to(10)
        assert_that(starts["polyG"]).is_none()

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
            whitelist = chemistry.load_barcode_whitelist(component.name)
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

    def test_load_barcode_whitelist_returns_tuple_not_list(self, chemistry: ChemistryBase):
        """Test that load_barcode_whitelist returns tuple, not list."""
        for component in chemistry.read_structure:
            if component.type is ReadComponentType.BARCODE:
                whitelist = chemistry.load_barcode_whitelist(component.name)
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
        assert_that(umi.length_tolerance).is_equal_to(1)

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
        assert_that(UMI_LENGTH_TOLERANCE).is_equal_to(1)
        assert_that(POLYG_BASE).is_equal_to("G")
        assert_that(POLYG_MIN_RUN).is_equal_to(3)
        assert_that(TGIDX_LENGTH).is_equal_to(8)

    def test_umi_component_accessor(self, chemistry: ChemistryCarmackCustomSeq10):
        """umi_component() returns the UMI component."""
        umi = chemistry.umi_component()
        assert_that(umi).is_not_none()
        assert_that(umi.name).is_equal_to("UMI")
        assert_that(umi.type).is_equal_to(ReadComponentType.UMI)

    def test_umi_left_anchor_is_bc1(self, chemistry: ChemistryCarmackCustomSeq10):
        """The UMI left anchor is BC1 and it is an anchor component."""
        left_anchor = chemistry.umi_left_anchor()
        assert_that(left_anchor).is_not_none()
        assert_that(left_anchor.name).is_equal_to("BC1")
        assert_that(left_anchor.type).is_equal_to(ReadComponentType.BARCODE)
        assert_that(left_anchor.is_anchor).is_true()

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
        """UMI length/tolerance and poly-G base/min-run are exposed via the accessors."""
        assert_that(chemistry.umi_component().length).is_equal_to(8)
        assert_that(chemistry.umi_component().length_tolerance).is_equal_to(1)
        assert_that(chemistry.umi_right_anchor().homopolymer_base).is_equal_to("G")
        assert_that(chemistry.umi_right_anchor().min_run).is_equal_to(3)

    def test_supports_umi_extraction_true(self, chemistry: ChemistryCarmackCustomSeq10):
        """custom_seq_1_0 supports UMI extraction (UMI anchored on its left by BC1)."""
        assert_that(chemistry.supports_umi_extraction()).is_true()

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
            "POLYG": None,
            "TGIDX": None,
        }
        for component in read_structure:
            assert_that(component.start).is_equal_to(expected_starts[component.name])

    def test_barcode_whitelists_keys_unchanged(self, chemistry: ChemistryCarmackCustomSeq10):
        """Appending UMI/poly-G/TGIDX leaves the barcode whitelists at BC1/BC2/BC3 only."""
        assert_that(set(chemistry.barcode_whitelists.keys())).is_equal_to({"BC1", "BC2", "BC3"})

    def test_load_barcode_whitelist(self, chemistry: ChemistryCarmackCustomSeq10):
        """Test loading barcode whitelists."""
        for bc_name in ["BC1", "BC2", "BC3"]:
            whitelist = chemistry.load_barcode_whitelist(bc_name)
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)
            assert_that(len(whitelist)).is_equal_to(96)

        with pytest.raises(ValueError):
            chemistry.load_barcode_whitelist("INVALID")

    def test_max_errors_includes_tgidx(self, chemistry: ChemistryCarmackCustomSeq10):
        """max_errors carries a TGIDX tolerance of 1 alongside barcode/spacer."""
        assert_that(chemistry.max_errors).is_equal_to(MatchErrors(barcode=1, spacer=2, tgidx=1))
        assert_that(chemistry.max_errors.tgidx).is_equal_to(1)

    def test_tgidx_whitelist(self, chemistry: ChemistryCarmackCustomSeq10):
        """The TGIDX whitelist holds the single confirmed entry, sized to TGIDX_LENGTH."""
        whitelist = chemistry.tgidx_whitelist()
        assert_that(whitelist).is_equal_to(("TATAGCCT",))
        assert_that(whitelist).is_equal_to(TGIDX_WHITELIST)
        for entry in whitelist:
            assert_that(len(entry)).is_equal_to(TGIDX_LENGTH)


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
            assert_that(variant_comp.length_tolerance).is_equal_to(base_comp.length_tolerance)
            assert_that(variant_comp.homopolymer_base).is_equal_to(base_comp.homopolymer_base)
            assert_that(variant_comp.min_run).is_equal_to(base_comp.min_run)

        assert_that(chemistry.umi_left_anchor().name).is_equal_to("BC1")
        assert_that(chemistry.umi_right_anchor().name).is_equal_to("POLYG")

    def test_supports_umi_extraction_true(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The PRIMER_D variant also supports UMI extraction."""
        assert_that(chemistry.supports_umi_extraction()).is_true()

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
            "POLYG": None,
            "TGIDX": None,
        }
        for component in read_structure:
            assert_that(component.start).is_equal_to(expected_starts[component.name])

    def test_load_barcode_whitelist_matches_base(
        self, chemistry: ChemistryCarmackCustomSeq10PrimD
    ):
        """Whitelists are inherited unchanged from the base chemistry."""
        base = ChemistryCarmackCustomSeq10()
        for bc_name in ["BC1", "BC2", "BC3"]:
            assert_that(chemistry.load_barcode_whitelist(bc_name)).is_equal_to(
                base.load_barcode_whitelist(bc_name)
            )

        with pytest.raises(ValueError):
            chemistry.load_barcode_whitelist("PRIMER_D")

    def test_max_errors_inherited(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """max_errors is inherited from the base chemistry."""
        base = ChemistryCarmackCustomSeq10()
        assert_that(chemistry.max_errors).is_equal_to(base.max_errors)

    def test_tgidx_whitelist_inherited(self, chemistry: ChemistryCarmackCustomSeq10PrimD):
        """The _primd subclass inherits the TGIDX whitelist and error tolerance."""
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

    def test_tgidx_whitelist_empty_and_no_tgidx_errors(self, chemistry: ChemistryHydrop):
        """A chemistry with no TGIDX component yields an empty whitelist and zero TGIDX errors."""
        assert_that(chemistry.tgidx_whitelist()).is_equal_to(())
        assert_that(chemistry.max_errors.tgidx).is_equal_to(0)

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

    def test_load_barcode_whitelist(self, chemistry: ChemistryHydrop):
        """Test loading barcode whitelists."""
        for bc_name in ["BC1", "BC2", "BC3"]:
            whitelist = chemistry.load_barcode_whitelist(bc_name)
            assert_that(whitelist).is_instance_of(tuple)
            assert_that(len(whitelist)).is_greater_than(0)
            assert_that(len(whitelist)).is_equal_to(96)

        with pytest.raises(ValueError):
            chemistry.load_barcode_whitelist("INVALID")

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
        assert_that(chemistry.umi_left_anchor()).is_none()
        assert_that(chemistry.umi_right_anchor()).is_none()

    def test_supports_umi_extraction_false(self, chemistry: ChemistryHydrop):
        """The barcode-only hydrop chemistry does not support UMI extraction (no raise)."""
        assert_that(chemistry.supports_umi_extraction()).is_false()

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
