"""Tests for the chemistry-agnostic insert-start arithmetic shared by both trim arms.

``prepare-reads`` trims R1 down to its genomic/cDNA insert on two different arms — the
matched arm trims after the target index, the unmatched arm trims after the UMI — and
both need the same three-way answer to "where does the insert actually start". When the
read structure names no anchor at all, the reference coordinate already is the answer:
there is nothing between it and the insert to trim past. When the next component is a
homopolymer, its real extent can only be read off the actual read, because polymerase
slippage means the run present in any given read is longer or shorter than any nominal
length the chemistry could declare; that scan already exists and is already tested as
``locate_anchor_run`` in ``carmack.assign_targets.tgidx_locator``, so this module must
reuse it rather than reimplement it. Every other anchor type has a length fixed by the
chemistry, so no read inspection is needed or wanted — consulting the read in that case
would be actively wrong, and one test below is built specifically to fail if the two
branches were ever crossed.

The two integration tests exercise this dispatch through the exact chemistry the
pipeline ships, rather than through synthetic components alone, because the two
branches insert_start needs to reach in production are also the two anchors
``prepare-reads`` will call it with: the unmatched arm trims off ``umi_right_anchor()``
(a ``POLYG`` homopolymer) and the matched arm trims off ``tgidx_right_anchor()`` (the
fixed-length ``ME`` primer). Getting either chemistry accessor wrong, or the trim
arithmetic wrong for the anchor it returns, would only show up here, not in the
isolated unit cases.

The final test is structural rather than behavioural: it guards against the module
quietly growing a dependency it was designed not to need. insert_start is pure
coordinate arithmetic taking a reference position, an already-resolved anchor
component, and a sequence string; it has no reason to parse a FASTQ header itself
(``carmack.io``), nor to know about the ``ChemistryBase`` machinery or the arm-specific
``ReadPreparer`` that will be the ones calling it from both sides of the trim.
"""

import ast
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.assign_targets.tgidx_locator import locate_anchor_run
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import ME
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.io.read_annotation import ReadAnnotation
from carmack.prepare_reads.insert_locator import insert_start

CHEMISTRY = "carmack_custom_seq_1_0"

# Where the module under test lives, resolved from this file's own location rather than
# by importing it, so the structural import-check test can inspect its source text even
# though (in the red phase) the module does not exist yet.
REPO_ROOT = Path(__file__).resolve().parent.parent
INSERT_LOCATOR_PATH = REPO_ROOT / "carmack" / "prepare_reads" / "insert_locator.py"

# Synthetic primitives for the dispatch-branch unit tests below. These need no chemistry
# object at all: insert_start only ever looks at the anchor component and, on the
# homopolymer branch, the read sequence itself.
HOMOPOLYMER_ANCHOR_NAME = "POLYG"
HOMOPOLYMER_ANCHOR_BASE = "G"
HOMOPOLYMER_MIN_RUN = 3

FIXED_LENGTH_ANCHOR_LENGTH = 5


class TestInsertStartWithNoAnchor:
    """Tests for the branch reached when the read structure names no anchor at all."""

    @pytest.mark.parametrize(
        "reference,seq",
        [
            (0, "ACGTACGTACGT"),
            (12, "GGGGGGGGGGGGGGGGGGGG"),
            (37, "A"),
        ],
    )
    def test_insert_start_returns_reference_unchanged(self, reference: int, seq: str) -> None:
        """Test that a None anchor is a no-op: the reference coordinate already is the answer.

        A component with nothing 3' of it to anchor means there is nothing between the
        reference position and the insert to trim past, so neither of the two other
        branches may be consulted. The third case even sets ``reference`` past the end
        of ``seq``, which would raise or misbehave if the result were derived from
        indexing into ``seq`` rather than returned untouched.
        """
        result = insert_start(reference, None, seq)

        assert_that(result).is_equal_to(reference)


class TestInsertStartWithHomopolymerAnchor:
    """Tests for the branch reached when the next component is a homopolymer run."""

    @pytest.mark.parametrize("run_length", [2, 3, 5, 9])
    def test_insert_start_delegates_to_locate_anchor_run(self, run_length: int) -> None:
        """Test that a homopolymer anchor's real run length, not any nominal one, decides the answer.

        Polymerase slippage means the run actually present in the read is the only
        source of truth for where it ends, so this asserts both the concrete expected
        offset and, equivalently, agreement with calling ``locate_anchor_run``
        directly. Every run length used is at least 2 so the scan is genuinely
        exercised, not merely returning its input unchanged by coincidence.
        """
        prefix = "ACTACTACTACT"
        reference = len(prefix)
        seq = prefix + HOMOPOLYMER_ANCHOR_BASE * run_length + "TATAGCCT"
        anchor = ReadComponent(
            name=HOMOPOLYMER_ANCHOR_NAME,
            type=ReadComponentType.HOMOPOLYMER,
            homopolymer_base=HOMOPOLYMER_ANCHOR_BASE,
            min_run=HOMOPOLYMER_MIN_RUN,
        )

        result = insert_start(reference, anchor, seq)

        assert_that(result).is_equal_to(reference + run_length)
        assert_that(result).is_equal_to(
            locate_anchor_run(seq, reference, anchor.homopolymer_base, anchor.min_run).end
        )


class TestInsertStartWithFixedLengthAnchor:
    """Tests for the branch reached by every non-homopolymer anchor type."""

    @pytest.mark.parametrize(
        "anchor_type,extra_kwargs",
        [
            (ReadComponentType.PRIMER, {"sequence": None}),
            (ReadComponentType.PRIMER, {"sequence": "ACGTA"}),
            (ReadComponentType.BARCODE, {}),
        ],
    )
    def test_insert_start_returns_reference_plus_length(
        self, anchor_type: ReadComponentType, extra_kwargs: dict[str, str | None]
    ) -> None:
        """Test that a fixed-length anchor is plain arithmetic, whatever its concrete type.

        PRIMER and BARCODE anchors carry a length the chemistry already knows, with or
        without a known sequence, so the dispatch only needs to know the anchor is not a
        homopolymer, not which of these it specifically is.
        """
        reference = 20
        anchor = ReadComponent(
            name="X", type=anchor_type, length=FIXED_LENGTH_ANCHOR_LENGTH, **extra_kwargs
        )
        seq = "ACGTACGTACGTACGTACGT"  # arbitrary content; irrelevant on this branch

        result = insert_start(reference, anchor, seq)

        assert_that(result).is_equal_to(reference + FIXED_LENGTH_ANCHOR_LENGTH)

    def test_insert_start_ignores_seq_even_when_seq_is_too_short_to_reach_the_answer(
        self,
    ) -> None:
        """Test that the fixed-length branch never consults seq, not even to bound its answer.

        ``locate_anchor_run`` never reports a run end past ``len(seq)`` for a reference
        inside the read, so if insert_start's fixed-length branch mistakenly called it
        instead of doing plain arithmetic, the result could never exceed ``len(seq)``.
        Making seq three bases shorter than ``reference + anchor.length`` turns that
        into a ceiling the wrong branch could not cross, so this fails if the dispatch
        is ever broken, regardless of seq's content.
        """
        reference = 20
        anchor = ReadComponent(
            name="X",
            type=ReadComponentType.PRIMER,
            length=FIXED_LENGTH_ANCHOR_LENGTH,
            sequence=None,
        )
        seq = "A" * (reference + 2)

        result = insert_start(reference, anchor, seq)

        assert_that(result).is_equal_to(reference + FIXED_LENGTH_ANCHOR_LENGTH)
        assert_that(result).is_greater_than(len(seq))


class TestInsertStartWithCarmackCustomSeqChemistry:
    """Integration tests exercising insert_start through a real chemistry's own anchors.

    These reach the two branches insert_start is actually dispatched to in production:
    the unmatched arm trims after ``umi_right_anchor()`` (a homopolymer) and the matched
    arm trims after ``tgidx_right_anchor()`` (a fixed-length primer). Getting either
    chemistry accessor wrong, or the arithmetic wrong for the anchor it returns, would
    only show up here, not in the synthetic unit cases above.
    """

    @pytest.fixture
    def chemistry(self):
        """Provide the shipped chemistry both trim arms are built against.

        Returns:
            The registered ``carmack_custom_seq_1_0`` chemistry instance.
        """
        return ChemistryFactory.get_chemistry(CHEMISTRY)

    def test_unmatched_arm_trims_past_the_observed_polyg_run(self, chemistry) -> None:
        """Test the unmatched-arm branch: insert_start walks the real poly-G run past the UMI.

        The UMI's own header annotation is built the way the codebase always builds
        one — ``ReadAnnotation`` plus ``position_key``/``format_span`` to write it, then
        ``parse_span`` to read it back — even though insert_start itself never parses a
        header. This keeps the reference coordinate fed to it exactly the kind a real
        caller would already have pulled out of a ``UMI_POS`` tag.
        """
        umi_start = 92
        umi_seq = "ACTACTAT"
        run_length = 5
        read = "A" * umi_start + umi_seq + "G" * run_length + "TATAGCCT" + "CTCTTATACACATCTCCTC"

        annotation = ReadAnnotation(read_id="read1")
        annotation.set(position_key("UMI"), format_span(umi_start, umi_start + len(umi_seq)))
        _, umi_pos_end = parse_span(annotation.get(position_key("UMI")))

        anchor = chemistry.umi_right_anchor()
        result = insert_start(reference=umi_pos_end, anchor=anchor, seq=read)

        assert_that(anchor.type).is_equal_to(ReadComponentType.HOMOPOLYMER)
        assert_that(result).is_equal_to(umi_pos_end + run_length)

    def test_matched_arm_trims_past_the_fixed_length_me_primer(self, chemistry) -> None:
        """Test the matched-arm branch: insert_start adds the fixed ME length past the target index.

        The literal 19 the design proposal calls for is cross-checked here against
        ``chemistry.tgidx_right_anchor().length``, so this test cannot silently drift
        from the chemistry definition it exists to verify.
        """
        tgidx_start = 104
        tgidx_seq = "TATAGCCT"
        read = "A" * tgidx_start + tgidx_seq + ME + "CTCTTATACACATCTCCTC"

        annotation = ReadAnnotation(read_id="read1")
        annotation.set(
            position_key("TGIDX"), format_span(tgidx_start, tgidx_start + len(tgidx_seq))
        )
        _, tgidx_pos_end = parse_span(annotation.get(position_key("TGIDX")))

        anchor = chemistry.tgidx_right_anchor()
        result = insert_start(reference=tgidx_pos_end, anchor=anchor, seq=read)

        assert_that(anchor.length).is_equal_to(19)
        assert_that(result).is_equal_to(tgidx_pos_end + 19)


class TestInsertLocatorHasNoForbiddenImports:
    """Structural guard against insert_locator regaining dependencies it must not need."""

    def test_module_does_not_import_chemistry_base_read_preparer_or_carmack_io(self) -> None:
        """Test that the module's own import statements never reference the disallowed names.

        insert_start is pure coordinate arithmetic: a reference position, an
        already-resolved anchor component, and a sequence string in, an int out. It has
        no business parsing a FASTQ header (``carmack.io``), knowing about
        ``ChemistryBase`` itself, or depending on the arm-specific ``ReadPreparer`` that
        will call it from both sides of the trim. Parsing the file's own source text,
        rather than introspecting ``sys.modules``, means this only sees what the module
        itself declares, not whatever a transitive dependency happens to already have
        loaded elsewhere in the test run.

        In the red phase this fails at ``read_text()`` (the file does not exist yet),
        before it ever reaches ``ast.parse``. That is expected: there is nothing to
        check an import list against until the module exists, and the module-level
        import of ``insert_start`` at the top of this file already fails collection for
        every test below for the same reason.
        """
        source = INSERT_LOCATOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(INSERT_LOCATOR_PATH))

        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module:
                    imported_names.append(module)
                imported_names.extend(
                    f"{module}.{alias.name}" if module else alias.name for alias in node.names
                )

        assert_that(imported_names).does_not_contain("carmack.chemistry.chemistry_base")
        assert_that([name for name in imported_names if "ReadPreparer" in name]).is_empty()
        assert_that(
            [
                name
                for name in imported_names
                if name == "carmack.io" or name.startswith("carmack.io.")
            ]
        ).is_empty()
