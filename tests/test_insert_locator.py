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

This module also holds the guard both arms put over that answer. ``insert_start`` is
free to return a coordinate at, or past, the end of the read: the homopolymer branch
saturates at ``len(seq)`` once the run consumes the rest of the read, and the
fixed-length branch is arithmetic that never consults ``seq`` at all. On a 2-colour
instrument an unsequenced tail is read as the anchor base, so a cluster that dies just
after the scaffold produces exactly that saturating run and leaves no insert behind.
``insert_not_sequenced`` is the predicate over the resulting ``(cut, seq)`` pair, and
the tests for it below assert the condition it exists to express — that ``seq[cut:]``
is the empty string — never that a kept read's insert equals ``seq[cut:]``, a form an
empty insert satisfies just as happily as a real one.

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
from carmack.prepare_reads.insert_locator import insert_not_sequenced, insert_start

CHEMISTRY = "carmack_custom_seq_1_0"

# Every shipped chemistry whose reads reach the trim path. ``hydrop`` is deliberately
# absent: it declares no UMI component, so ``ReadPreparer`` rejects it long before a
# cut is ever computed for one of its reads.
TRIMMED_CHEMISTRIES = [
    "carmack_custom_seq_1_0",
    "carmack_custom_seq_1_0_primd",
]

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


class TestInsertNotSequenced:
    """Tests for the guard predicate both trim arms put over the cut insert_start returns.

    The predicate answers one question — is there anything left of the read once the
    cut is applied — and it is asked of a coordinate that is allowed to be out of
    range. That is why it is ``cut >= len(seq)`` and not ``cut == len(seq)``: the
    homopolymer branch saturates at exactly the read end, while the fixed-length branch
    adds a chemistry-declared length to a reference it never checks against the read and
    so can land beyond it, a possibility
    ``test_insert_start_ignores_seq_even_when_seq_is_too_short_to_reach_the_answer``
    above already pins for insert_start itself. Both mean the same thing downstream: a
    record with an empty sequence and an empty quality line, which is syntactically
    valid, passes every length-equality check, and desyncs the next reader.

    The cases here are deliberately built from bare coordinates and bare strings rather
    than from any chemistry, because the predicate is chemistry-agnostic by
    construction: it is told the cut, it is not asked to work one out.
    """

    @pytest.mark.parametrize(
        "cut,seq",
        [
            (0, "ACGTACGT"),
            (4, "ACGTACGT"),
            (7, "ACGTACGT"),
            (0, "A"),
        ],
    )
    def test_insert_not_sequenced_is_false_when_bases_remain_after_the_cut(
        self, cut: int, seq: str
    ) -> None:
        """Test that a cut with any read left after it reports an insert that was sequenced.

        These are the reads the stage must keep. The last two cases are the tightest
        ones the predicate has to get right: a cut one base short of the read end leaves
        a single-base insert, and a cut of zero leaves a single-base read untouched.
        Neither is a drop — nothing here filters on how *much* insert was sequenced,
        only on whether any of it was — so a guard written with ``>`` slipping to
        ``>=``'s neighbour, or one that quietly imposed a minimum length, fails here.
        """
        result = insert_not_sequenced(cut, seq)

        assert_that(result).is_false()

    @pytest.mark.parametrize("seq", ["ACGTACGT", "GGGGGGGGGGGG", "A"])
    def test_insert_not_sequenced_is_true_when_the_cut_lands_on_the_read_end(
        self, seq: str
    ) -> None:
        """Test that a cut at exactly the read end reports no insert — the observed defect.

        This is the coordinate a saturating homopolymer run produces: on a 2-colour
        instrument the absence of signal is read as the anchor base, so a cluster that
        dies just after the scaffold leaves a run of it that walks to the read end, and
        ``locate_anchor_run`` correctly reports that end. Slicing there yields the empty
        string, which is exactly the record that must never be written.
        """
        result = insert_not_sequenced(len(seq), seq)

        assert_that(result).is_true()

    @pytest.mark.parametrize("overshoot", [1, 2, 7])
    def test_insert_not_sequenced_is_true_when_the_cut_overshoots_the_read_end(
        self, overshoot: int
    ) -> None:
        """Test that a cut past the read end reports no insert, not merely a cut at it.

        The fixed-length branch adds a length the chemistry declares to a reference
        coordinate and never consults the read, so on a read that ended early it returns
        a coordinate strictly greater than ``len(seq)``. Python slices that as the empty
        string without complaint, so the resulting record is indistinguishable from the
        saturating case and must be caught by the same guard. A predicate written as
        ``cut == len(seq)`` passes every case above and fails every case here, which is
        the whole reason this test exists separately.
        """
        seq = "ACGTACGT"

        result = insert_not_sequenced(len(seq) + overshoot, seq)

        assert_that(result).is_true()

    @pytest.mark.parametrize("cut", [0, 1, 4])
    def test_insert_not_sequenced_is_true_for_an_empty_read_at_every_cut(self, cut: int) -> None:
        """Test that an empty read has no insert at any cut, including a cut of zero.

        The degenerate boundary: with ``len(seq)`` at zero, every cut is at or past the
        read end, so the predicate must hold without ever indexing into the read. A cut
        of zero is the case that separates the correct ``>=`` from a ``>``, which would
        report an empty read as having an insert.
        """
        result = insert_not_sequenced(cut, "")

        assert_that(result).is_true()

    def test_insert_not_sequenced_agrees_with_the_slice_at_the_cut_being_empty(self) -> None:
        """Test that the predicate is exactly the condition "the slice at the cut is empty".

        Stated as the definition rather than as an implementation: for every cut from
        the start of the read to well past its end, the predicate must agree with
        ``seq[cut:] == ""``, which is the property the caller actually depends on — it
        is about to write ``seq[cut:]`` into a FASTQ record. Comparing whole mappings
        rather than asserting case by case means a disagreement names the cut it
        happened at.

        This is deliberately not the shape ``written_seq == seq[cut:]``, which is
        satisfied vacuously when both sides are empty and so cannot distinguish a read
        that was correctly trimmed from one that was emptied. Asserting the cut range
        produces both answers keeps the comparison from passing on a degenerate range
        where every cut happened to fall on the same side.
        """
        seq = "ACGTACGTACGT"
        cuts = range(len(seq) + 4)

        observed = {cut: insert_not_sequenced(cut, seq) for cut in cuts}

        assert_that(observed).is_equal_to({cut: seq[cut:] == "" for cut in cuts})
        assert_that(set(observed.values())).is_equal_to({True, False})


class TestInsertNotSequencedOverRealAnchorCuts:
    """Tests putting the predicate over cuts insert_start really returns for shipped anchors.

    The unit cases above hand the predicate a coordinate directly, which says nothing
    about whether the coordinates the trim arms produce ever reach the read end. These
    build the reads that produce them, through both shipped chemistries' own anchor
    components, so every offset is derived from ``ReadStructure`` and no chemistry
    constant appears in the test at all. ``hydrop`` is absent by design: it declares no
    UMI component, so its reads never reach this path.
    """

    @pytest.mark.parametrize("chemistry_name", TRIMMED_CHEMISTRIES)
    def test_insert_not_sequenced_is_true_for_a_run_that_terminates_the_read(
        self, chemistry_name: str
    ) -> None:
        """Test the unmatched arm's real failure: the anchor run consumes the rest of the read.

        Built the way the instrument builds it — a read whose anchor base repeats from
        the reference coordinate to the last base and stops there, with nothing after
        it. The run length is taken from the anchor's own ``min_run`` so the scan is
        genuinely entered rather than falling through its shift search, and the
        assertion that the cut equals ``len(seq)`` records what makes this case reach
        the predicate at all.
        """
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        anchor = chemistry.umi_right_anchor()
        prefix = "ACTACTACTACT"
        reference = len(prefix)
        seq = prefix + anchor.homopolymer_base * (anchor.min_run + 2)

        cut = insert_start(reference=reference, anchor=anchor, seq=seq)

        assert_that(cut).is_equal_to(len(seq))
        assert_that(insert_not_sequenced(cut, seq)).is_true()

    @pytest.mark.parametrize("chemistry_name", TRIMMED_CHEMISTRIES)
    def test_insert_not_sequenced_is_true_for_a_bridged_run_that_reaches_the_read_end(
        self, chemistry_name: str
    ) -> None:
        """Test that a run carrying one interrupting base still reaches the guard.

        The trim path's scan bridges a single non-anchor base, so a tail carrying one
        sequencing error inside it still walks to the read end and still yields a cut
        with nothing after it. An unbridged scan would stop at the interruption and
        report a shorter run, which is why the two counts measured over one library
        differ — and why the guard has to sit on the bridged cut rather than on any
        separately measured run length.
        """
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        anchor = chemistry.umi_right_anchor()
        prefix = "ACTACTACTACT"
        reference = len(prefix)
        interrupted = anchor.homopolymer_base * (anchor.min_run + 2) + "A"
        seq = prefix + interrupted + anchor.homopolymer_base * anchor.min_run

        cut = insert_start(reference=reference, anchor=anchor, seq=seq)

        assert_that(cut).is_equal_to(len(seq))
        assert_that(insert_not_sequenced(cut, seq)).is_true()

    @pytest.mark.parametrize("chemistry_name", TRIMMED_CHEMISTRIES)
    def test_insert_not_sequenced_is_false_for_a_single_base_after_the_run(
        self, chemistry_name: str
    ) -> None:
        """Test the positive control: one base of insert past the run is kept, not dropped.

        The same read as the terminating case with a single base appended, so the only
        difference between being dropped and being kept is whether anything was
        sequenced after the run. Nothing here is padded and no minimum insert length is
        imposed, so a one-base insert must come back as a read that was sequenced. Kept
        alongside the drop cases because a guard that reported every read as unsequenced
        would satisfy all of them and fail only this one.
        """
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        anchor = chemistry.umi_right_anchor()
        prefix = "ACTACTACTACT"
        reference = len(prefix)
        seq = prefix + anchor.homopolymer_base * (anchor.min_run + 2) + "A"

        cut = insert_start(reference=reference, anchor=anchor, seq=seq)

        assert_that(cut).is_less_than(len(seq))
        assert_that(insert_not_sequenced(cut, seq)).is_false()

    @pytest.mark.parametrize("chemistry_name", TRIMMED_CHEMISTRIES)
    @pytest.mark.parametrize("deficit", [0, 1])
    def test_insert_not_sequenced_is_true_when_the_fixed_length_anchor_passes_the_read_end(
        self, chemistry_name: str, deficit: int
    ) -> None:
        """Test the matched arm's real failure: fixed-length arithmetic walking off the read.

        The matched arm adds the length of the anchor 3' of the target index, taken here
        from the chemistry rather than written down, and never verifies those bases are
        present. A read that ends at the anchor's last base gives a cut of exactly
        ``len(seq)``; a read one base shorter gives a cut beyond it. Both are covered by
        the one ``deficit`` parameter so the ``==`` and the ``>`` cases are exercised as
        the same situation seen from one base apart, which is what the ``>=`` in the
        predicate is for.
        """
        chemistry = ChemistryFactory.get_chemistry(chemistry_name)
        anchor = chemistry.tgidx_right_anchor()
        reference = 12
        seq = "A" * (reference + anchor.length - deficit)

        cut = insert_start(reference=reference, anchor=anchor, seq=seq)

        assert_that(cut).is_equal_to(reference + anchor.length)
        assert_that(cut).is_greater_than_or_equal_to(len(seq))
        assert_that(insert_not_sequenced(cut, seq)).is_true()


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
