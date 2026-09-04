"""Tests for the assign-targets target assigner.

The assigner streams the UMI-annotated R1 FASTQ, derives a bounded window off
the end of the anchor homopolymer run recorded by UMI extraction, matches the
target index inside that window against the chemistry whitelist and re-emits
every read carrying either a whitelisted ``TGIDX`` value with its ``TGIDX_POS``
span or ``TGIDX=NONE``.

This file covers construction: the parameters the assigner resolves from its
chemistry, the fail-fast checks that reject a chemistry it could never assign a
target from, and the header validation that rejects an annotated FASTQ produced
without UMI extraction.

It also covers the streaming assignment pass itself: the sensitivity of the
window-and-match step over synthetic annotated FASTQ files, the four mutually
exclusive per-read outcomes and their counters, and the conservation guarantee
that every input read is re-emitted exactly once carrying either a whitelist
entry or the unassigned sentinel.

Finally it covers the stage's command line wiring - the arguments the
``assign-targets`` command forwards to the assigner and the command's place in
the grouped help listing - and one end-to-end run chaining barcode extraction,
UMI extraction and target assignment over a committed golden input.
"""

import gzip
from collections.abc import Callable
from functools import cached_property
from inspect import signature
from pathlib import Path
from unittest import mock

import pytest
import rich_click as click
from assertpy import assert_that
from click.testing import CliRunner

import carmack.__main__

# The unassigned sentinel is read off the assigner module rather than imported
# by name, so this file states the module's own value instead of restating the
# literal that every downstream consumer keys on.
from carmack.assign_targets import target_assigner
from carmack.assign_targets.target_assigner import TargetAssigner
from carmack.assign_targets.tgidx_locator import locate_tgidx_window
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import ChemistryCarmackCustomSeq10
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.fastq_file import FastqFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_extractor import UmiExtractor

CHEMISTRY = "carmack_custom_seq_1_0"

# The constructor only wraps this path in a FastqFile, it never reads it.
DUMMY_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"

# Header vocabulary shared with the upstream extract-umis stage and with every
# downstream consumer, so it is pinned literally rather than derived from the
# chemistry: a rename here is a breaking change, not a chemistry detail.
TGIDX_NAME = "TGIDX"
UMI_POS_KEY = "UMI_POS"

# The seed length the matcher defaults to, read from its own signature so this
# file states the assigner takes that default rather than restating its value.
DEFAULT_MATCHER_K = signature(KmerMatcher.__init__).parameters["k"].default

# Layout constants for the synthetic chemistries below. They mirror the shipped
# custom_seq layout closely enough to stay realistic while varying only the one
# relationship each chemistry exists to break.
BC_LENGTH = 10
UMI_LENGTH = 8
UMI_LENGTH_TOLERANCE = 1
ANCHOR_BASE = "G"
ANCHOR_MIN_RUN = 3
TGIDX_LENGTH = 8
LINKER_SEQUENCE = "ACTGACTGAC"

# Names the synthetic chemistries are constructed under. Each class reports its
# own name as the same string, so a message naming either the requested name or
# the resolved chemistry satisfies the assertions.
EMPTY_WHITELIST_CHEMISTRY = "custom_seq_empty_target_whitelist"
LINKER_ANCHOR_CHEMISTRY = "custom_seq_linker_between_umi_and_anchor"
NO_RIGHT_ANCHOR_CHEMISTRY = "custom_seq_umi_without_right_anchor"
NO_UMI_CHEMISTRY = "custom_seq_without_umi"


def build_umi_component() -> ReadComponent:
    """Return a UMI component matching the shipped custom_seq layout."""
    return ReadComponent(
        name="UMI",
        type=ReadComponentType.UMI,
        length=UMI_LENGTH,
        length_tolerance=UMI_LENGTH_TOLERANCE,
    )


def build_read_structure(middle: list[ReadComponent]) -> ReadStructure:
    """Assemble a read structure of a barcode, a middle, the anchor and the index.

    The target index always sits immediately 3' of the homopolymer, so the
    chemistry still declares a locatable index and only the UMI's own right
    neighbour varies between the synthetic chemistries.

    Args:
        middle: Components sitting between the BC1 barcode and the anchor
            homopolymer run.

    Returns:
        A read structure carrying BC1, ``middle``, the POLYG anchor and TGIDX.
    """
    return ReadStructure(
        [
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=BC_LENGTH),
            *middle,
            ReadComponent(
                name="POLYG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base=ANCHOR_BASE,
                min_run=ANCHOR_MIN_RUN,
            ),
            ReadComponent(name=TGIDX_NAME, type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
        ]
    )


class ChemistryEmptyTargetWhitelist(ChemistryCarmackCustomSeq10):
    """Shipped chemistry whose target index whitelist loads no entries."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return EMPTY_WHITELIST_CHEMISTRY

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return no whitelist entries while keeping the shipped read structure."""
        return ()


class ChemistryLinkerAfterUmi(ChemistryCarmackCustomSeq10):
    """Chemistry whose UMI is followed by a linker, not the target index anchor."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return LINKER_ANCHOR_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout whose UMI is bounded by a primer instead of the anchor."""
        return build_read_structure(
            [
                build_umi_component(),
                ReadComponent(
                    name="LINKER",
                    type=ReadComponentType.PRIMER,
                    length=len(LINKER_SEQUENCE),
                    sequence=LINKER_SEQUENCE,
                ),
            ]
        )


class ChemistryUmiWithoutRightAnchor(ChemistryCarmackCustomSeq10):
    """Chemistry whose UMI is followed by a component that cannot anchor it."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return NO_RIGHT_ANCHOR_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout whose UMI is followed by a non-anchoring component."""
        return build_read_structure(
            [
                build_umi_component(),
                ReadComponent(
                    name="LINKER",
                    type=ReadComponentType.OTHER,
                    length=len(LINKER_SEQUENCE),
                    sequence=LINKER_SEQUENCE,
                ),
            ]
        )


class ChemistryWithoutUmi(ChemistryCarmackCustomSeq10):
    """Chemistry declaring a target index but no UMI to read the run start from."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return NO_UMI_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout carrying the anchor and index but no UMI component."""
        return build_read_structure([])


def patch_chemistry(chemistry: ChemistryBase):
    """Return a patcher making the factory answer with the supplied chemistry.

    Args:
        chemistry: The chemistry instance the assigner should resolve.

    Returns:
        An unstarted :func:`unittest.mock.patch` context manager.
    """
    return mock.patch(
        "carmack.assign_targets.target_assigner.ChemistryFactory.get_chemistry",
        return_value=chemistry,
    )


class TestTargetAssignerConstruction:
    """Parameter resolution and up-front validation in the constructor."""

    def test_wraps_the_input_path_and_resolves_the_chemistry(self) -> None:
        assigner = TargetAssigner(DUMMY_FASTQ, CHEMISTRY)

        assert_that(assigner.fastq).is_instance_of(FastqFile)
        assert_that(assigner.fastq.filename).is_equal_to(DUMMY_FASTQ)
        assert_that(assigner.chemistry_name).is_equal_to(CHEMISTRY)
        assert_that(assigner.chemistry.name).is_equal_to(CHEMISTRY)

    def test_resolves_parameters_from_chemistry(self) -> None:
        chemistry = ChemistryFactory.get_chemistry(CHEMISTRY)
        assigner = TargetAssigner(DUMMY_FASTQ, CHEMISTRY)

        assert_that(assigner.tgidx_name).is_equal_to(TGIDX_NAME)
        assert_that(assigner.umi_pos_key).is_equal_to(UMI_POS_KEY)
        assert_that(assigner.tgidx_length).is_equal_to(chemistry.tgidx_component().length)
        assert_that(assigner.anchor_base).is_equal_to(chemistry.tgidx_anchor().homopolymer_base)
        assert_that(assigner.max_errors).is_equal_to(chemistry.max_errors.tgidx)

    def test_builds_a_kmer_matcher_over_the_target_whitelist(self) -> None:
        assigner = TargetAssigner(DUMMY_FASTQ, CHEMISTRY)
        matcher = assigner.matcher

        assert_that(matcher).is_instance_of(KmerMatcher)
        assert_that(matcher.whitelist_set).is_equal_to(
            frozenset(assigner.chemistry.tgidx_whitelist())
        )
        assert_that(matcher.component.name).is_equal_to(TGIDX_NAME)
        assert_that(matcher.component.type).is_equal_to(ReadComponentType.TGIDX)
        assert_that(matcher.chemistry).is_same_as(assigner.chemistry)
        assert_that(matcher.max_errors).is_equal_to(assigner.max_errors)
        assert_that(matcher.k).is_equal_to(DEFAULT_MATCHER_K)
        assert_that(matcher.kmer_index).is_not_empty()

    def test_matcher_is_constructed_once_at_construction(self) -> None:
        """Pin that the k-mer index is built once, not once per read.

        The index is built inside the matcher's constructor, so a matcher built
        per read would rebuild it for every read in the FASTQ and dominate the
        stage's runtime.
        """
        chemistry = ChemistryFactory.get_chemistry(CHEMISTRY)

        with mock.patch(
            "carmack.assign_targets.target_assigner.KmerMatcher", autospec=True
        ) as matcher_class:
            assigner = TargetAssigner(DUMMY_FASTQ, CHEMISTRY)

        matcher_class.assert_called_once_with(
            whitelist=chemistry.tgidx_whitelist(),
            component=mock.ANY,
            chemistry=assigner.chemistry,
            max_errors=chemistry.max_errors.tgidx,
        )
        assert_that(assigner.matcher).is_same_as(matcher_class.return_value)

    def test_chemistry_without_target_index_raises(self) -> None:
        with pytest.raises(ValueError, match="hydrop"):
            TargetAssigner(DUMMY_FASTQ, "hydrop")

    def test_empty_target_whitelist_raises_though_support_check_passes(self) -> None:
        """Pin that the constructor, not the chemistry accessor, closes the empty-whitelist gap.

        ``ChemistryBase.supports_target_assignment`` asks only whether the target
        index is preceded by a homopolymer anchor, so a chemistry whose whitelist
        loads no entries still answers ``True``. No read could ever be assigned a
        target from it, so rejecting it falls to the constructor.
        """
        chemistry = ChemistryEmptyTargetWhitelist()

        assert_that(chemistry.supports_target_assignment()).is_true()
        assert_that(chemistry.tgidx_whitelist()).is_empty()

        with (
            patch_chemistry(chemistry),
            pytest.raises(ValueError, match=EMPTY_WHITELIST_CHEMISTRY),
        ):
            TargetAssigner(DUMMY_FASTQ, EMPTY_WHITELIST_CHEMISTRY)

    def test_umi_right_anchor_other_than_the_index_anchor_raises(self) -> None:
        """Pin that the UMI's right anchor must be the index's own anchor run.

        The anchor run's start is read from the end of the UMI span, which is
        only the run's start when the component following the UMI is that same
        homopolymer. Where they diverge every window would be cut off arbitrary
        sequence, so the mismatch has to be fatal at construction rather than
        silently mis-assigning reads.
        """
        chemistry = ChemistryLinkerAfterUmi()

        assert_that(chemistry.supports_target_assignment()).is_true()
        assert_that(chemistry.umi_right_anchor().name).is_not_equal_to(
            chemistry.tgidx_anchor().name
        )

        with (
            patch_chemistry(chemistry),
            pytest.raises(ValueError, match=LINKER_ANCHOR_CHEMISTRY),
        ):
            TargetAssigner(DUMMY_FASTQ, LINKER_ANCHOR_CHEMISTRY)

    @pytest.mark.parametrize(
        "chemistry_class,chemistry_name",
        [
            (ChemistryUmiWithoutRightAnchor, NO_RIGHT_ANCHOR_CHEMISTRY),
            (ChemistryWithoutUmi, NO_UMI_CHEMISTRY),
        ],
    )
    def test_chemistry_that_cannot_locate_the_anchor_run_raises(
        self, chemistry_class: type[ChemistryBase], chemistry_name: str
    ) -> None:
        chemistry = chemistry_class()

        with patch_chemistry(chemistry), pytest.raises(ValueError, match=chemistry_name):
            TargetAssigner(DUMMY_FASTQ, chemistry_name)

    def test_unknown_chemistry_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            TargetAssigner(DUMMY_FASTQ, "does_not_exist")


class TestTargetAssignerHeaderValidation:
    """The annotated header must carry the UMI position tag the stage reads."""

    @pytest.fixture
    def assigner(self) -> TargetAssigner:
        """Return an assigner built against the shipped custom_seq chemistry."""
        return TargetAssigner(DUMMY_FASTQ, CHEMISTRY)

    @staticmethod
    def make_annotation(tags: dict[str, str]) -> ReadAnnotation:
        """Return an annotation carrying the supplied tags."""
        ann = ReadAnnotation(read_id="read1")
        for key, value in tags.items():
            ann.set(key, value)
        return ann

    @pytest.mark.parametrize(
        "tags",
        [
            {},
            {"BC1": "ACTACTACTA", "UMI": "ACTACTAC"},
        ],
        ids=["no_tags", "other_tags_only"],
    )
    def test_missing_umi_position_tag_raises(
        self, assigner: TargetAssigner, tags: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            assigner.validate_header(self.make_annotation(tags))

        assert_that(str(excinfo.value)).contains(UMI_POS_KEY, "extract-umis")

    def test_present_umi_position_tag_returns_none(self, assigner: TargetAssigner) -> None:
        ann = self.make_annotation({UMI_POS_KEY: format_span(20, 28)})

        assert_that(assigner.validate_header(ann)).is_none()


# ---------------------------------------------------------------------------
# The streaming assignment pass
# ---------------------------------------------------------------------------

# Layout of the synthetic annotated reads below. The anchor run begins exactly
# where the UMI span ends, which is the one relationship the stage depends on.
UMI_NAME = "UMI"
UMI_SEQ = "ACTACTAC"
UMI_START = 20
RUN_LENGTH = 5
TAIL = "TTTTTTTTTT"

# The single entry the shipped custom_seq target whitelist loads.
TARGET_SEQ = "TATAGCCT"

# Observed index sequences planted in the read, each one error from TARGET_SEQ.
SUBSTITUTED_TARGET = "TATCGCCT"
DELETED_TARGET = "TATAGCT"

# Two errors from TARGET_SEQ, so outside the shipped budget of one.
TWO_ERROR_TARGET = "TATCGCCA"

# Single insertions. The first resolves to the whitelist entry, the second falls
# through to the sentinel: an inserted base can leave two equidistant candidates
# that no spacer can separate, so the outcome is a property of the sequence and
# not something a read carrying one insertion is entitled to.
INSERTED_TARGETS = ("TATTAGCCT", "TATAGGCCT")

# Transcript sequence standing in for a read that carries no target index at all,
# which is the correct and common answer in a mixed library. Chosen by driving the
# real matcher over the real whitelist, since a whitelist entry is falsely
# assigned to index-free sequence at a low but non-zero rate.
NO_INDEX_TAIL = "CAGCTGATCGGAAGAGCACA"

# Test-local whitelist covering an index that begins with zero, one and two anchor
# bases. Those leading bases are indistinguishable from the run at their boundary,
# so the window has to absorb the run end overshooting the index start.
LEADING_ANCHOR_TARGETS = ("TATAGCCT", "GACCTTAC", "GGCATTAC")
LEADING_ANCHOR_CHEMISTRY = "custom_seq_leading_anchor_targets"

# Test-local whitelist whose two entries differ by a single base, so an observed
# index one error from both ties between them.
TIED_TARGETS = ("TATAGCCT", "TATAGCCA")
TIED_INDEX = "TATAGCCG"
TIED_TARGET_CHEMISTRY = "custom_seq_tied_targets"

# Output naming. The input name is shaped like the file extract-umis writes, so
# the derived prefix is exercised against a realistic filename.
INPUT_FASTQ_NAME = "SK462.r1_umi.fastq.gz"
INPUT_PREFIX = "SK462"
OUT_PREFIX = "out"


class ChemistryLeadingAnchorTargets(ChemistryCarmackCustomSeq10):
    """Shipped chemistry whose target whitelist varies the leading anchor bases."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return LEADING_ANCHOR_CHEMISTRY

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return indices beginning with zero, one and two anchor bases."""
        return LEADING_ANCHOR_TARGETS


class ChemistryTiedTargets(ChemistryCarmackCustomSeq10):
    """Shipped chemistry whose two target whitelist entries differ by one base."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return TIED_TARGET_CHEMISTRY

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return two entries a single substitution apart."""
        return TIED_TARGETS


def make_annotated_read(
    read_id: str,
    index: str = TARGET_SEQ,
    run_length: int = RUN_LENGTH,
    tail: str = TAIL,
    umi: str = UMI_SEQ,
    umi_start: int = UMI_START,
    with_umi_pos: bool = True,
) -> tuple[str, str, str]:
    """Build a synthetic ``(header, seq, qual)`` UMI-annotated read.

    The sequence is ``A * umi_start`` + ``umi`` + the anchor run + ``index`` +
    ``tail``, so the anchor run begins exactly at ``umi_start + len(umi)`` and
    the header's UMI span ends on that same coordinate.

    Args:
        read_id: Identifier written as the header's first token.
        index: Target index sequence planted immediately after the anchor run.
            Empty for a read carrying no index at all.
        run_length: Number of anchor bases written before ``index``.
        tail: Sequence written after ``index``.
        umi: UMI sequence written immediately before the anchor run.
        umi_start: 0-based index at which the UMI begins.
        with_umi_pos: Whether to write the UMI position tag the stage reads.

    Returns:
        The rendered header, the read sequence and a matching quality string.
    """
    seq = "A" * umi_start + umi + ANCHOR_BASE * run_length + index + tail
    ann = ReadAnnotation(read_id=read_id)
    ann.set(UMI_NAME, umi)
    if with_umi_pos:
        ann.set(UMI_POS_KEY, format_span(umi_start, umi_start + len(umi)))
    return ann.render(), seq, "I" * len(seq)


def write_fastq(path: Path, records: list[tuple[str, str, str]]) -> None:
    """Write ``(header, seq, qual)`` records to a gzipped FASTQ file.

    Args:
        path: Destination path for the gzipped FASTQ.
        records: The records to write, in the order they should appear.
    """
    with gzip.open(path, "wt") as handle:
        for header, seq, qual in records:
            handle.write(f"@{header}\n{seq}\n+\n{qual}\n")


def read_fastq(path: Path) -> list[tuple[str, str, str, str]]:
    """Read a gzipped FASTQ file back into ``(header, seq, plus, qual)`` records.

    Args:
        path: Path of the gzipped FASTQ to read.

    Returns:
        One four-line tuple per read, in file order.
    """
    with gzip.open(path, "rt") as handle:
        lines = handle.read().splitlines()
    return [tuple(lines[i : i + 4]) for i in range(0, len(lines), 4)]


def read_annotations(path: Path) -> list[ReadAnnotation]:
    """Return the parsed header annotations of a gzipped FASTQ, in file order.

    Args:
        path: Path of the gzipped FASTQ to read.

    Returns:
        One :class:`ReadAnnotation` per read.
    """
    return [ReadAnnotation.parse(record[0][1:]) for record in read_fastq(path)]


def tgidx_fastq(directory: Path, prefix: str = OUT_PREFIX) -> Path:
    """Return the path of the assigned FASTQ the stage writes for ``prefix``."""
    return directory / f"{prefix}.r1_tgidx.fastq.gz"


def tgidx_stats(directory: Path, prefix: str = OUT_PREFIX) -> Path:
    """Return the path of the stats report the stage writes for ``prefix``."""
    return directory / f"{prefix}.tgidx_stats.txt"


@pytest.fixture
def build_assigner(tmp_path: Path) -> Callable[..., TargetAssigner]:
    """Return a factory that writes records to a FASTQ and builds an assigner.

    Args:
        tmp_path: Directory the synthetic FASTQ is written into.

    Returns:
        A callable taking the records, an optional filename and an optional
        chemistry instance the factory should be made to resolve, and returning
        the assigner constructed over the written file.
    """

    def build(
        records: list[tuple[str, str, str]],
        name: str = INPUT_FASTQ_NAME,
        chemistry: ChemistryBase | None = None,
    ) -> TargetAssigner:
        fastq_path = tmp_path / name
        write_fastq(fastq_path, records)
        if chemistry is None:
            return TargetAssigner(str(fastq_path), CHEMISTRY)
        with patch_chemistry(chemistry):
            return TargetAssigner(str(fastq_path), chemistry.name)

    return build


class TestAssignTargetsMatching:
    """Sensitivity of the window-and-match step over synthetic annotated reads."""

    def test_exact_index_after_anchor_run_is_matched(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        assigner = build_assigner([make_annotated_read("exact")])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.target_counts).is_equal_to({TARGET_SEQ: 1})
        assert_that(stats.edit_distance_counts).is_equal_to({0: 1})

        seq = read_fastq(tgidx_fastq(tmp_path))[0][1]
        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(TARGET_SEQ)
        start, end = parse_span(ann.get(position_key(TGIDX_NAME)))
        assert_that(seq[start:end]).is_equal_to(TARGET_SEQ)

    @pytest.mark.parametrize(
        "observed",
        [SUBSTITUTED_TARGET, DELETED_TARGET],
        ids=["substitution", "deletion"],
    )
    def test_one_error_index_is_matched_and_span_is_the_observed_slice(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path, observed: str
    ) -> None:
        """Pin that the recorded span bounds the read, not the entry it verified against.

        ``read_idx`` bounds the sequence found in the read while ``match`` is the
        whitelist entry it verified against, so the two coincide only at edit
        distance zero. Slicing the span back must therefore return the observed
        sequence.
        """
        assigner = build_assigner([make_annotated_read("oneerror", index=observed)])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.target_counts).is_equal_to({TARGET_SEQ: 1})
        assert_that(stats.edit_distance_counts).is_equal_to({1: 1})

        seq = read_fastq(tgidx_fastq(tmp_path))[0][1]
        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(TARGET_SEQ)
        start, end = parse_span(ann.get(position_key(TGIDX_NAME)))
        assert_that(seq[start:end]).is_equal_to(observed)

    def test_two_error_index_is_not_matched(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        assigner = build_assigner([make_annotated_read("twoerror", index=TWO_ERROR_TARGET)])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(0)
        assert_that(stats.unmatched_no_match).is_equal_to(1)

        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(ann.get(position_key(TGIDX_NAME))).is_none()

    @pytest.mark.parametrize("observed", INSERTED_TARGETS)
    def test_single_insertion_index_is_matched_or_unassigned(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path, observed: str
    ) -> None:
        """Pin the read is conserved whichever way a single insertion resolves.

        An inserted base can leave two equidistant candidates that no spacer can
        separate, so an index carrying one insertion is not entitled to a match.
        What it is entitled to is being emitted exactly once carrying an answer.
        """
        assigner = build_assigner([make_annotated_read("insertion", index=observed)])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(1)
        assert_that(stats.matched + stats.unmatched_no_match).is_equal_to(1)

        annotations = read_annotations(tgidx_fastq(tmp_path))
        assert_that(annotations).is_length(1)
        assert_that(annotations[0].get(TGIDX_NAME)).is_in(TARGET_SEQ, target_assigner.NO_TARGET)

    @pytest.mark.parametrize("index", LEADING_ANCHOR_TARGETS, ids=["lead0", "lead1", "lead2"])
    @pytest.mark.parametrize("run_length", range(3, 9))
    def test_index_beginning_with_anchor_bases_is_located(
        self,
        build_assigner: Callable[..., TargetAssigner],
        tmp_path: Path,
        index: str,
        run_length: int,
    ) -> None:
        """Pin that an index opening with anchor bases is still located exactly.

        The run end overshoots the index start by the number of leading anchor
        bases, and the observed run length varies independently, so the window's
        left margin has to absorb both without the index sliding along the run.
        """
        assigner = build_assigner(
            [make_annotated_read("lead", index=index, run_length=run_length)],
            chemistry=ChemistryLeadingAnchorTargets(),
        )

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.target_counts).is_equal_to({index: 1})

        seq = read_fastq(tgidx_fastq(tmp_path))[0][1]
        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(index)
        start, end = parse_span(ann.get(position_key(TGIDX_NAME)))
        assert_that(seq[start:end]).is_equal_to(index)

    def test_window_is_located_with_the_matchers_own_seed_length(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin that the seed length is handed to the locator rather than defaulted.

        ``TrimWindow.is_matchable`` only keeps a window the matcher would raise on
        away from the matcher while the locator's seed length and the matcher's
        agree. Both currently default to the same value, so leaving the argument
        out looks correct until one of them moves.
        """
        assigner = build_assigner([make_annotated_read("seed")])

        with mock.patch(
            "carmack.assign_targets.target_assigner.locate_tgidx_window",
            wraps=locate_tgidx_window,
        ) as locator:
            assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(locator.call_count).is_equal_to(1)
        bound = signature(locate_tgidx_window).bind(
            *locator.call_args.args, **locator.call_args.kwargs
        )
        assert_that(bound.arguments).contains_key("k")
        assert_that(bound.arguments["k"]).is_equal_to(assigner.matcher.k)


class TestAssignTargetsOutcomes:
    """The four mutually exclusive per-read outcomes and the counters they feed."""

    def test_read_without_an_index_is_unassigned_and_carries_no_span(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [make_annotated_read("noindex", index="", tail=NO_INDEX_TAIL)]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(0)
        assert_that(stats.unmatched_no_match).is_equal_to(1)
        assert_that(stats.target_counts).is_empty()

        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(ann.get(position_key(TGIDX_NAME))).is_none()

    def test_tied_candidates_are_unassigned_and_counted_as_no_match(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin that a tie between whitelist entries reaches the stage as no match.

        The target index is the last component and its 5' neighbour is a
        homopolymer carrying no sequence, so both spacers are always absent and
        the matcher has no rung left to break a tie on. Every tie therefore falls
        through to the same untouched attempt a no-candidate search returns.
        """
        assigner = build_assigner(
            [make_annotated_read("tie", index=TIED_INDEX)],
            chemistry=ChemistryTiedTargets(),
        )

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(0)
        assert_that(stats.unmatched_no_match).is_equal_to(1)

        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)

    def test_window_below_the_matcher_floor_never_reaches_the_matcher(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin that a short window is counted rather than handed to the matcher.

        ``KmerMatcher.collect_candidates`` raises on an input shorter than its
        seed length instead of reporting a miss, so the floor check has to stand
        between the two.
        """
        records = [make_annotated_read("short", index="", tail="")]
        assigner = build_assigner(records)

        with mock.patch.object(assigner.matcher, "match", autospec=True) as matcher_match:
            stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(matcher_match.called).is_false()
        assert_that(stats.unmatched_short_window).is_equal_to(1)
        assert_that(stats.matched).is_equal_to(0)

        ann = read_annotations(tgidx_fastq(tmp_path))[0]
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(ann.get(position_key(TGIDX_NAME))).is_none()

    def test_mid_stream_read_without_umi_position_is_counted_and_emitted(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin that only the first read's header is fatal, and the rest are counted.

        Validation is a fail-fast check on the chemistry the FASTQ was produced
        with, not a per-read filter: a later read missing the tag is annotated
        unassigned so that reads written still reconciles with reads read.
        """
        records = [
            make_annotated_read("first"),
            make_annotated_read("second", with_umi_pos=False),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(2)
        assert_that(stats.unmatched_no_umi_pos).is_equal_to(1)
        assert_that(stats.matched).is_equal_to(1)

        annotations = read_annotations(tgidx_fastq(tmp_path))
        assert_that([ann.read_id for ann in annotations]).is_equal_to(["first", "second"])
        assert_that(annotations[1].get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(annotations[1].get(position_key(TGIDX_NAME))).is_none()


class TestAssignTargetsOutputs:
    """Conservation of reads, the reconciling counts and the files written."""

    def test_every_read_is_emitted_once_in_order_with_sequence_and_quality_intact(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("first"),
            make_annotated_read("second", index=SUBSTITUTED_TARGET, run_length=3),
            make_annotated_read("third", index="", tail=NO_INDEX_TAIL),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        written = read_fastq(tgidx_fastq(tmp_path))
        assert_that(written).is_length(len(records))
        assert_that(stats.total_reads).is_equal_to(len(records))

        ids = [ReadAnnotation.parse(record[0][1:]).read_id for record in written]
        assert_that(ids).is_equal_to(["first", "second", "third"])
        for record, out in zip(records, written):
            assert_that(out[1]).is_equal_to(record[1])
            assert_that(out[3]).is_equal_to(record[2])

    def test_outcome_counts_reconcile_with_reads_read_and_written(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("matched"),
            make_annotated_read("noumipos", with_umi_pos=False),
            make_annotated_read("nomatch", index="", tail=NO_INDEX_TAIL),
            make_annotated_read("shortwindow", index="", tail=""),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.unmatched_no_match).is_equal_to(1)
        assert_that(stats.unmatched_no_umi_pos).is_equal_to(1)
        assert_that(stats.unmatched_short_window).is_equal_to(1)
        assert_that(
            stats.matched
            + stats.unmatched_no_match
            + stats.unmatched_no_umi_pos
            + stats.unmatched_short_window
        ).is_equal_to(stats.total_reads)
        assert_that(read_fastq(tgidx_fastq(tmp_path))).is_length(stats.total_reads)

    def test_stats_report_is_written_and_reflects_the_counts(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("matched"),
            make_annotated_read("noumipos", with_umi_pos=False),
            make_annotated_read("nomatch", index="", tail=NO_INDEX_TAIL),
            make_annotated_read("shortwindow", index="", tail=""),
        ]
        assigner = build_assigner(records)

        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        report = tgidx_stats(tmp_path).read_text()
        assert_that(report).contains("# Target Index Assignment Stats")
        assert_that(report).contains("Total reads: 4")
        assert_that(report).contains("Matched: 1")
        assert_that(report).contains("Unmatched (no_match): 1")
        assert_that(report).contains("Unmatched (no_umi_pos): 1")
        assert_that(report).contains("Unmatched (short_window): 1")
        assert_that(report).contains(f"\t{TARGET_SEQ}\t1")

    def test_prefix_defaults_to_the_input_filename(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        assigner = build_assigner([make_annotated_read("a")], name=INPUT_FASTQ_NAME)

        assigner.assign_targets(output_dir=str(tmp_path))

        assert_that(tgidx_fastq(tmp_path, INPUT_PREFIX).exists()).is_true()
        assert_that(tgidx_stats(tmp_path, INPUT_PREFIX).exists()).is_true()

    def test_empty_input_writes_an_empty_fastq_and_a_zero_count_report(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin the degenerate case of the conservation guarantee: none in, none out.

        An empty FASTQ offers no first read to validate against, so the header
        check is skipped rather than raising, and both output files are still
        written: a run over no reads is an empty run, not a failed one.
        """
        assigner = build_assigner([])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(0)
        assert_that(stats.matched).is_equal_to(0)
        assert_that(stats.reads_with_measured_run).is_equal_to(0)
        assert_that(read_fastq(tgidx_fastq(tmp_path))).is_empty()
        assert_that(tgidx_stats(tmp_path).read_text()).contains("Total reads: 0")

    def test_first_read_without_umi_position_raises_before_any_output_is_written(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("first", with_umi_pos=False),
            make_annotated_read("second"),
        ]
        assigner = build_assigner(records)

        with pytest.raises(ValueError, match=UMI_POS_KEY):
            assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(tgidx_fastq(tmp_path).exists()).is_false()
        assert_that(tgidx_stats(tmp_path).exists()).is_false()

    def test_observed_anchor_run_lengths_are_recorded(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("run3", run_length=3),
            make_annotated_read("run6a", run_length=6),
            make_annotated_read("run6b", run_length=6),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.homopolymer_base).is_equal_to(ANCHOR_BASE)
        assert_that(stats.homopolymer_run_counts).is_equal_to({3: 1, 6: 2})
        assert_that(stats.reads_with_measured_run).is_equal_to(3)


# ---------------------------------------------------------------------------
# CLI wiring and end-to-end composition
# ---------------------------------------------------------------------------

# The name the stage is exposed under and the metavar its one positional
# argument renders as. Both are user-facing vocabulary rather than internals, so
# they are pinned literally: renaming either is a breaking change to anyone with
# a pipeline script or a documented command line.
COMMAND_NAME = "assign-targets"
FASTQ_METAVAR = "<r1_umi_fastq>"

# The help group the command belongs to and the two commands it sits between.
# The stage runs after extract-umis and before fastq-filter, and the grouped
# help listing is the only place a user reads that order off, so the position is
# part of the contract and not a cosmetic detail.
USER_COMMAND_GROUP = "Commands for users"
PRECEDING_COMMAND = "extract-umis"
FOLLOWING_COMMAND = "fastq-filter"

# The committed 200-read golden input and the prefix every stage of the
# end-to-end run writes under. It is the smallest input carrying the full
# custom_seq layout, so it is the cheapest real fixture that can be driven
# through barcode extraction, UMI extraction and target assignment in turn.
GOLDEN_INPUT_DIR = Path(__file__).parent / "data" / "golden"
GOLDEN_INPUT_NAME = "custom_seq_1_0_small_R1.fastq.gz"
GOLDEN_PREFIX = "custom_seq_1_0_small"


def user_command_names() -> list[str]:
    """Return the commands listed in the user-facing help group, in help order.

    Returns:
        The command names of the ``Commands for users`` group, in the order the
        grouped help listing renders them.
    """
    groups = click.rich_click.COMMAND_GROUPS["carmack"]
    return next(group["commands"] for group in groups if group["name"] == USER_COMMAND_GROUP)


class TestAssignTargetsCli:
    """The assign-targets command's wiring into the carmack command line."""

    @pytest.mark.parametrize(
        "prefix_args, expected_prefix",
        [
            pytest.param([], None, id="prefix-defaults-to-none"),
            pytest.param(["--prefix", OUT_PREFIX], OUT_PREFIX, id="prefix-passed-through"),
        ],
    )
    def test_command_constructs_the_assigner_and_assigns_targets(
        self, tmp_path: Path, prefix_args: list[str], expected_prefix: str | None
    ) -> None:
        runner = CliRunner()
        with mock.patch("carmack.__main__.TargetAssigner", autospec=True) as mock_assigner:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [
                    COMMAND_NAME,
                    DUMMY_FASTQ,
                    "--chemistry",
                    CHEMISTRY,
                    "--output_dir",
                    str(tmp_path),
                    *prefix_args,
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_assigner.assert_called_once_with(DUMMY_FASTQ, CHEMISTRY)
        mock_assigner.return_value.assign_targets.assert_called_once_with(
            str(tmp_path), expected_prefix
        )

    def test_command_is_listed_in_the_top_level_help(self) -> None:
        runner = CliRunner()

        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains(COMMAND_NAME)

    def test_command_group_places_it_between_extract_umis_and_fastq_filter(self) -> None:
        commands = user_command_names()

        assert_that(commands).contains(COMMAND_NAME)
        assert_that(commands.index(COMMAND_NAME)).is_equal_to(
            commands.index(PRECEDING_COMMAND) + 1
        )
        assert_that(commands.index(FOLLOWING_COMMAND)).is_equal_to(
            commands.index(COMMAND_NAME) + 1
        )

    def test_chemistry_option_is_required(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with mock.patch("carmack.__main__.TargetAssigner", autospec=True) as mock_assigner:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [COMMAND_NAME, DUMMY_FASTQ, "--output_dir", str(tmp_path)],
            )

        assert_that(result.exit_code).is_not_equal_to(0)
        assert_that(result.output).contains("--chemistry")
        mock_assigner.assert_not_called()

    def test_command_help_documents_its_argument_and_chemistry_option(self) -> None:
        runner = CliRunner()

        result = runner.invoke(carmack.__main__.carmack_cli, [COMMAND_NAME, "--help"])

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains(COMMAND_NAME)
        assert_that(result.output).contains(FASTQ_METAVAR)
        assert_that(result.output).contains("--chemistry")


class TestAssignTargetsEndToEnd:
    """The three real stages composed over the committed 200-read golden input."""

    def test_stages_compose_and_every_umi_read_reaches_the_assigned_fastq(
        self, tmp_path: Path
    ) -> None:
        """Chain barcode extraction, UMI extraction and target assignment.

        The barcode step runs with ``n_workers=1`` and ``fast=True``. This test
        asserts pipeline composition, not matcher sensitivity: the local
        alignment tier that ``fast`` skips is what makes that stage slow, and
        dropping it changes only how many reads survive barcode extraction,
        never whether the three stages hand their files to one another
        correctly.

        Args:
            tmp_path: Directory all three stages write their outputs into.
        """
        input_fastq = GOLDEN_INPUT_DIR / GOLDEN_INPUT_NAME
        extractor = BarcodeExtractor(str(input_fastq), CHEMISTRY, n_workers=1, fast=True)
        extractor.extract_barcodes(str(tmp_path), GOLDEN_PREFIX)

        annotated_fastq = tmp_path / f"{GOLDEN_PREFIX}.r1_annotated.fastq.gz"
        UmiExtractor(str(annotated_fastq), CHEMISTRY).extract_umis(str(tmp_path), GOLDEN_PREFIX)

        umi_fastq = tmp_path / f"{GOLDEN_PREFIX}.r1_umi.fastq.gz"
        assigner = TargetAssigner(str(umi_fastq), CHEMISTRY)
        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=GOLDEN_PREFIX)

        assigned_fastq = tgidx_fastq(tmp_path, GOLDEN_PREFIX)
        assert_that(assigned_fastq.is_file()).is_true()
        assert_that(tgidx_stats(tmp_path, GOLDEN_PREFIX).is_file()).is_true()

        emitted = read_fastq(umi_fastq)
        written = read_fastq(assigned_fastq)

        assert_that(written).is_not_empty()
        assert_that(stats.total_reads).is_equal_to(len(emitted))
        assert_that(written).is_length(stats.total_reads)
        assert_that(
            stats.matched
            + stats.unmatched_no_match
            + stats.unmatched_no_umi_pos
            + stats.unmatched_short_window
        ).is_equal_to(stats.total_reads)

        source_ids = [ReadAnnotation.parse(record[0][1:]).read_id for record in emitted]
        written_ids = [ReadAnnotation.parse(record[0][1:]).read_id for record in written]
        assert_that(written_ids).is_equal_to(source_ids)
        for source, out in zip(emitted, written):
            assert_that(out[1]).is_equal_to(source[1])
            assert_that(out[3]).is_equal_to(source[3])

        whitelist = ChemistryFactory.get_chemistry(CHEMISTRY).tgidx_whitelist()
        allowed = set(whitelist) | {target_assigner.NO_TARGET}
        assigned = [ann.get(TGIDX_NAME) for ann in read_annotations(assigned_fastq)]
        assert_that(set(assigned)).is_subset_of(allowed)
        assert_that(assigned).does_not_contain(None)
