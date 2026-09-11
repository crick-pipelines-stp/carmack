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

Then it covers the two pieces that pass is built from: the per-read body, which
assigns one read and tallies its outcome into the accumulator it is handed, and
the batcher that lazily groups a caller's read iterator into the batches a pool
consumes.

Next it covers putting that pass on a pool: the once-per-process handover of the
assigner, the worker one batch is handed to, and the order the stage's with-block
creates and tears down its pool, its progress bar and its output stream in. Those
orderings are then exercised over a real forked pool, whose output has to be the
same bytes at every worker count and on every run.

Finally it covers the stage's command line wiring - the arguments the
``assign-targets`` command forwards to the assigner and the command's place in
the grouped help listing - and one end-to-end run chaining barcode extraction,
UMI extraction and target assignment over a committed golden input.
"""

import gzip
import importlib.util
import json
import multiprocessing
import os
import pickle
import re
import signal
from collections.abc import Callable, Iterable, Iterator
from dataclasses import fields
from functools import cached_property
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import NamedTuple
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
from carmack.assign_targets.assign_reporting import AssignCounts, AssignStats
from carmack.assign_targets.target_assigner import (
    DEFAULT_MAX_WORKERS,
    MAX_READS_PER_BATCH,
    TargetAssigner,
    assign_read_batch,
    init_assign_worker,
    iter_read_batches,
)
from carmack.assign_targets.tgidx_locator import locate_tgidx_window
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.barcode.barcode_utils import edit_distance
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import ChemistryCarmackCustomSeq10
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.io.subprocess_stream import SubprocessStream
from carmack.parallel import map_batches_in_order
from carmack.umi.umi_extractor import UmiExtractor
from carmack.utils import get_cpu_count
from tests.utils import read_gzip_text, strip_report_run_details

CHEMISTRY = "carmack_custom_seq_1_0"

# The constructor only wraps this path in a FastqFile, it never reads it.
DUMMY_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"

# Header vocabulary shared with the upstream stages and with every downstream
# consumer, so it is pinned literally rather than derived from the chemistry: a
# rename here is a breaking change, not a chemistry detail. The stage reads the
# barcode position tag, which barcode extraction writes, and derives the anchor
# run start from it; it does not read anything extract-umis writes.
TGIDX_NAME = "TGIDX"
ANCHOR_POS_KEY = "BC1_POS"

# Bases the chemistry places between the anchor's end and the run's start, which
# for the shipped layout is the fixed-length UMI.
RUN_OFFSET = 8

# The seed length the matcher defaults to, read from its own signature so this
# file states the assigner takes that default rather than restating its value.
DEFAULT_MATCHER_K = signature(KmerMatcher.__init__).parameters["k"].default

# Layout constants for the synthetic chemistries below. They mirror the shipped
# custom_seq layout closely enough to stay realistic while varying only the one
# relationship each chemistry exists to break.
BC_LENGTH = 10
UMI_LENGTH = 8
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
VARIABLE_BEFORE_ANCHOR_CHEMISTRY = "custom_seq_variable_before_anchor"
UNRECORDED_ANCHOR_CHEMISTRY = "custom_seq_unrecorded_anchor_only"


def build_umi_component() -> ReadComponent:
    """Return a UMI component matching the shipped custom_seq layout."""
    return ReadComponent(
        name="UMI",
        type=ReadComponentType.UMI,
        length=UMI_LENGTH,
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


class ChemistryVariableBeforeAnchor(ChemistryCarmackCustomSeq10):
    """Chemistry with a variable-length component between the barcode and the anchor."""

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return VARIABLE_BEFORE_ANCHOR_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout carrying a second homopolymer ahead of the index anchor."""
        return build_read_structure(
            [
                ReadComponent(
                    name="POLYA",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base="A",
                    min_run=ANCHOR_MIN_RUN,
                ),
            ]
        )


class ChemistryUnrecordedAnchorOnly(ChemistryCarmackCustomSeq10):
    """Chemistry whose only component 5' of the anchor never has its position written.

    A primer anchors a neighbour perfectly well, but barcode extraction writes a
    position tag for barcodes alone, so there is no span to measure from.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return UNRECORDED_ANCHOR_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout whose leading component is a primer rather than a barcode."""
        return ReadStructure(
            [
                ReadComponent(
                    name="PRIMER_A",
                    type=ReadComponentType.PRIMER,
                    length=len(LINKER_SEQUENCE),
                    sequence=LINKER_SEQUENCE,
                ),
                ReadComponent(
                    name="POLYG",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base=ANCHOR_BASE,
                    min_run=ANCHOR_MIN_RUN,
                ),
                ReadComponent(name=TGIDX_NAME, type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
            ]
        )


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
        assert_that(assigner.anchor_pos_key).is_equal_to(ANCHOR_POS_KEY)
        assert_that(assigner.run_offset).is_equal_to(RUN_OFFSET)
        assert_that(assigner.anchor_min_run).is_equal_to(chemistry.tgidx_anchor().min_run)
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

    @pytest.mark.parametrize(
        "chemistry_class,chemistry_name,expected_offset",
        [
            (ChemistryLinkerAfterUmi, LINKER_ANCHOR_CHEMISTRY, UMI_LENGTH + len(LINKER_SEQUENCE)),
            (
                ChemistryUmiWithoutRightAnchor,
                NO_RIGHT_ANCHOR_CHEMISTRY,
                UMI_LENGTH + len(LINKER_SEQUENCE),
            ),
            (ChemistryWithoutUmi, NO_UMI_CHEMISTRY, 0),
        ],
    )
    def test_any_fixed_distance_from_the_anchor_is_supported(
        self, chemistry_class: type[ChemistryBase], chemistry_name: str, expected_offset: int
    ) -> None:
        """Layouts the old coordinate scheme rejected now work, and give the right offset.

        All three used to be fatal, because the run start was read off the end of
        the UMI span and so had to have the anchor run as the UMI's immediate
        neighbour. Deriving the coordinate from the chemistry instead means the
        UMI is no longer special: what matters is only that the distance between
        the recorded anchor and the run is fixed. A linker in between is simply
        crossed, whatever its type, and a chemistry with no UMI at all puts the
        run directly on the barcode.
        """
        chemistry = chemistry_class()

        with patch_chemistry(chemistry):
            assigner = TargetAssigner(DUMMY_FASTQ, chemistry_name)

        assert_that(assigner.anchor_pos_key).is_equal_to(ANCHOR_POS_KEY)
        assert_that(assigner.run_offset).is_equal_to(expected_offset)

    @pytest.mark.parametrize(
        "chemistry_class,chemistry_name",
        [
            (ChemistryVariableBeforeAnchor, VARIABLE_BEFORE_ANCHOR_CHEMISTRY),
            (ChemistryUnrecordedAnchorOnly, UNRECORDED_ANCHOR_CHEMISTRY),
        ],
    )
    def test_chemistry_that_cannot_locate_the_anchor_run_raises(
        self, chemistry_class: type[ChemistryBase], chemistry_name: str
    ) -> None:
        """A run whose distance from a recorded anchor is not fixed is fatal.

        These are the two ways the derivation can genuinely fail: a
        variable-length component in between, so there is no fixed distance to
        add; and nothing to the left whose position was ever written down, so
        there is no coordinate to add it to. Either would have every window cut
        off a coordinate belonging to some other component, with no downstream
        symptom to catch it by.
        """
        chemistry = chemistry_class()

        with patch_chemistry(chemistry), pytest.raises(ValueError, match=chemistry_name):
            TargetAssigner(DUMMY_FASTQ, chemistry_name)

    def test_unknown_chemistry_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            TargetAssigner(DUMMY_FASTQ, "does_not_exist")


class TestTargetAssignerHeaderValidation:
    """The annotated header must carry the barcode position tag the stage reads."""

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
    def test_missing_anchor_position_tag_raises(
        self, assigner: TargetAssigner, tags: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            assigner.validate_header(self.make_annotation(tags))

        assert_that(str(excinfo.value)).contains(ANCHOR_POS_KEY, "extract-barcodes")

    def test_present_anchor_position_tag_returns_none(self, assigner: TargetAssigner) -> None:
        ann = self.make_annotation({ANCHOR_POS_KEY: format_span(20, 28)})

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
    with_anchor_pos: bool = True,
) -> tuple[str, str, str]:
    """Build a synthetic ``(header, seq, qual)`` UMI-annotated read.

    The sequence is ``A * umi_start`` + ``umi`` + the anchor run + ``index`` +
    ``tail``, so the anchor run begins exactly at ``umi_start + len(umi)``. The
    header carries the barcode span ending at ``umi_start``, from which the stage
    recomputes that same coordinate by adding the chemistry's fixed offset.

    Args:
        read_id: Identifier written as the header's first token.
        index: Target index sequence planted immediately after the anchor run.
            Empty for a read carrying no index at all.
        run_length: Number of anchor bases written before ``index``.
        tail: Sequence written after ``index``.
        umi: UMI sequence written immediately before the anchor run.
        umi_start: 0-based index at which the UMI begins.
        with_anchor_pos: Whether to write the barcode position tag the stage reads.

    Returns:
        The rendered header, the read sequence and a matching quality string.
    """
    seq = "A" * umi_start + umi + ANCHOR_BASE * run_length + index + tail
    ann = ReadAnnotation(read_id=read_id)
    ann.set(UMI_NAME, umi)
    if with_anchor_pos:
        ann.set(ANCHOR_POS_KEY, format_span(umi_start - BC_LENGTH, umi_start))
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


def tgidx_stats_mqc(directory: Path, prefix: str = OUT_PREFIX) -> Path:
    """Return the path of the MultiQC stats payload the stage writes for ``prefix``."""
    return directory / f"{prefix}.tgidx_stats_mqc.json"


def tgidx_edit_distance_mqc(directory: Path, prefix: str = OUT_PREFIX) -> Path:
    """Return the path of the MultiQC edit-distance payload the stage writes for ``prefix``."""
    return directory / f"{prefix}.tgidx_edit_distance_mqc.json"


def tgidx_anchor_run_mqc(directory: Path, prefix: str = OUT_PREFIX) -> Path:
    """Return the path of the MultiQC anchor-run payload the stage writes for ``prefix``."""
    return directory / f"{prefix}.tgidx_anchor_run_mqc.json"


@pytest.fixture
def build_assigner(tmp_path: Path) -> Callable[..., TargetAssigner]:
    """Return a factory that writes records to a FASTQ and builds an assigner.

    Args:
        tmp_path: Directory the synthetic FASTQ is written into.

    Returns:
        A callable taking the records, an optional filename, an optional
        chemistry instance the factory should be made to resolve and any further
        constructor arguments, and returning the assigner constructed over the
        written file.
    """

    def build(
        records: list[tuple[str, str, str]],
        name: str = INPUT_FASTQ_NAME,
        chemistry: ChemistryBase | None = None,
        **assigner_kwargs: int | None,
    ) -> TargetAssigner:
        fastq_path = tmp_path / name
        write_fastq(fastq_path, records)
        if chemistry is None:
            return TargetAssigner(str(fastq_path), CHEMISTRY, **assigner_kwargs)
        with patch_chemistry(chemistry):
            return TargetAssigner(str(fastq_path), chemistry.name, **assigner_kwargs)

    return build


# Behavioural whole-stage tests run over the recording pool rather than a real
# one. The harness substitutes only the fork: it wraps and calls through to the
# real driver, runs the real per-batch worker in this process and opens a real
# compressor, so batching, assignment, ordering and gzip output are all still
# exercised. A real pool here would buy no coverage and would cost the suite its
# only warning: a teardown-order regression deadlocks an unguarded run instead of
# failing it, hanging the session before it ever reaches the guarded real-pool
# classes that exist to catch exactly that.
substitute_pool = pytest.mark.usefixtures("pool_harness")


@substitute_pool
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
    def test_one_error_index_is_matched_and_span_is_the_entrys_own_length(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path, observed: str
    ) -> None:
        """Pin that the recorded span bounds the read, at the entry's own length.

        ``read_idx`` bounds the sequence found in the read while ``match`` is the whitelist
        entry it verified against, so the two coincide only at edit distance zero; slicing the
        span back returns the read, not the entry.

        Among the windows that tie at the minimum edit distance, the one reported is the window
        whose length equals the entry's. For a substitution that is simply the observed
        sequence. For a deletion it is one base longer than what was planted, and that is
        deliberate rather than sloppy: a read carrying a deletion and a read carrying a
        substitution at the entry's last base are *byte-identical* over this window -- both read
        ``TATAGCTT`` here -- so no rule can tell them apart from the sequence alone, and both
        the 7bp and the 8bp window sit at edit distance one. Preferring the entry's length is
        what stops a component's reported 3' boundary drifting inwards, which is what sent the
        adjacent-spacer check a base early and shifted the UMI off the end of the barcode
        before it. The cost is this one borrowed base on a genuine deletion.

        Nothing downstream measures off the target index -- it is the last component, and the
        poly-G anchor is taken from the UMI span -- so the borrowed base is reported and not
        acted upon.
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

        assert_that(end - start).is_equal_to(len(TARGET_SEQ))
        assert_that(seq[start:end]).starts_with(observed[: len(TARGET_SEQ) - 1])
        assert_that(edit_distance(seq[start:end], TARGET_SEQ)).is_equal_to(1)

    def test_run_starting_one_base_early_gives_the_same_target_and_span(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """A run beginning one base before the layout predicts is indistinguishable.

        This is the common case -- roughly a fifth of a real library, where the
        UMI's last base happens to be the anchor base, so the run genuinely spans
        one base more than the layout says. The extra base sits before the derived
        start, and counting forward from there reaches the same run end, so the
        window, the target, its span and even the measured run length all match
        the exact read's exactly. That equivalence is what lets the coordinate be
        asserted from the chemistry rather than found in the read.
        """
        early_umi = UMI_SEQ[:-1] + ANCHOR_BASE
        assigner = build_assigner(
            [make_annotated_read("early", umi=early_umi), make_annotated_read("exact")]
        )

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(2)
        assert_that(stats.target_counts).is_equal_to({TARGET_SEQ: 2})
        spans = {
            ann.get(position_key(TGIDX_NAME)) for ann in read_annotations(tgidx_fastq(tmp_path))
        }
        assert_that(spans).is_length(1)
        assert_that(stats.homopolymer_run_counts).is_equal_to({RUN_LENGTH: 2})

    def test_run_starting_late_is_recovered_by_the_forward_search(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """A read carrying an insertion before the run is still matched.

        This is the case the derived coordinate cannot absorb on its own: the
        count stops immediately and the window would be cut short of the index.
        The bounded forward search is what recovers it, and it is the tolerance
        the extractor's old window search used to provide.
        """
        late_umi = UMI_SEQ + "C"
        assigner = build_assigner([make_annotated_read("late", umi=late_umi)])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.target_counts).is_equal_to({TARGET_SEQ: 1})
        assert_that(stats.homopolymer_run_counts).is_equal_to({RUN_LENGTH: 1})

    def test_runs_without_extract_umis_having_written_anything(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """The stage needs only the barcode position tag, not the UMI tag.

        Deriving the run start from the chemistry removed the last thing this
        stage read from extract-umis. The canonical pipeline order still puts UMI
        extraction first, because that is what carries the UMI onto the read, but
        it is no longer a coordinate dependency.
        """
        header, seq, qual = make_annotated_read("nobcumi")
        ann = ReadAnnotation.parse(header)
        ann.tags.pop(UMI_NAME, None)
        assigner = build_assigner([(ann.render(), seq, qual)])

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.target_counts).is_equal_to({TARGET_SEQ: 1})

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
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """Pin that the seed length is handed to the locator rather than defaulted.

        ``TrimWindow.is_matchable`` only keeps a window the matcher would raise on
        away from the matcher while the locator's seed length and the matcher's
        agree. Both currently default to the same value, so leaving the argument
        out looks correct until one of them moves.

        Observed through the per-read body rather than a whole-stage run: the
        stage hands its reads to worker processes, so a locator patched here
        would be called in a child and this process would record nothing.
        """
        name, seq, _ = make_annotated_read("seed")
        assigner = build_assigner([make_annotated_read("seed")])

        with mock.patch(
            "carmack.assign_targets.target_assigner.locate_tgidx_window",
            wraps=locate_tgidx_window,
        ) as locator:
            assigner.assign_read(name, seq, AssignCounts())

        assert_that(locator.call_count).is_equal_to(1)
        bound = signature(locate_tgidx_window).bind(
            *locator.call_args.args, **locator.call_args.kwargs
        )
        assert_that(bound.arguments).contains_key("k")
        assert_that(bound.arguments["k"]).is_equal_to(assigner.matcher.k)


@substitute_pool
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
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """Pin that a short window is counted rather than handed to the matcher.

        ``KmerMatcher.collect_candidates`` raises on an input shorter than its
        seed length instead of reporting a miss, so the floor check has to stand
        between the two.

        Observed through the per-read body rather than a whole-stage run: the
        stage matches its reads in worker processes, so a matcher patched here
        would be called on a copy of it in a child and this process would record
        nothing - leaving an assertion that the matcher was never called true
        whether the floor check stands or not.
        """
        name, seq, _ = make_annotated_read("short", index="", tail="")
        assigner = build_assigner([make_annotated_read("short", index="", tail="")])
        counts = AssignCounts()

        with mock.patch.object(assigner.matcher, "match", autospec=True) as matcher_match:
            header = assigner.assign_read(name, seq, counts)

        assert_that(matcher_match.called).is_false()
        assert_that(counts.short_window).is_equal_to(1)
        assert_that(counts.matched).is_equal_to(0)

        ann = ReadAnnotation.parse(header)
        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(ann.get(position_key(TGIDX_NAME))).is_none()

    def test_mid_stream_read_without_anchor_position_is_counted_and_emitted(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin that only the first read's header is fatal, and the rest are counted.

        Validation is a fail-fast check on the chemistry the FASTQ was produced
        with, not a per-read filter: a later read missing the tag is annotated
        unassigned so that reads written still reconciles with reads read.
        """
        records = [
            make_annotated_read("first"),
            make_annotated_read("second", with_anchor_pos=False),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(2)
        assert_that(stats.unmatched_no_left_anchor_pos).is_equal_to(1)
        assert_that(stats.matched).is_equal_to(1)

        annotations = read_annotations(tgidx_fastq(tmp_path))
        assert_that([ann.read_id for ann in annotations]).is_equal_to(["first", "second"])
        assert_that(annotations[1].get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(annotations[1].get(position_key(TGIDX_NAME))).is_none()


@substitute_pool
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
            make_annotated_read("noumipos", with_anchor_pos=False),
            make_annotated_read("nomatch", index="", tail=NO_INDEX_TAIL),
            make_annotated_read("shortwindow", index="", tail=""),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_equal_to(1)
        assert_that(stats.unmatched_no_match).is_equal_to(1)
        assert_that(stats.unmatched_no_left_anchor_pos).is_equal_to(1)
        assert_that(stats.unmatched_short_window).is_equal_to(1)
        assert_that(
            stats.matched
            + stats.unmatched_no_match
            + stats.unmatched_no_left_anchor_pos
            + stats.unmatched_short_window
        ).is_equal_to(stats.total_reads)
        assert_that(read_fastq(tgidx_fastq(tmp_path))).is_length(stats.total_reads)

    def test_stats_report_is_written_and_reflects_the_counts(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("matched"),
            make_annotated_read("noumipos", with_anchor_pos=False),
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
        assert_that(report).contains("Unmatched (no_left_anchor_pos): 1")
        assert_that(report).contains("Unmatched (short_window): 1")
        assert_that(report).contains(f"\t{TARGET_SEQ}\t1")

    def test_mqc_stats_json_is_written_and_parses_with_the_expected_top_level_keys(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """A real assignment with at least one matched read produces a parseable
        ``{prefix}.tgidx_stats_mqc.json`` carrying the general stats, breakdown
        and target distribution payloads, plus the conditional edit-distance and
        anchor-run payloads this fixture's matched and anchor-measured reads
        genuinely populate. This is a wiring check, not an arithmetic one -- the
        payload contents are pinned in detail against ``AssignStats`` directly in
        ``tests/test_assign_reporting.py``.
        """
        records = [
            make_annotated_read("matched"),
            make_annotated_read("noumipos", with_anchor_pos=False),
            make_annotated_read("nomatch", index="", tail=NO_INDEX_TAIL),
            make_annotated_read("shortwindow", index="", tail=""),
        ]
        assigner = build_assigner(records)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.matched).is_greater_than(0)
        mqc_stats_path = tgidx_stats_mqc(tmp_path)
        assert_that(mqc_stats_path.exists()).is_true()

        payload = json.loads(mqc_stats_path.read_text())
        assert_that(payload).contains_key("general_stats")
        assert_that(payload).contains_key("breakdown")
        assert_that(payload).contains_key("target_distribution")

        assert_that(tgidx_edit_distance_mqc(tmp_path).exists()).is_true()
        assert_that(tgidx_anchor_run_mqc(tmp_path).exists()).is_true()

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

    def test_first_read_without_anchor_position_raises_before_any_output_is_written(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        records = [
            make_annotated_read("first", with_anchor_pos=False),
            make_annotated_read("second"),
        ]
        assigner = build_assigner(records)

        with pytest.raises(ValueError, match=ANCHOR_POS_KEY):
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
# The per-read body and the batcher
# ---------------------------------------------------------------------------

# One read per outcome, each planted with a distinct anchor run length so the
# run-length counter alone says which reads reached the forward scan.
MATCHED_RUN_LENGTH = 4
NO_MATCH_RUN_LENGTH = 6
SHORT_WINDOW_RUN_LENGTH = 3
NO_ANCHOR_POS_RUN_LENGTH = 7

MATCHED_READ = make_annotated_read("matched", run_length=MATCHED_RUN_LENGTH)

# The three unassigned outcomes, keyed by the counter each one bumps, which is
# also the field name it carries on the accumulator.
UNASSIGNED_READS = {
    "no_match": make_annotated_read(
        "nomatch", index="", tail=NO_INDEX_TAIL, run_length=NO_MATCH_RUN_LENGTH
    ),
    "short_window": make_annotated_read(
        "shortwindow", index="", tail="", run_length=SHORT_WINDOW_RUN_LENGTH
    ),
    "no_left_anchor_pos": make_annotated_read(
        "noumipos", with_anchor_pos=False, run_length=NO_ANCHOR_POS_RUN_LENGTH
    ),
}

# All four mutually exclusive outcomes, matched first so the reads double as a
# whole-stage input whose first read carries the anchor position tag the header
# check demands of it.
OUTCOME_READS = {"matched": MATCHED_READ, **UNASSIGNED_READS}
OUTCOME_TALLIES = tuple(OUTCOME_READS)
ALL_OUTCOME_READS = list(OUTCOME_READS.values())

# Batch shape the batcher tests drive: small enough that the synthetic input
# stays short, several batches deep so laziness is observable across it.
BATCH_SIZE = 3
FULL_BATCHES = 3


def make_annotated_reads(count: int) -> list[tuple[str, str, str]]:
    """Return ``count`` distinctly identified synthetic annotated reads.

    Args:
        count: Number of reads to build.

    Returns:
        One ``(header, seq, qual)`` read per index, in index order.
    """
    return [make_annotated_read(f"read{index}") for index in range(count)]


def outcome_tallies(counts: AssignCounts) -> dict[str, int]:
    """Return the four mutually exclusive outcome tallies, keyed by name.

    Read by name so a test can assert on all four at once: a read counted in
    the wrong bin only shows up when the three tallies it should not have
    touched are asserted alongside the one it should.

    Args:
        counts: Accumulator to read the tallies off.

    Returns:
        Mapping of tally name to the value it currently holds.
    """
    return {tally: getattr(counts, tally) for tally in OUTCOME_TALLIES}


class CountingReads:
    """A read source that counts how many reads have been pulled out of it.

    The batch source in the parallel driver's own tests, one level down: here it
    is single reads being counted rather than batches, because the batcher is
    the thing standing between the two.
    """

    def __init__(self, reads: Iterable[tuple[str, ...]]) -> None:
        """Wrap a sequence of reads.

        Args:
            reads: Reads to hand out one at a time.
        """
        self.reads = reads
        self.pulled = 0

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        """Yield reads, counting each one as it leaves.

        Yields:
            The next read.
        """
        for read in self.reads:
            self.pulled += 1
            yield read


@substitute_pool
class TestAssignReadHeader:
    """The rendered header the per-read body returns for a single read."""

    def test_assign_read_returns_the_header_the_whole_stage_writes(
        self, build_assigner: Callable[..., TargetAssigner], tmp_path: Path
    ) -> None:
        """Pin the per-read body against the streaming pass it was lifted from.

        Asserted against the headers a whole-stage run writes for the same reads
        rather than against copied literals, so the two cannot drift: the body is
        only a faithful extraction while it renders what the loop rendered.
        """
        assigner = build_assigner(ALL_OUTCOME_READS)

        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)
        written = [record[0][1:] for record in read_fastq(tgidx_fastq(tmp_path))]

        counts = AssignCounts()
        assigned = [assigner.assign_read(name, seq, counts) for name, seq, _ in ALL_OUTCOME_READS]

        assert_that(assigned).is_length(len(ALL_OUTCOME_READS))
        assert_that(assigned).is_equal_to(written)

    def test_assign_read_sets_the_matched_entry_and_the_span_of_the_observed_index(
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """The matched path is the only one that carries a position span."""
        name, seq, _ = MATCHED_READ
        assigner = build_assigner([MATCHED_READ])

        ann = ReadAnnotation.parse(assigner.assign_read(name, seq, AssignCounts()))

        assert_that(ann.get(TGIDX_NAME)).is_equal_to(TARGET_SEQ)
        start, end = parse_span(ann.get(position_key(TGIDX_NAME)))
        assert_that(seq[start:end]).is_equal_to(TARGET_SEQ)

    @pytest.mark.parametrize("read", list(UNASSIGNED_READS.values()), ids=tuple(UNASSIGNED_READS))
    def test_assign_read_sets_the_sentinel_and_no_span_when_unassigned(
        self, build_assigner: Callable[..., TargetAssigner], read: tuple[str, str, str]
    ) -> None:
        """Every unassigned outcome carries the sentinel and no span at all.

        A span left behind on an unassigned read would bound sequence no
        whitelist entry was ever verified against, and a consumer keying on the
        sentinel would still find coordinates to slice.
        """
        name, seq, _ = read
        assigner = build_assigner([MATCHED_READ, read])

        ann = ReadAnnotation.parse(assigner.assign_read(name, seq, AssignCounts()))

        assert_that(ann.get(TGIDX_NAME)).is_equal_to(target_assigner.NO_TARGET)
        assert_that(ann.get(position_key(TGIDX_NAME))).is_none()


class TestAssignReadCounts:
    """The tallies one call accumulates into the counts it is handed."""

    @pytest.mark.parametrize("outcome,read", list(OUTCOME_READS.items()), ids=OUTCOME_TALLIES)
    def test_assign_read_bumps_the_total_and_exactly_one_outcome_tally(
        self,
        build_assigner: Callable[..., TargetAssigner],
        outcome: str,
        read: tuple[str, str, str],
    ) -> None:
        """One call counts one read, once, in one bin.

        The three tallies the outcome should not have touched are asserted
        alongside the one it should: those are what stop a read being counted
        twice or counted in the wrong bin.
        """
        name, seq, _ = read
        assigner = build_assigner([MATCHED_READ, read])
        counts = AssignCounts()

        assigner.assign_read(name, seq, counts)

        assert_that(counts.total).is_equal_to(1)
        assert_that(outcome_tallies(counts)).is_equal_to(
            {tally: int(tally == outcome) for tally in OUTCOME_TALLIES}
        )

    def test_assign_read_keeps_the_outcome_invariant_after_every_call(
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """The four tallies sum to the total after every single call.

        Checked call by call rather than once at the end, since a read counted
        twice and a read never counted cancel out in the totals of a whole run
        while leaving the invariant broken after each of the two calls.
        """
        assigner = build_assigner(ALL_OUTCOME_READS)
        counts = AssignCounts()

        for reads_assigned, (name, seq, _) in enumerate(ALL_OUTCOME_READS, start=1):
            assigner.assign_read(name, seq, counts)

            assert_that(counts.total).is_equal_to(reads_assigned)
            assert_that(sum(outcome_tallies(counts).values())).is_equal_to(counts.total)

    def test_assign_read_records_the_target_and_edit_distance_of_a_matched_read(
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """Only matched reads reach the per-target and edit distance counters.

        Both distributions are rendered over ``matched`` as their denominator, so
        an unassigned read contributing to either would report a fraction of a
        number it was never part of.
        """
        records = [
            make_annotated_read("exact"),
            make_annotated_read("oneerror", index=SUBSTITUTED_TARGET),
            make_annotated_read("nomatch", index="", tail=NO_INDEX_TAIL),
        ]
        assigner = build_assigner(records)
        counts = AssignCounts()

        for name, seq, _ in records:
            assigner.assign_read(name, seq, counts)

        assert_that(counts.matched).is_equal_to(2)
        assert_that(dict(counts.target_counts)).is_equal_to({TARGET_SEQ: 2})
        assert_that(dict(counts.edit_distance_counts)).is_equal_to({0: 1, 1: 1})

    def test_assign_read_records_the_run_length_only_when_the_scan_is_reached(
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """Pin the denominator of the anchor run-length distribution.

        A read whose header carries no anchor position tag never reaches the
        forward scan, so it has no run length to place in any bin. Recording one
        for it would mean counting a run from a coordinate the read does not
        carry, and ``reads_with_measured_run`` would stop being the number of
        reads whose run was actually measured.
        """
        assigner = build_assigner(ALL_OUTCOME_READS)
        counts = AssignCounts()

        for name, seq, _ in ALL_OUTCOME_READS:
            assigner.assign_read(name, seq, counts)

        assert_that(counts.total).is_equal_to(len(ALL_OUTCOME_READS))
        assert_that(dict(counts.run_counts)).is_equal_to(
            {
                MATCHED_RUN_LENGTH: 1,
                NO_MATCH_RUN_LENGTH: 1,
                SHORT_WINDOW_RUN_LENGTH: 1,
            }
        )
        assert_that(counts.run_counts).does_not_contain_key(NO_ANCHOR_POS_RUN_LENGTH)

    def test_assign_read_accumulates_into_the_counts_it_is_handed(
        self, build_assigner: Callable[..., TargetAssigner]
    ) -> None:
        """Repeated calls add to the same accumulator rather than resetting it.

        The body takes the accumulator and mutates it instead of returning a
        per-read outcome object, which is what keeps a production run from
        allocating one such object per read.
        """
        calls = 3
        name, seq, _ = MATCHED_READ
        assigner = build_assigner([MATCHED_READ])
        counts = AssignCounts()

        for _ in range(calls):
            assigner.assign_read(name, seq, counts)

        assert_that(counts.total).is_equal_to(calls)
        assert_that(counts.matched).is_equal_to(calls)
        assert_that(dict(counts.target_counts)).is_equal_to({TARGET_SEQ: calls})
        assert_that(dict(counts.run_counts)).is_equal_to({MATCHED_RUN_LENGTH: calls})


class TestIterReadBatches:
    """Grouping a caller's annotated read iterator into batches, lazily."""

    def test_iter_read_batches_yields_full_batches_then_a_short_final_batch(self) -> None:
        """Reads come back grouped, in input order, with none added or lost."""
        reads = make_annotated_reads(BATCH_SIZE * FULL_BATCHES + 1)

        batches = list(iter_read_batches(iter(reads), BATCH_SIZE))

        assert_that([len(batch) for batch in batches]).is_equal_to(
            [BATCH_SIZE] * FULL_BATCHES + [1]
        )
        assert_that(list(chain.from_iterable(batches))).is_equal_to(reads)
        assert_that(sum(len(batch) for batch in batches)).is_equal_to(len(reads))

    def test_iter_read_batches_yields_nothing_for_an_empty_iterator(self) -> None:
        """An empty input yields no batches at all, not one empty batch.

        A stage run over no reads writes an empty FASTQ and a zero-count report,
        and it gets there by there being nothing to hand a worker: an empty batch
        would be a batch tallied, folded and written for reads that do not exist.
        """
        assert_that(list(iter_read_batches(iter([]), BATCH_SIZE))).is_empty()

    def test_iter_read_batches_yields_no_trailing_empty_batch_on_an_exact_multiple(
        self,
    ) -> None:
        """An input dividing exactly by the batch size ends on its last full batch."""
        reads = make_annotated_reads(BATCH_SIZE * FULL_BATCHES)

        batches = list(iter_read_batches(iter(reads), BATCH_SIZE))

        assert_that(batches).is_length(FULL_BATCHES)
        assert_that([len(batch) for batch in batches]).is_equal_to([BATCH_SIZE] * FULL_BATCHES)

    def test_iter_read_batches_drops_tuple_elements_beyond_the_first_three(self) -> None:
        """Trailing elements of a read tuple are dropped, not carried along.

        The read iterator yields six elements for a paired-end FASTQ and three
        for a single-end one. This stage reads R1 alone, so a batch holds
        ``(name, seq, qual)`` whichever it was handed.
        """
        reads = make_annotated_reads(BATCH_SIZE)
        paired = [(*read, f"{read[0]}_r2", read[1], read[2]) for read in reads]

        batches = list(iter_read_batches(iter(paired), BATCH_SIZE))

        assert_that(list(chain.from_iterable(batches))).is_equal_to(reads)

    def test_iter_read_batches_pulls_one_batch_of_reads_before_yielding_it(self) -> None:
        """Taking one batch consumes one batch of reads, not the whole source.

        The driver bounds resident memory by pulling its batch source no further
        ahead than its in-flight window. A batcher that drained its own source to
        build the first batch would defeat that, making peak memory proportional
        to the input rather than to the window.
        """
        source = CountingReads(make_annotated_reads(BATCH_SIZE * FULL_BATCHES + 1))

        batch = next(iter_read_batches(source, BATCH_SIZE))

        assert_that(batch).is_length(BATCH_SIZE)
        assert_that(source.pulled).is_equal_to(BATCH_SIZE)

    def test_iter_read_batches_never_reads_more_than_one_batch_ahead(self) -> None:
        """The read source stays within one batch of the batches handed out."""
        reads = make_annotated_reads(BATCH_SIZE * FULL_BATCHES + 1)
        source = CountingReads(reads)

        batched = 0
        for batch in iter_read_batches(source, BATCH_SIZE):
            batched += len(batch)
            assert_that(source.pulled - batched).is_less_than_or_equal_to(BATCH_SIZE)

        assert_that(batched).is_equal_to(len(reads))
        assert_that(source.pulled).is_equal_to(len(reads))


class TestBatchSizePolicy:
    """The batch size cap, and the sizing machinery this stage does without."""

    def test_max_reads_per_batch_is_the_committed_cap(self) -> None:
        """Pin the cap, restated here rather than read off the module.

        It is the constant that bounds resident memory: a batch is held whole in
        the worker that tallies it and again in the parent that writes it, so
        raising it raises the stage's memory ceiling by the same factor.
        """
        assert_that(MAX_READS_PER_BATCH).is_equal_to(2500)

    def test_module_declares_no_read_count_derived_batch_sizing(self) -> None:
        """Pin the deliberate divergence from the barcode stage's batch sizing.

        Barcode extraction divides a total read count between its workers, and
        pays for that count with a decompress pass over the whole FASTQ before
        it starts. This stage has no such count and must not acquire one: its
        batch size is the cap alone, so there is no floor to interpolate towards
        and nothing to divide. Restoring the symmetry would buy a whole extra
        pass over the input for a number the stage never uses.
        """
        assert_that(dir(target_assigner)).does_not_contain(
            "MIN_READS_PER_BATCH", "calc_batch_size"
        )
        assert_that(dir(TargetAssigner)).does_not_contain("calc_batch_size")


# ---------------------------------------------------------------------------
# The pool worker and the executor the stage runs it on
# ---------------------------------------------------------------------------

# Pool settings the stubbed-pool tests drive the stage at. The worker count is
# neither the constructor default nor the window the driver derives from it, so
# an assertion on it cannot pass by coincidence, and the batch size puts the five
# reads below over three batches.
POOL_WORKERS = 3
POOL_BATCH_SIZE = 2
POOL_READS = [*ALL_OUTCOME_READS, make_annotated_read("last")]
POOL_BATCH_LENGTHS = [POOL_BATCH_SIZE, POOL_BATCH_SIZE, 1]

# A read whose index is one error from the shipped whitelist entry and equidistant
# from both entries of the tied whitelist, so the answer it gets names which
# assigner produced it.
TIED_READ = make_annotated_read("tie", index=TIED_INDEX)
TIED_INPUT_NAME = "tied.r1_umi.fastq.gz"

# Events the stubbed pool, the progress bar and the recording write stream append
# to one shared log. The order the stage's with-block unwinds in is the whole
# point of this sub-task, so it has to be something a test reads back rather than
# something it takes on trust.
STREAM_OPEN_EVENT = "stream_open"
STREAM_CLOSE_EVENT = "stream_close"
EXECUTOR_ENTER_EVENT = "executor_enter"
EXECUTOR_EXIT_EVENT = "executor_exit"
EXECUTOR_SHUTDOWN_EVENT = "executor_shutdown"
PROGRESS_ENTER_EVENT = "progress_enter"
PROGRESS_EXIT_EVENT = "progress_exit"


class RecordingWriteStream:
    """A gzip write stream that reports its open and close to a shared event log.

    Wraps the real stream rather than standing in for it, so the stage still
    writes a genuine gzipped FASTQ while the moment that stream closes becomes
    observable. That moment is what the ordering test turns on: the pool has to
    be gone before it arrives.
    """

    def __init__(self, stream: SubprocessStream, events: list[str]) -> None:
        """Wrap a real write stream.

        Args:
            stream: The stream to delegate every write to.
            events: Shared log to append this stream's lifecycle events to.
        """
        self.stream = stream
        self.events = events

    def __enter__(self) -> SubprocessStream:
        """Open the underlying stream, noting the open.

        Returns:
            The real stream, so the stage writes to the compressor itself.
        """
        self.events.append(STREAM_OPEN_EVENT)
        return self.stream.__enter__()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the underlying stream, noting the close first.

        Args:
            exc_type: Type of any exception leaving the block.
            exc: The exception leaving the block, if any.
            tb: Traceback of that exception, if any.
        """
        self.events.append(STREAM_CLOSE_EVENT)
        self.stream.__exit__(exc_type, exc, tb)


class RecordingGzipFile:
    """A ``GzipFile`` stand-in handing out streams that report their lifecycle."""

    def __init__(self, filename: str, events: list[str]) -> None:
        """Note the file to compress into.

        Args:
            filename: Path the real stream compresses into.
            events: Shared log the stream appends its lifecycle events to.
        """
        self.filename = filename
        self.events = events

    def open_write_stream(self) -> RecordingWriteStream:
        """Open the real compressor, wrapped so its close is recorded.

        Returns:
            The recording wrapper around a real write stream.
        """
        return RecordingWriteStream(GzipFile(self.filename).open_write_stream(), self.events)


class RecordingAssignFuture:
    """A future stand-in that runs its batch on submission and holds the outcome.

    The work runs eagerly and its outcome is held until the result is asked for,
    which is how a real future looks from the driver's side: a worker's exception
    surfaces where the result is retrieved, never where the batch was submitted.
    """

    def __init__(self, worker: Callable[..., object], batch: list[tuple[str, str, str]]) -> None:
        """Run the worker over the batch and hold what it did.

        Args:
            worker: Single-argument callable applied to the batch.
            batch: Batch of reads handed to the worker.
        """
        self.value: object = None
        self.error: Exception | None = None
        try:
            self.value = worker(batch)
        except Exception as error:
            self.error = error

    def result(self, timeout: float | None = None) -> object:
        """Return what the worker returned, or raise what it raised.

        Args:
            timeout: Accepted for signature compatibility and ignored, because
                the work has already run.

        Returns:
            The value the worker returned.

        Raises:
            Exception: Whatever the worker raised, re-raised in the caller's frame.
        """
        if self.error is not None:
            raise self.error
        return self.value


class RecordingAssignExecutor:
    """An executor stand-in recording how the stage created, used and closed it.

    Submissions run eagerly in this process, so nothing read back here depends on
    timing. Unlike a bare stand-in it accepts and keeps the pool's keyword
    arguments, because what the stage passes for ``initializer`` and ``initargs``
    is the contract that gives each worker its assigner. It also runs that
    initializer once, exactly as a real pool runs it once per worker process
    before handing it any batch, which is where the in-process worker reads its
    assigner back from.
    """

    def __init__(
        self,
        events: list[str],
        max_workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        """Take the pool's settings and stand its initializer up.

        Args:
            events: Shared log to append this executor's lifecycle events to.
            max_workers: Pool width the stage asked for.
            initializer: Callable the stage wants run once per worker process.
            initargs: Arguments that initializer is to be called with.
        """
        self.events = events
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        self.submitted_batches: list[list[tuple[str, str, str]]] = []
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> "RecordingAssignExecutor":
        """Note that the pool has been entered.

        Returns:
            This executor, as a real pool hands back itself.
        """
        self.events.append(EXECUTOR_ENTER_EVENT)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Shut the pool down, noting the exit, as a real pool's exit does.

        Args:
            exc_type: Type of any exception leaving the block.
            exc: The exception leaving the block, if any.
            tb: Traceback of that exception, if any.

        Returns:
            False, so an exception in the block is never swallowed.
        """
        self.events.append(EXECUTOR_EXIT_EVENT)
        self.shutdown()
        return False

    def submit(
        self, worker: Callable[..., object], batch: list[tuple[str, str, str]]
    ) -> RecordingAssignFuture:
        """Accept a batch and run its worker straight away.

        Args:
            worker: Callable to apply to the batch.
            batch: Batch to hand to the worker.

        Returns:
            A future stand-in holding the outcome of the call.
        """
        self.submitted_batches.append(batch)
        return RecordingAssignFuture(worker, batch)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Note that the pool has been shut down.

        Args:
            wait: Accepted for signature compatibility and ignored.
            cancel_futures: Accepted for signature compatibility and ignored.
        """
        self.events.append(EXECUTOR_SHUTDOWN_EVENT)


class RecordingProgressBar:
    """A progress bar stand-in recording its task, its advances and its entry.

    It records how many batches had already been submitted at the moment it was
    entered, which is the property the stage's layout turns on: the driver
    submits its opening window when it is called, and that first submit is when a
    process pool forks its workers. Forking after this bar has started its
    refresh thread can leave an inherited lock held for ever in the child, so the
    driver call has to come first.
    """

    def __init__(self, harness: "PoolHarness") -> None:
        """Attach to the harness whose executor the entry is measured against.

        Args:
            harness: Harness holding the shared event log and the executor.
        """
        self.harness = harness
        self.tasks: list[tuple[str, int | None]] = []
        self.updates: list[tuple[str, int]] = []
        self.batches_submitted_on_entry: int | None = None

    def __enter__(self) -> "RecordingProgressBar":
        """Note the entry and how much work had already been submitted by then.

        Returns:
            This progress bar, as a real one hands back itself.
        """
        self.batches_submitted_on_entry = len(self.harness.submitted_batches)
        self.harness.events.append(PROGRESS_ENTER_EVENT)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Note that the progress bar's block has been left.

        Args:
            exc_type: Type of any exception leaving the block.
            exc: The exception leaving the block, if any.
            tb: Traceback of that exception, if any.

        Returns:
            False, so an exception in the block is never swallowed.
        """
        self.harness.events.append(PROGRESS_EXIT_EVENT)
        return False

    def add_task(self, description: str, total: int | None) -> str:
        """Record a task, keeping the total exactly as it was given.

        The total is required rather than defaulted, so a stage that never states
        one is a failure here instead of quietly taking a default.

        Args:
            description: Description the task renders under.
            total: Total the task counts towards, or None for an unknown one.

        Returns:
            The description, standing in for the task handle.
        """
        self.tasks.append((description, total))
        return description

    def update(self, task: str, advance: int) -> None:
        """Record one advance of a task.

        Args:
            task: Task handle being advanced.
            advance: Number of units to advance it by.
        """
        self.updates.append((task, advance))


class DriverCall(NamedTuple):
    """One call the stage made to the in-order driver.

    Attributes:
        executor: Executor the batches were submitted to.
        worker: Callable each batch was to be handed to.
        n_workers: Window argument, which sizes the driver's in-flight window.
    """

    executor: object
    worker: Callable[..., object]
    n_workers: int


class PoolHarness:
    """Stands in for the executor, the driver, the progress bar and the stream.

    The four are instrumented together because what the stage's with-block has to
    get right is the order between them: the pool is created after the output
    stream is open, torn down before that stream closes, and started before the
    progress bar begins refreshing. Recording all of them into one shared event
    log is what makes that order assertable rather than assumed.
    """

    def __init__(self) -> None:
        """Start with nothing created and nothing recorded."""
        self.events: list[str] = []
        self.executors: list[RecordingAssignExecutor] = []
        self.progress_bars: list[RecordingProgressBar] = []
        self.driver_calls: list[DriverCall] = []

    def executor_factory(
        self,
        max_workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> RecordingAssignExecutor:
        """Create a recording executor, keeping it for the tests to read.

        Args:
            max_workers: Pool width the stage asked for.
            initializer: Callable the stage wants run once per worker process.
            initargs: Arguments that initializer is to be called with.

        Returns:
            The recording executor the stage will use.
        """
        executor = RecordingAssignExecutor(
            self.events, max_workers, initializer=initializer, initargs=initargs
        )
        self.executors.append(executor)
        return executor

    def gzip_file_factory(self, filename: str) -> RecordingGzipFile:
        """Create a recording gzip file over a real compressor.

        Args:
            filename: Path the stage means to compress into.

        Returns:
            The recording gzip file the stage will open a stream on.
        """
        return RecordingGzipFile(filename, self.events)

    def progress_bar_factory(self, unit: str) -> RecordingProgressBar:
        """Create a recording progress bar, keeping it for the tests to read.

        Args:
            unit: Unit the real bar would label its counts with, ignored here.

        Returns:
            The recording progress bar the stage will use.
        """
        progress = RecordingProgressBar(self)
        self.progress_bars.append(progress)
        return progress

    def driver(
        self,
        executor: object,
        worker: Callable[..., object],
        batches: Iterable[list[tuple[str, str, str]]],
        n_workers: int,
    ) -> Iterator[object]:
        """Record the driver call and hand it straight to the real driver.

        Wrapping rather than replacing keeps the property under test intact: the
        opening window really is submitted when the driver is called.

        Args:
            executor: Executor the driver submits to.
            worker: Callable applied to each batch.
            batches: Batches to map over.
            n_workers: Window argument sizing the in-flight window.

        Returns:
            The real driver's iterator over the batch results.
        """
        self.driver_calls.append(DriverCall(executor, worker, n_workers))
        return map_batches_in_order(executor, worker, batches, n_workers)

    @property
    def submitted_batches(self) -> list[list[tuple[str, str, str]]]:
        """Return the batches submitted so far, over every executor created."""
        return [batch for executor in self.executors for batch in executor.submitted_batches]


@pytest.fixture
def pool_harness(monkeypatch: pytest.MonkeyPatch) -> PoolHarness:
    """Install the recording pool, driver, progress bar and write stream.

    The worker assigner is reset through ``monkeypatch`` before anything runs, so
    the module global the pool initializer writes is restored when the test ends
    and no test leaks its assigner into another.

    Args:
        monkeypatch: Patcher installing the stand-ins and undoing them after.

    Returns:
        The harness every stand-in records into.
    """
    harness = PoolHarness()
    monkeypatch.setattr(target_assigner, "WORKER_ASSIGNER", None)
    monkeypatch.setattr(target_assigner, "ProcessPoolExecutor", harness.executor_factory)
    monkeypatch.setattr(target_assigner, "GzipFile", harness.gzip_file_factory)
    monkeypatch.setattr(target_assigner, "progress_bar", harness.progress_bar_factory)
    monkeypatch.setattr(target_assigner, "map_batches_in_order", harness.driver)
    return harness


class TestTargetAssignerPoolConfiguration:
    """The pool width and the batch size the constructor resolves."""

    def test_defaults_to_one_worker_and_the_batch_size_cap(self) -> None:
        """A stage built without pool settings runs one worker over capped batches.

        The cap is the whole of the batch-size policy here. With no read count to
        divide there is nothing to interpolate towards, so the constant is the
        default itself rather than a ceiling on a computed value.
        """
        assigner = TargetAssigner(DUMMY_FASTQ, CHEMISTRY)

        assert_that(assigner.n_workers).is_equal_to(1)
        assert_that(assigner.batch_size).is_equal_to(MAX_READS_PER_BATCH)

    def test_worker_count_and_batch_size_are_kept_as_given(self) -> None:
        """Both are constructor parameters, so a fixture can span several batches.

        The batch size is deliberately a constructor parameter and nothing more:
        it exists so a small input can be put over more than one batch, which is
        the only way a test reaches the in-order fold at all.
        """
        assigner = TargetAssigner(
            DUMMY_FASTQ, CHEMISTRY, n_workers=POOL_WORKERS, batch_size=POOL_BATCH_SIZE
        )

        assert_that(assigner.n_workers).is_equal_to(POOL_WORKERS)
        assert_that(assigner.batch_size).is_equal_to(POOL_BATCH_SIZE)


class TestInitAssignWorker:
    """The once-per-process handover of the assigner a worker reuses."""

    def test_worker_assigner_starts_out_unset(self) -> None:
        """No assigner is installed until a worker process is initialised.

        Every test that installs one restores it, so a failure here is either a
        module declaring the global already filled in or another test leaking its
        own assigner into this one.
        """
        assert_that(target_assigner.WORKER_ASSIGNER).is_none()

    def test_init_assign_worker_installs_the_assigner_it_is_handed(
        self, build_assigner: Callable[..., TargetAssigner], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker's assigner is the very object the initializer was given.

        Set once per worker process and read-only after, so every batch that
        process handles reuses the one k-mer index instead of rebuilding it.
        """
        assigner = build_assigner([MATCHED_READ])
        monkeypatch.setattr(target_assigner, "WORKER_ASSIGNER", None)

        init_assign_worker(assigner)

        assert_that(target_assigner.WORKER_ASSIGNER).is_same_as(assigner)


class TestAssignReadBatch:
    """The single-argument worker one batch is handed to inside a pool process."""

    @pytest.fixture
    def worker_assigner(
        self, build_assigner: Callable[..., TargetAssigner], monkeypatch: pytest.MonkeyPatch
    ) -> TargetAssigner:
        """Install an assigner the way the pool initializer would, then restore it.

        Args:
            build_assigner: Factory writing the reads and building the assigner.
            monkeypatch: Patcher restoring the module global after the test.

        Returns:
            The assigner installed as this process's worker assigner.
        """
        assigner = build_assigner(POOL_READS)
        monkeypatch.setattr(target_assigner, "WORKER_ASSIGNER", None)
        init_assign_worker(assigner)
        return assigner

    def test_assign_read_batch_returns_one_read_per_read_it_was_given(
        self, worker_assigner: TargetAssigner
    ) -> None:
        """A batch of N reads comes back as N reads carrying N tallies.

        The stage annotates and never filters, so the batch a worker hands back
        is the batch it was given, read for read.
        """
        assigned, counts = assign_read_batch(POOL_READS)

        assert_that(assigned).is_length(len(POOL_READS))
        assert_that(counts.total).is_equal_to(len(POOL_READS))

    def test_assign_read_batch_passes_sequence_and_quality_through_untouched(
        self, worker_assigner: TargetAssigner
    ) -> None:
        """Only the header is rewritten; the read and its qualities are carried."""
        assigned, _ = assign_read_batch(POOL_READS)

        assert_that([(seq, qual) for _, seq, qual in assigned]).is_equal_to(
            [(seq, qual) for _, seq, qual in POOL_READS]
        )

    def test_assign_read_batch_returns_the_headers_assign_read_returns(
        self, worker_assigner: TargetAssigner
    ) -> None:
        """The worker renders exactly what the per-read body renders, in order.

        Asserted against the per-read body rather than against copied literals,
        because that is what makes the worker demonstrably the same computation
        as the serial pass it was lifted from.
        """
        expected = [
            worker_assigner.assign_read(name, seq, AssignCounts()) for name, seq, _ in POOL_READS
        ]

        assigned, _ = assign_read_batch(POOL_READS)

        assert_that([header for header, _, _ in assigned]).is_equal_to(expected)

    def test_assign_read_batch_returns_the_counts_accumulating_assign_read_gives(
        self, worker_assigner: TargetAssigner
    ) -> None:
        """The batch's tallies are the per-read body's tallies, accumulated.

        Every tally the report is rendered from is compared at once, so a worker
        that dropped one distribution while keeping the outcome bins would fail
        here rather than at the point someone reads the report.
        """
        expected = AssignCounts()
        for name, seq, _ in POOL_READS:
            worker_assigner.assign_read(name, seq, expected)

        _, counts = assign_read_batch(POOL_READS)

        assert_that(counts).is_equal_to(expected)

    def test_assign_read_batch_takes_the_batch_alone(self) -> None:
        """One parameter, so the driver can submit the worker as it stands.

        The driver applies its worker to one batch and binds nothing else, so an
        assigner taken as a second argument would have to be bound in by the
        caller for every batch it submits.
        """
        assert_that(list(signature(assign_read_batch).parameters)).is_length(1)

    def test_assign_read_batch_reads_the_assigner_from_the_module(
        self,
        worker_assigner: TargetAssigner,
        build_assigner: Callable[..., TargetAssigner],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The answer follows whichever assigner the process has installed.

        The read's index is one error from the shipped whitelist's single entry
        and equidistant from both entries of the tied whitelist, so it is matched
        under one assigner and unassigned under the other. That the answer
        changes with nothing but the module global is what pins where the worker
        reads its assigner from.
        """
        installed, _ = assign_read_batch([TIED_READ])
        tied_assigner = build_assigner(
            [TIED_READ], name=TIED_INPUT_NAME, chemistry=ChemistryTiedTargets()
        )

        monkeypatch.setattr(target_assigner, "WORKER_ASSIGNER", tied_assigner)
        swapped, _ = assign_read_batch([TIED_READ])

        assert_that(ReadAnnotation.parse(installed[0][0]).get(TGIDX_NAME)).is_equal_to(TARGET_SEQ)
        assert_that(ReadAnnotation.parse(swapped[0][0]).get(TGIDX_NAME)).is_equal_to(
            target_assigner.NO_TARGET
        )

    def test_assign_read_batch_maps_an_empty_batch_to_no_reads_and_zero_counts(
        self, worker_assigner: TargetAssigner
    ) -> None:
        """An empty batch is answered, not refused: nothing in, nothing out."""
        assigned, counts = assign_read_batch([])

        assert_that(assigned).is_empty()
        assert_that(counts).is_equal_to(AssignCounts())


class TestAssignTargetsPoolWiring:
    """How the stage creates, orders and feeds the pool it runs its worker on."""

    @pytest.fixture
    def assigner(self, build_assigner: Callable[..., TargetAssigner]) -> TargetAssigner:
        """Return an assigner over five reads, three workers and two per batch.

        Args:
            build_assigner: Factory writing the reads and building the assigner.

        Returns:
            The assigner every test in this class runs the stage from.
        """
        return build_assigner(POOL_READS, n_workers=POOL_WORKERS, batch_size=POOL_BATCH_SIZE)

    def test_executor_is_created_with_the_worker_initializer_and_the_parent_assigner(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """The pool is built at the stage's width and initialises every worker.

        The parent's already-constructed assigner is what is handed over, rather
        than the arguments for a worker to rebuild one from: the chemistry
        factory resolves by an explicit registry, and a chemistry that is not in
        that registry cannot be resolved again in the child at all.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(pool_harness.executors).is_length(1)
        executor = pool_harness.executors[0]
        assert_that(executor.max_workers).is_equal_to(POOL_WORKERS)
        assert_that(executor.initializer).is_same_as(init_assign_worker)
        assert_that(executor.initargs).is_length(1)
        assert_that(executor.initargs[0]).is_same_as(assigner)

    def test_driver_is_called_with_the_batch_worker_and_the_stages_worker_count(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """One driver call, over the stage's own pool, worker and worker count.

        The worker count is the driver's window argument, so a stage that passed
        something else would keep a window of the wrong width in flight however
        many workers it had started.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(pool_harness.driver_calls).is_length(1)
        call = pool_harness.driver_calls[0]
        assert_that(call.executor).is_same_as(pool_harness.executors[0])
        assert_that(call.worker).is_same_as(assign_read_batch)
        assert_that(call.n_workers).is_equal_to(POOL_WORKERS)

    def test_reads_are_submitted_in_batches_of_the_assigners_batch_size(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """The reads reach the pool grouped by the stage's own batch size.

        Full batches then a short final one, holding every read exactly once in
        input order, which is what the pool's results are folded back in.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        submitted = pool_harness.submitted_batches
        assert_that([len(batch) for batch in submitted]).is_equal_to(POOL_BATCH_LENGTHS)
        assert_that(list(chain.from_iterable(submitted))).is_equal_to(POOL_READS)

    def test_executor_is_torn_down_before_the_output_stream_is_closed(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """The pool is gone before the compressor is asked to finish.

        ``with`` unwinds in reverse, so the stream has to be entered first and
        the executor innermost. Pool workers are forked on the first submit and
        inherit the compressor's stdin write end, so a compressor still held open
        by a live worker never sees end of file and the run stops making
        progress. Getting this backwards is a hang rather than a failure, which
        is exactly why the order is asserted instead of assumed.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        events = pool_harness.events
        assert_that(events).contains(
            STREAM_OPEN_EVENT,
            EXECUTOR_ENTER_EVENT,
            EXECUTOR_EXIT_EVENT,
            EXECUTOR_SHUTDOWN_EVENT,
            STREAM_CLOSE_EVENT,
        )
        assert_that(events.index(STREAM_OPEN_EVENT)).is_less_than(
            events.index(EXECUTOR_ENTER_EVENT)
        )
        assert_that(events.index(EXECUTOR_EXIT_EVENT)).is_less_than(
            events.index(STREAM_CLOSE_EVENT)
        )
        assert_that(events.index(EXECUTOR_SHUTDOWN_EVENT)).is_less_than(
            events.index(STREAM_CLOSE_EVENT)
        )

    def test_batch_results_are_folded_inside_the_block_that_keeps_the_pool_alive(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """The results are consumed while the pool is still up.

        The driver never creates, enters, exits or shuts down the pool, so a
        caller that abandoned its iterator outside the block would leave futures
        still queued for the shutdown to drain, with their results never written.
        The fold therefore has to happen between the pool's entry and its exit,
        which is where the progress bar's own block sits.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        events = pool_harness.events
        assert_that(events).contains(PROGRESS_ENTER_EVENT, PROGRESS_EXIT_EVENT)
        assert_that(events.index(EXECUTOR_ENTER_EVENT)).is_less_than(
            events.index(PROGRESS_ENTER_EVENT)
        )
        assert_that(events.index(PROGRESS_EXIT_EVENT)).is_less_than(
            events.index(EXECUTOR_EXIT_EVENT)
        )

    def test_first_batch_is_submitted_before_the_progress_bar_is_entered(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """The pool has work in flight before the bar starts refreshing.

        The driver submits its opening window when it is called, and that first
        submit is when a process pool forks its workers. The progress bar runs a
        refresh thread for the whole of its block, and forking a multi-threaded
        process can leave an inherited lock held for ever in the child, so the
        driver call has to sit above that block. A test that merely checked the
        bar exists would not catch a regression that moved the call inside it.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(pool_harness.progress_bars).is_length(1)
        submitted_on_entry = pool_harness.progress_bars[0].batches_submitted_on_entry
        assert_that(submitted_on_entry).is_not_none()
        assert_that(submitted_on_entry).is_greater_than_or_equal_to(1)

    def test_progress_task_states_no_total_and_advances_by_each_batch_read_count(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """One task with an unknown total, advanced batch by batch to the input.

        The total is unknown on purpose. A determinate one would need a second
        full decompress pass over the input just to count its reads, which is a
        cost this stage does not pay, so the bar renders its count against a
        question mark instead.
        """
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        progress = pool_harness.progress_bars[0]
        assert_that(progress.tasks).is_length(1)
        description, total = progress.tasks[0]
        assert_that(total).is_none()
        assert_that([task for task, _ in progress.updates]).is_equal_to(
            [description] * len(POOL_BATCH_LENGTHS)
        )
        assert_that([advance for _, advance in progress.updates]).is_equal_to(POOL_BATCH_LENGTHS)
        assert_that(sum(advance for _, advance in progress.updates)).is_equal_to(len(POOL_READS))

    def test_every_read_is_written_once_in_input_order_across_batches(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """Reads leave in input order however the batches they came in finished."""
        assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        written = read_fastq(tgidx_fastq(tmp_path))
        assert_that(written).is_length(len(POOL_READS))
        ids = [ReadAnnotation.parse(record[0][1:]).read_id for record in written]
        assert_that(ids).is_equal_to(
            [ReadAnnotation.parse(name).read_id for name, _, _ in POOL_READS]
        )
        for read, out in zip(POOL_READS, written):
            assert_that(out[1]).is_equal_to(read[1])
            assert_that(out[3]).is_equal_to(read[2])

    def test_batch_tallies_are_folded_into_the_run_stats(
        self, assigner: TargetAssigner, pool_harness: PoolHarness, tmp_path: Path
    ) -> None:
        """Three batches of tallies fold to what one serial pass would have counted.

        The fold adds counts where a plain dictionary update would overwrite
        them, so every count an earlier batch recorded for a key a later batch
        also saw has to survive. Comparing the whole stats object is what catches
        the loss.
        """
        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        expected = AssignCounts()
        for name, seq, _ in POOL_READS:
            assigner.assign_read(name, seq, expected)

        assert_that(stats).is_equal_to(expected.to_stats(assigner.anchor_base))

    def test_first_read_without_anchor_position_raises_before_the_pool_is_created(
        self,
        build_assigner: Callable[..., TargetAssigner],
        pool_harness: PoolHarness,
        tmp_path: Path,
    ) -> None:
        """A FASTQ that never went through UMI extraction forks nothing at all.

        The check already ran before either output file was opened. Now it also
        has to run before the pool exists, so a rejected run leaves neither
        truncated outputs nor worker processes behind.
        """
        assigner = build_assigner(
            [make_annotated_read("first", with_anchor_pos=False), MATCHED_READ],
            n_workers=POOL_WORKERS,
            batch_size=POOL_BATCH_SIZE,
        )

        with pytest.raises(ValueError, match=ANCHOR_POS_KEY):
            assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(pool_harness.executors).is_empty()
        assert_that(pool_harness.events).is_empty()
        assert_that(tgidx_fastq(tmp_path).exists()).is_false()
        assert_that(tgidx_stats(tmp_path).exists()).is_false()

    def test_empty_input_submits_no_batch_and_still_writes_both_outputs(
        self,
        build_assigner: Callable[..., TargetAssigner],
        pool_harness: PoolHarness,
        tmp_path: Path,
    ) -> None:
        """A run over no reads builds a pool, hands it nothing and closes cleanly.

        The batcher yields no batch at all rather than one empty batch, so there
        is nothing to submit, and both output files are still written: a run over
        no reads is an empty run, not a failed one.
        """
        assigner = build_assigner([], n_workers=POOL_WORKERS, batch_size=POOL_BATCH_SIZE)

        stats = assigner.assign_targets(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(pool_harness.executors).is_length(1)
        assert_that(pool_harness.submitted_batches).is_empty()
        assert_that(stats.total_reads).is_equal_to(0)
        assert_that(read_fastq(tgidx_fastq(tmp_path))).is_empty()
        assert_that(tgidx_stats(tmp_path).read_text()).contains("Total reads: 0")


# ---------------------------------------------------------------------------
# The stage over a real forked process pool
# ---------------------------------------------------------------------------

# Cap on how long one real-pool run may take. A teardown-order regression
# deadlocks rather than fails, so every run is given a deadline and fails the
# suite on it instead of hanging the session.
ASSIGNMENT_DEADLINE_S = 180

# The real-pool fixture. Small enough to stay fast, deep enough that the input
# spans ten batches at every worker count under test, so the in-order fold is
# reached rather than merely available.
REAL_POOL_READS = 250
REAL_POOL_BATCH_SIZE = 25
REAL_POOL_WORKER_COUNTS = (1, 4, 16)
REAL_POOL_PREFIX = "real_pool"

# Worker count and run count for the repeat-run comparison. More than one worker,
# so the two runs differ in nothing but when their workers happened to be
# scheduled.
REPEAT_RUN_WORKERS = 4
REPEAT_RUNS = 2

# The stats object a run returns lives in the child that ran it, so the child
# writes it out beside the stage's own outputs for the parent to read back.
STATS_PICKLE_NAME = "assign_stats.pickle"

# Read shapes the real-pool input cycles through, so every outcome, both edit
# distances and two anchor run lengths are represented. The first shape carries
# the anchor position tag, since the first read of the file is the one the header
# check is made against.
REAL_POOL_READ_SHAPES = (
    {},
    {"index": SUBSTITUTED_TARGET},
    {"index": "", "tail": NO_INDEX_TAIL},
    {"index": "", "tail": ""},
    {"with_anchor_pos": False},
    {"run_length": 7},
)


def make_real_pool_reads(count: int) -> list[tuple[str, str, str]]:
    """Return distinctly identified reads cycling through every outcome shape.

    Args:
        count: Number of reads to build.

    Returns:
        One ``(header, seq, qual)`` read per index, in index order.
    """
    return [
        make_annotated_read(
            f"read{index}", **REAL_POOL_READ_SHAPES[index % len(REAL_POOL_READ_SHAPES)]
        )
        for index in range(count)
    ]


def run_assignment_in_process_group(
    fastq_path: Path, output_dir: Path, n_workers: int, batch_size: int
) -> None:
    """Run one real-pool assignment in a forked child leading its own process group.

    A teardown-order regression deadlocks rather than fails - a compressor still
    held open by a live pool worker never sees end of file - so the run is given a
    deadline. The child leads its own process group so that killing it takes the
    pool workers with it: that releases the inherited pipe write ends, letting the
    stranded compressor see end of file and exit instead of lingering for the rest
    of the session.

    The stats the run returns live in the child, so they are written out beside
    the stage's own outputs for the parent to read back.

    Args:
        fastq_path: UMI-annotated FASTQ to assign target indices in.
        output_dir: Directory the run writes its outputs and its stats into.
        n_workers: Pool width for the run.
        batch_size: Reads per batch, which sets how many batches the input spans.

    Raises:
        AssertionError: If the child misses its deadline, or exits non-zero.
    """

    def target() -> None:
        os.setsid()
        assigner = TargetAssigner(
            str(fastq_path), CHEMISTRY, n_workers=n_workers, batch_size=batch_size
        )
        stats = assigner.assign_targets(output_dir=str(output_dir), prefix=REAL_POOL_PREFIX)
        (output_dir / STATS_PICKLE_NAME).write_bytes(pickle.dumps(stats))

    proc = multiprocessing.get_context("fork").Process(target=target)
    proc.start()
    proc.join(ASSIGNMENT_DEADLINE_S)

    if proc.is_alive():
        os.killpg(proc.pid, signal.SIGKILL)
        proc.join(ASSIGNMENT_DEADLINE_S)
        pytest.fail(
            f"assign_targets did not finish within {ASSIGNMENT_DEADLINE_S}s - the compressor "
            "writing the assigned FASTQ is most likely blocked waiting on EOF for a pipe "
            "still held open by a pool worker"
        )

    assert_that(proc.exitcode).is_equal_to(0)


def run_assignment_into(
    work_dir: Path, name: str, fastq_path: Path, n_workers: int, batch_size: int
) -> Path:
    """Run one configuration into a directory of its own and return it.

    Args:
        work_dir: Directory the run's own output directory is created under.
        name: Name of that output directory, naming the configuration.
        fastq_path: UMI-annotated FASTQ to assign target indices in.
        n_workers: Pool width for the run.
        batch_size: Reads per batch for the run.

    Returns:
        The directory the run wrote its outputs into.
    """
    output_dir = work_dir / name
    output_dir.mkdir()
    run_assignment_in_process_group(fastq_path, output_dir, n_workers, batch_size)
    return output_dir


def assigned_text(output_dir: Path) -> str:
    """Return the decompressed text of the assigned FASTQ a run wrote.

    Args:
        output_dir: Directory the run wrote its outputs into.

    Returns:
        The decompressed contents of the assigned FASTQ.
    """
    return read_gzip_text(tgidx_fastq(output_dir, REAL_POOL_PREFIX))


def assigned_report(output_dir: Path) -> str:
    """Return a run's stats report with its volatile run-detail lines dropped.

    Args:
        output_dir: Directory the run wrote its outputs into.

    Returns:
        The report text, without the version and timestamp lines that differ
        between any two runs.
    """
    return strip_report_run_details(tgidx_stats(output_dir, REAL_POOL_PREFIX).read_text())


def assigned_stats(output_dir: Path) -> AssignStats:
    """Return the stats object the run in a directory handed back.

    Args:
        output_dir: Directory the run wrote its outputs into.

    Returns:
        The :class:`AssignStats` the child process returned from the run.
    """
    return pickle.loads((output_dir / STATS_PICKLE_NAME).read_bytes())


def first_difference(produced: str, reference: str) -> str | None:
    """Describe the first line at which two assignment outputs diverge.

    An equality dump of two whole 250-read outputs runs to tens of kilobytes and
    buries the only fact worth reporting: the line at which the order first
    diverged. Naming that line and both of its values keeps a failure readable.

    Args:
        produced: Text produced by the run under test.
        reference: Text produced by the reference run.

    Returns:
        A one-line description of the first divergence, or None when the two
        texts hold exactly the same lines in exactly the same order.
    """
    produced_lines = produced.splitlines()
    reference_lines = reference.splitlines()

    for index, (produced_line, reference_line) in enumerate(
        zip(produced_lines, reference_lines), start=1
    ):
        if produced_line != reference_line:
            return (
                f"first differs at line {index}: produced {produced_line!r}, "
                f"reference {reference_line!r}"
            )

    if len(produced_lines) != len(reference_lines):
        return (
            f"line counts differ: produced {len(produced_lines)}, "
            f"reference {len(reference_lines)}"
        )

    if produced != reference:
        return "every line matches but the two texts differ in their line endings"

    return None


@pytest.fixture(scope="session")
def real_pool_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the multi-outcome FASTQ every real-pool run reads, once per session.

    Args:
        tmp_path_factory: Factory for the directory the FASTQ is written into.

    Returns:
        Path of the written UMI-annotated FASTQ.
    """
    fastq_path = tmp_path_factory.mktemp("real_pool_input") / INPUT_FASTQ_NAME
    write_fastq(fastq_path, make_real_pool_reads(REAL_POOL_READS))
    return fastq_path


class TestAssignTargetsWorkerCountEquivalence:
    """Assignment output is identical whatever the worker count.

    Every other test of the stage in this file substitutes the executor, so none
    of them exercises the interaction between forked pool workers and the
    compressor writing the assigned FASTQ. This class and the repeat-run one
    below are where that interaction is pinned. Here the pool is real: workers
    are forked on the first submit and inherit that compressor's pipe write end,
    so tearing the two down in the wrong order strands it on an end of file that
    never arrives.

    Nothing is compared against a stored golden. The runs are compared against
    each other and against a single-batch run no scheduling can perturb, so the
    assertions hold whatever the input happens to contain.

    Every run is driven through ``run_assignment_in_process_group``, so a
    teardown-order regression fails on the deadline instead of hanging the
    session.
    """

    @pytest.fixture(scope="class")
    def outputs_by_worker_count(
        self, real_pool_input: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> dict[int, Path]:
        """Assign targets once per worker count, mapping each count to its output.

        Args:
            real_pool_input: The FASTQ every run reads.
            tmp_path_factory: Factory for the directory the runs write under.

        Returns:
            Mapping of worker count to the directory that run wrote into.
        """
        work_dir = tmp_path_factory.mktemp("real_pool_worker_counts")
        return {
            n_workers: run_assignment_into(
                work_dir,
                f"workers_{n_workers}",
                real_pool_input,
                n_workers,
                REAL_POOL_BATCH_SIZE,
            )
            for n_workers in REAL_POOL_WORKER_COUNTS
        }

    @pytest.fixture(scope="class")
    def single_batch_output(
        self, real_pool_input: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        """Assign targets over the whole input as a single batch at one worker.

        Args:
            real_pool_input: The FASTQ every run reads.
            tmp_path_factory: Factory for the directory the run writes under.

        Returns:
            The directory the single-batch reference run wrote into.
        """
        work_dir = tmp_path_factory.mktemp("real_pool_single_batch")
        return run_assignment_into(work_dir, "single_batch", real_pool_input, 1, REAL_POOL_READS)

    def test_every_run_assigns_the_whole_input_across_every_outcome(
        self, outputs_by_worker_count: dict[int, Path]
    ) -> None:
        """Proof the comparisons below are not vacuous.

        Every read of the input is accounted for, and all four outcomes are
        represented, so the texts being compared carry the whole of the stage's
        behaviour rather than a single repeated answer.
        """
        for n_workers, output_dir in outputs_by_worker_count.items():
            stats = assigned_stats(output_dir)
            described = f"at {n_workers} workers"
            assert_that(stats.total_reads).described_as(described).is_equal_to(REAL_POOL_READS)
            assert_that(stats.matched).described_as(described).is_greater_than(0)
            assert_that(stats.unmatched_no_match).described_as(described).is_greater_than(0)
            assert_that(stats.unmatched_no_left_anchor_pos).described_as(
                described
            ).is_greater_than(0)
            assert_that(stats.unmatched_short_window).described_as(described).is_greater_than(0)
            assert_that(stats.homopolymer_run_counts).described_as(described).is_not_empty()

    def test_assigned_fastq_is_identical_across_worker_counts(
        self, outputs_by_worker_count: dict[int, Path]
    ) -> None:
        """The assigned FASTQ decompresses to the same text at every worker count."""
        reference_workers, *other_worker_counts = REAL_POOL_WORKER_COUNTS
        reference = assigned_text(outputs_by_worker_count[reference_workers])

        for n_workers in other_worker_counts:
            produced = assigned_text(outputs_by_worker_count[n_workers])
            assert_that(first_difference(produced, reference)).described_as(
                f"assigned FASTQ at {n_workers} workers against the "
                f"{reference_workers}-worker run"
            ).is_none()

    def test_assign_stats_are_identical_field_for_field_across_worker_counts(
        self, outputs_by_worker_count: dict[int, Path]
    ) -> None:
        """Every field of the returned stats is the same at every worker count.

        Asserted field by field rather than as one object, so a failure names the
        tally that diverged instead of dumping two whole stats objects.
        """
        reference_workers, *other_worker_counts = REAL_POOL_WORKER_COUNTS
        reference = assigned_stats(outputs_by_worker_count[reference_workers])

        for n_workers in other_worker_counts:
            produced = assigned_stats(outputs_by_worker_count[n_workers])
            for stats_field in fields(AssignStats):
                assert_that(getattr(produced, stats_field.name)).described_as(
                    f"{stats_field.name} at {n_workers} workers against the "
                    f"{reference_workers}-worker run"
                ).is_equal_to(getattr(reference, stats_field.name))

    def test_stats_report_is_identical_across_worker_counts(
        self, outputs_by_worker_count: dict[int, Path]
    ) -> None:
        """The written report is identical once its run details are dropped.

        The version and timestamp lines differ between any two runs, so they are
        stripped. Everything else, distributions included, has to match to the
        character.
        """
        reference_workers, *other_worker_counts = REAL_POOL_WORKER_COUNTS
        reference = assigned_report(outputs_by_worker_count[reference_workers])

        for n_workers in other_worker_counts:
            produced = assigned_report(outputs_by_worker_count[n_workers])
            assert_that(produced).described_as(
                f"stats report at {n_workers} workers against the "
                f"{reference_workers}-worker run"
            ).is_equal_to(reference)

    def test_multi_batch_output_matches_the_single_batch_reference(
        self, outputs_by_worker_count: dict[int, Path], single_batch_output: Path
    ) -> None:
        """Many batches over many workers reproduce one batch on one worker exactly.

        This is the strongest statement of the input-order guarantee available
        here: the single-batch run has no fold order to get wrong, so it is a
        reference no scheduling can perturb, and every multi-batch run has to
        reproduce it byte for byte.
        """
        reference = assigned_text(single_batch_output)

        for n_workers, output_dir in outputs_by_worker_count.items():
            produced = assigned_text(output_dir)
            assert_that(first_difference(produced, reference)).described_as(
                f"assigned FASTQ at {n_workers} workers against the single-batch run"
            ).is_none()


class TestAssignTargetsRepeatRunStability:
    """The same real-pool configuration, run twice, writes the same bytes.

    Identical input giving byte-identical output is a hard requirement of the
    stage, and the worker-count comparison alone does not establish it: two runs
    at the same width still differ in when their workers were scheduled.
    """

    @pytest.fixture(scope="class")
    def repeated_outputs(
        self, real_pool_input: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> list[Path]:
        """Run one multi-worker, multi-batch configuration twice over.

        Args:
            real_pool_input: The FASTQ both runs read.
            tmp_path_factory: Factory for the directory the runs write under.

        Returns:
            The output directory of each run, in the order they were run.
        """
        work_dir = tmp_path_factory.mktemp("real_pool_repeat")
        return [
            run_assignment_into(
                work_dir,
                f"run_{run}",
                real_pool_input,
                REPEAT_RUN_WORKERS,
                REAL_POOL_BATCH_SIZE,
            )
            for run in range(REPEAT_RUNS)
        ]

    def test_repeat_runs_write_the_same_assigned_fastq(self, repeated_outputs: list[Path]) -> None:
        """Two runs of one configuration decompress to exactly the same text."""
        first, *later_runs = repeated_outputs
        reference = assigned_text(first)

        for run, output_dir in enumerate(later_runs, start=1):
            assert_that(first_difference(assigned_text(output_dir), reference)).described_as(
                f"assigned FASTQ of run {run} against run 0"
            ).is_none()

    def test_repeat_runs_write_the_same_stats_report(self, repeated_outputs: list[Path]) -> None:
        """Two runs of one configuration report exactly the same tallies."""
        first, *later_runs = repeated_outputs
        reference = assigned_report(first)

        for run, output_dir in enumerate(later_runs, start=1):
            assert_that(assigned_report(output_dir)).described_as(
                f"stats report of run {run} against run 0"
            ).is_equal_to(reference)


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
# The stage runs after extract-umis and before bam-tag-deduplicate, and the grouped
# help listing is the only place a user reads that order off, so the position is
# part of the contract and not a cosmetic detail.
USER_COMMAND_GROUP = "Commands for users"
PRECEDING_COMMAND = "extract-umis"
FOLLOWING_COMMAND = "bam-tag-deduplicate"

# The committed 200-read golden input and the prefix every stage of the
# end-to-end run writes under. It is the smallest input carrying the full
# custom_seq layout, so it is the cheapest real fixture that can be driven
# through barcode extraction, UMI extraction and target assignment in turn.
GOLDEN_INPUT_DIR = Path(__file__).parent / "data" / "golden"
GOLDEN_INPUT_NAME = "custom_seq_1_0_small_R1.fastq.gz"
GOLDEN_PREFIX = "custom_seq_1_0_small"

# The parameter name the pool width is bound to and the two spellings it is
# offered under. Both spellings are the convention extract-barcodes and split-bam
# already use, so a user who knows one command knows this one.
CPU_COUNT_OPTION = "cpu_count"
CPU_COUNT_OPTS = ["-n", "--cpu_count"]

# The saturation cap, restated here rather than imported. A test that read the
# number off the module would agree with whatever the module happened to hold,
# which is no statement at all about what the number should be.
EXPECTED_MAX_WORKERS = 16

# Core counts the option's default is resolved under. The first stands for a
# machine far wider than the cap, where the default must stop at the cap; the
# second for one narrower than it, where the default must come down to the cores
# actually available rather than ask for more than the machine has.
CORES_ABOVE_CAP = 1024
CORES_BELOW_CAP = 3

# Module name the CLI source is re-executed under to resolve the option default
# against a chosen core count. It is deliberately not "carmack.__main__": the
# probe must not displace the imported module every other CLI test patches.
PROBE_MODULE_NAME = "carmack_cli_probe"

# The command whose worker default is every usable core. This stage's default is
# the one that differs, so the divergence is asserted against the neighbouring
# command rather than described in a comment nobody has to keep true.
UNCAPPED_WORKER_COMMAND = "extract-barcodes"

# Fragments the worker option's own help text must carry. The cap alone is not
# enough: a number with no stated reason reads as arbitrary and is exactly the
# kind of thing a later reader raises to match the neighbouring command. The
# count is asserted against the declared help rather than the rendered screen
# because show_default renders it in a "[default: N]" badge regardless.
HELP_FRAGMENTS = [str(EXPECTED_MAX_WORKERS), "no faster"]

# Pool width handed to the command to check the option reaches the constructor.
# It differs from both the resolved default and from one, so a wiring that
# ignored the option and passed either would still fail.
CLI_WORKERS = 3

# The width the command runs at when no -n is given: every usable core, capped.
# Not get_cpu_count() - beyond the cap the parent's own parse-and-write loop is
# the bound, so the extra workers cost processes and buy no throughput.
DEFAULT_CPU_COUNT = min(DEFAULT_MAX_WORKERS, get_cpu_count())


def user_command_names() -> list[str]:
    """Return the commands listed in the user-facing help group, in help order.

    Returns:
        The command names of the ``Commands for users`` group, in the order the
        grouped help listing renders them.
    """
    groups = click.rich_click.COMMAND_GROUPS["carmack"]
    return next(group["commands"] for group in groups if group["name"] == USER_COMMAND_GROUP)


def command_option(cli: click.Group, command_name: str, option_name: str) -> click.Parameter:
    """Return one declared option of a command on a carmack command group.

    Args:
        cli: The command group the command is registered on.
        command_name: Name the command is registered under.
        option_name: Parameter name the option binds to.

    Returns:
        The Click parameter, so its names, default and rendering flags can be
        read off the declaration itself rather than inferred from help text.
    """
    command = cli.commands[command_name]
    declared = [param for param in command.params if param.name == option_name]
    assert_that(declared).described_as(f"{command_name} option '{option_name}'").is_length(1)
    return declared[0]


def cli_group_with_core_count(core_count: int) -> click.Group:
    """Re-execute the CLI source in a fresh namespace under a chosen core count.

    An option default is resolved once, while the module is being imported, so a
    patch applied afterwards cannot reach it. Running the same source under a
    private module name resolves it again against the supplied core count while
    leaving the imported ``carmack.__main__`` - the module every other CLI test
    patches - untouched.

    Args:
        core_count: Value ``get_cpu_count`` reports while the source runs.

    Returns:
        The command group built by that run.
    """
    spec = importlib.util.spec_from_file_location(PROBE_MODULE_NAME, carmack.__main__.__file__)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("carmack.utils.get_cpu_count", return_value=core_count):
        spec.loader.exec_module(module)
    return module.carmack_cli


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def flatten_help(output: str) -> str:
    """Collapse a rendered help screen into one lowercase line.

    rich_click lays the options out in a bordered table, so a help string longer
    than its column wraps over several rows with a rule character at each end.
    Dropping the rules and collapsing the whitespace puts the string back
    together as the user reads it, which is what an assertion on a phrase needs.

    Colour codes are stripped first, and that is not cosmetic. rich_click styles
    the help whenever the terminal accepts colour, which puts escape sequences
    between the words of a wrapped phrase and inside a metavar's angle brackets.
    Collapsing whitespace alone leaves those sequences behind as tokens, so an
    assertion on a phrase passes only on a terminal that refused colour -- the
    tests would pass in CI and fail for anyone running them locally.

    Args:
        output: The help screen as the runner captured it.

    Returns:
        The same text as a single lowercase line of space-separated words.
    """
    plain = ANSI_ESCAPE.sub("", output)
    return " ".join(plain.replace("│", " ").split()).lower()


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
        mock_assigner.assert_called_once_with(DUMMY_FASTQ, CHEMISTRY, n_workers=DEFAULT_CPU_COUNT)
        mock_assigner.return_value.assign_targets.assert_called_once_with(
            str(tmp_path), expected_prefix
        )

    def test_command_is_listed_in_the_top_level_help(self) -> None:
        runner = CliRunner()

        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains(COMMAND_NAME)

    def test_command_group_places_it_between_extract_umis_and_bam_tag_deduplicate(self) -> None:
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
        rendered = flatten_help(result.output)

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(rendered).contains(COMMAND_NAME)
        assert_that(rendered).contains(FASTQ_METAVAR)
        assert_that(rendered).contains("--chemistry")


class TestAssignTargetsCliWorkerCount:
    """The option the stage's pool width is set through, and its capped default.

    Every test here either patches the constructor or reads the option off the
    command declaration. None of them runs an assignment: the pool forks, so an
    object patched in the parent records nothing a child did, and a test that
    asserted on one would pass whether or not the wiring worked.
    """

    @pytest.mark.parametrize("option", CPU_COUNT_OPTS)
    def test_worker_count_reaches_the_assigner_as_n_workers(
        self, tmp_path: Path, option: str
    ) -> None:
        """A worker count given on the command line is the width the stage runs at.

        Args:
            tmp_path: Output directory the command is pointed at.
            option: Spelling of the worker option the count is passed under.
        """
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
                    option,
                    str(CLI_WORKERS),
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_assigner.assert_called_once_with(DUMMY_FASTQ, CHEMISTRY, n_workers=CLI_WORKERS)

    def test_default_max_workers_is_the_measured_saturation_point(self) -> None:
        """The cap is sixteen, the width past which the stage measured no faster.

        Beyond it the parent's own single-threaded parse-and-write loop is the
        bound, so the number is a measurement rather than a preference and is
        restated here instead of read off the module it constrains.
        """
        assert_that(DEFAULT_MAX_WORKERS).is_equal_to(EXPECTED_MAX_WORKERS)

    def test_worker_option_defaults_to_the_core_count_capped_at_the_saturation_point(
        self,
    ) -> None:
        """The default is the machine's usable cores, held down to the cap.

        Asserted against the cap and the core-count helper rather than a literal:
        on a machine narrower than the cap the correct default is the lower
        number, and a test naming sixteen would be wrong there.
        """
        option = command_option(carmack.__main__.carmack_cli, COMMAND_NAME, CPU_COUNT_OPTION)

        assert_that(option.default).is_equal_to(min(DEFAULT_MAX_WORKERS, get_cpu_count()))

    @pytest.mark.parametrize(
        "core_count, expected_default",
        [
            pytest.param(CORES_ABOVE_CAP, EXPECTED_MAX_WORKERS, id="wide-machine-stops-at-cap"),
            pytest.param(CORES_BELOW_CAP, CORES_BELOW_CAP, id="narrow-machine-follows-cores"),
        ],
    )
    def test_worker_option_default_resolves_against_the_cap_and_the_core_count(
        self, core_count: int, expected_default: int
    ) -> None:
        """Both halves of the default hold: the cap binds above it, the cores below.

        Args:
            core_count: Cores the machine reports while the default is resolved.
            expected_default: Width the option must settle on at that core count.
        """
        cli = cli_group_with_core_count(core_count)

        option = command_option(cli, COMMAND_NAME, CPU_COUNT_OPTION)

        assert_that(option.default).is_equal_to(expected_default)

    def test_worker_option_default_is_capped_where_extract_barcodes_takes_every_core(
        self,
    ) -> None:
        """This stage's default diverges from its neighbour's, deliberately.

        Both commands offer the same option under the same names, which makes
        the differing default look like an oversight worth tidying away. It is
        not: extract-barcodes scales with cores and this stage stops scaling at
        the cap, so the two are compared here on one machine wide enough for the
        difference to show.
        """
        cli = cli_group_with_core_count(CORES_ABOVE_CAP)

        assigned = command_option(cli, COMMAND_NAME, CPU_COUNT_OPTION)
        extracted = command_option(cli, UNCAPPED_WORKER_COMMAND, CPU_COUNT_OPTION)

        assert_that(extracted.default).is_equal_to(CORES_ABOVE_CAP)
        assert_that(assigned.default).is_equal_to(EXPECTED_MAX_WORKERS)
        assert_that(assigned.default).is_not_equal_to(extracted.default)

    def test_worker_option_follows_the_cpu_count_option_convention(self) -> None:
        """The option is spelled and rendered as extract-barcodes and split-bam spell it.

        The two names and the shown default are user-facing vocabulary: someone
        who has driven either of the other commands should not have to read the
        help to know how to set the width here.
        """
        option = command_option(carmack.__main__.carmack_cli, COMMAND_NAME, CPU_COUNT_OPTION)

        assert_that(option.opts).is_equal_to(CPU_COUNT_OPTS)
        assert_that(option.show_default).is_true()
        assert_that(option.required).is_false()
        assert_that(option.type.name).is_equal_to("integer")

    @pytest.mark.parametrize("fragment", HELP_FRAGMENTS)
    def test_worker_option_help_states_the_saturation_point_and_its_reason(
        self, fragment: str
    ) -> None:
        """The help says both the number and why it stops there.

        Asserted against the declared help rather than the rendered screen: the
        shown default puts the number on screen whatever the help says, so a
        rendered check alone would pass on a help text that never explained it.

        Args:
            fragment: Wording the option's help must carry.
        """
        option = command_option(carmack.__main__.carmack_cli, COMMAND_NAME, CPU_COUNT_OPTION)

        assert_that(option.help).is_not_none()
        assert_that(option.help.lower()).contains(fragment)

    def test_command_help_renders_the_worker_option_with_its_reason_and_default(self) -> None:
        """The reason survives rendering, so a user reads it off --help."""
        runner = CliRunner()

        result = runner.invoke(carmack.__main__.carmack_cli, [COMMAND_NAME, "--help"])
        rendered = flatten_help(result.output)

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(rendered).contains(*CPU_COUNT_OPTS)
        assert_that(rendered).contains("no faster")
        assert_that(rendered).contains(f"[default: {DEFAULT_CPU_COUNT}]")


@substitute_pool
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
            + stats.unmatched_no_left_anchor_pos
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
