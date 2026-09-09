"""Tests for the prepare-reads read preparer.

`ReadPreparer` dispatches one UMI- and target-annotated R1 read at a time to
either the scRNA (unmatched) arm or a matched scTIP target bucket, tallying
the outcome into a `PrepareCounts` accumulator as it goes. The module under
test does not exist yet, so every test here is expected to fail at collection
or on its very first call -- that failure is the correct state for this
stage of the work.

This file covers construction: the chemistry-derived parameters the
constructor resolves and caches once, and the fail-fast checks that reject a
chemistry it could never dispatch reads for.

It also covers the per-read dispatch body itself -- the unmatched arm, all
three branches of the matched arm's trim-point arithmetic, and the counters
both arms bump -- and the invariant that holds across repeated calls into the
same accumulator.

Finally it covers the two streaming helpers a later driver runs `prepare_read`
inside: pairing an R1 stream with its R2 stream while validating that every
pulled pair's read ids agree, and lazily grouping that paired stream into
batches.
"""

import gzip
import io
import multiprocessing
import os
import re
import signal
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from functools import cached_property
from inspect import signature
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import pytest
import rich_click as click
from assertpy import assert_that
from click.testing import CliRunner

import carmack.__main__
from carmack.assign_targets import target_assigner
from carmack.assign_targets.target_assigner import NO_TARGET
from carmack.chemistry.annotation import format_span
from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import (
    BC_CHUNK_LEN,
    POLYG_BASE,
    POLYG_MIN_RUN,
    PRIMER_A,
    PRIMER_C,
    TGIDX_LENGTH,
    UMI_LENGTH,
    UMI_LENGTH_TOLERANCE,
    ChemistryCarmackCustomSeq10,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.io.subprocess_stream import SubprocessStream
from carmack.parallel import map_batches_in_order
from carmack.prepare_reads import read_preparer
from carmack.prepare_reads.prepare_reporting import PrepareCounts
from carmack.prepare_reads.read_preparer import (
    DEFAULT_MAX_WORKERS,
    MAX_READS_PER_BATCH,
    MatchedOutcome,
    ReadPreparer,
    UnmatchedOutcome,
    iter_paired_reads,
    iter_read_pair_batches,
)
from carmack.prepare_reads.scrna_writer import ScrnaWriter
from carmack.prepare_reads.sctip_writer import (
    SctipBucketWriters,
    render_sctip_header,
    write_sctip_read,
)
from carmack.utils import get_cpu_count, get_prefix
from tests.utils import strip_report_run_details

CHEMISTRY = "carmack_custom_seq_1_0"

# The constructor only wraps these paths in FastqFile objects, it never reads
# them, so any path shaped like a real FASTQ will do.
DUMMY_R1_FASTQ = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"
DUMMY_R2_FASTQ = "tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz"

# Names the synthetic chemistries below are constructed under. Each class
# reports its own name as the same string, so a message naming either the
# requested name or the resolved chemistry satisfies the assertions.
NO_TARGET_INDEX_CHEMISTRY = "prepare_custom_seq_no_target_index"
TWO_TARGETS_CHEMISTRY = "prepare_custom_seq_two_targets"
HOMOPOLYMER_RIGHT_ANCHOR_CHEMISTRY = "prepare_custom_seq_tgidx_homopolymer_right_anchor"
TGIDX_LAST_CHEMISTRY = "prepare_custom_seq_tgidx_last_component"
WITHOUT_UMI_CHEMISTRY = "prepare_custom_seq_without_umi"
EMPTY_WHITELIST_CHEMISTRY = "prepare_custom_seq_empty_target_whitelist"

# Two entries that each begin with zero copies of the POLYG anchor base, so
# both clear the leading-anchor bound the base chemistry's whitelist
# validation enforces at construction.
TWO_TARGET_WHITELIST = ("TATAGCCT", "CATTGGAC")

# The homopolymer base of the synthetic right-anchor chemistry's anchor,
# deliberately different from POLYG_BASE so a test mixing the two up would be
# caught by a wrong trim point rather than passing by coincidence.
RIGHT_ANCHOR_BASE = "T"

# The single entry the shipped custom_seq target whitelist loads. Reused here
# as an arbitrary matched target value: `prepare_read` never checks a target
# value against any whitelist, it only relays whatever the header already
# carries, so this need not be realistic beyond being a plausible string.
TGIDX_VALUE = "TATAGCCT"

# Arbitrary, mutually distinguishable barcode and UMI tag values shared by
# every matched-arm test read below.
DEFAULT_BARCODES = {"BC3": "GGGGGCCCCC", "BC2": "TTTTTAAAAA", "BC1": "CCCCCGGGGG"}
DEFAULT_UMI = "ACGTACGT"

# Marker sequence planted immediately after the computed trim point in every
# matched-arm test read, so a wrong cut is caught by its content, not only by
# its length.
INSERT_SEQ = "ACGTACGTAC"


def make_qual(length: int) -> str:
    """Return a quality string whose characters vary by position.

    A uniform quality string could not distinguish a correct trim from an
    off-by-one one, since every slice of it looks identical to every other
    slice of the same length.

    Args:
        length: Number of characters to generate.

    Returns:
        A deterministic, position-varying quality string of the given length.
    """
    return "".join(chr(33 + (i % 40)) for i in range(length))


def build_read(
    read_id: str, tags: dict[str, str], seq: str, qual: str | None = None
) -> tuple[str, str, str]:
    """Build a `(name, seq, qual)` read carrying the given header tags verbatim.

    Args:
        read_id: Read identifier, the header's first token.
        tags: Header tags to set, in insertion order.
        seq: The full R1 sequence.
        qual: The full R1 quality string. Defaults to a position-varying
            string of the same length as `seq`.

    Returns:
        The rendered header, the sequence, and the quality string.
    """
    ann = ReadAnnotation(read_id=read_id)
    for key, value in tags.items():
        ann.set(key, value)
    return ann.render(), seq, qual if qual is not None else make_qual(len(seq))


def build_matched_read(
    read_id: str,
    tgidx_value: str,
    tgidx_pos_end: int,
    after_tgidx: str,
    tgidx_pos_start: int | None = None,
    barcodes: dict[str, str] | None = None,
    umi: str = DEFAULT_UMI,
) -> tuple[str, str, str]:
    """Build a matched `(name, seq, qual)` read whose TGIDX span ends at `tgidx_pos_end`.

    `prepare_read` never reads barcode, UMI, or target sequence content from
    the read body -- only from header tags a chemistry-agnostic upstream stage
    already wrote -- so only two things about `seq` matter to it: its length
    up to `tgidx_pos_end`, and the content of `after_tgidx`, which is exactly
    what `insert_start` inspects to find the trim point. Everything before
    `tgidx_pos_end` is therefore arbitrary filler.

    Args:
        read_id: Read identifier, the header's first token.
        tgidx_value: Value written to the TGIDX tag.
        tgidx_pos_end: End coordinate of the TGIDX_POS span -- the
            `reference` coordinate `insert_start` receives.
        after_tgidx: Sequence appended starting exactly at `tgidx_pos_end`,
            shaped per branch under test: a fixed-length anchor's filler plus
            the insert, a homopolymer run plus the insert, or the insert
            alone.
        tgidx_pos_start: Start coordinate of the TGIDX_POS span. Defaults to
            `TGIDX_LENGTH` bases before the end; never read by `insert_start`.
        barcodes: Barcode tag values, defaulting to `DEFAULT_BARCODES`.
        umi: UMI tag value.

    Returns:
        The rendered header, the sequence, and a position-varying quality
        string of the same length.
    """
    start = tgidx_pos_start if tgidx_pos_start is not None else tgidx_pos_end - TGIDX_LENGTH
    seq = "A" * tgidx_pos_end + after_tgidx
    ann = ReadAnnotation(read_id=read_id)
    for key, value in {**(barcodes or DEFAULT_BARCODES), "UMI": umi}.items():
        ann.set(key, value)
    ann.set("TGIDX", tgidx_value)
    ann.set("TGIDX_POS", format_span(start, tgidx_pos_end))
    return ann.render(), seq, make_qual(len(seq))


def build_head_components() -> list[ReadComponent]:
    """Return the BC3/PRIMER_C/BC2/PRIMER_A/BC1/UMI head shared by every synthetic chemistry.

    Every synthetic chemistry below varies only what follows the UMI, so the
    barcode and UMI layout -- and therefore `DEFAULT_BARCODES`'s keys and
    `DEFAULT_UMI`'s tag name -- stays identical across all of them.

    Returns:
        The head components, in read order.
    """
    return [
        ReadComponent(name="BC3", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
        ReadComponent(
            name="PRIMER_C", type=ReadComponentType.PRIMER, length=len(PRIMER_C), sequence=PRIMER_C
        ),
        ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
        ReadComponent(
            name="PRIMER_A", type=ReadComponentType.PRIMER, length=len(PRIMER_A), sequence=PRIMER_A
        ),
        ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
        ReadComponent(
            name="UMI",
            type=ReadComponentType.UMI,
            length=UMI_LENGTH,
            length_tolerance=UMI_LENGTH_TOLERANCE,
        ),
    ]


def build_polyg_component() -> ReadComponent:
    """Return the POLYG anchor that must precede TGIDX for target assignment to be supported."""
    return ReadComponent(
        name="POLYG",
        type=ReadComponentType.HOMOPOLYMER,
        homopolymer_base=POLYG_BASE,
        min_run=POLYG_MIN_RUN,
    )


class ChemistryNoTargetIndex(ChemistryCarmackCustomSeq10):
    """A chemistry with a UMI but no target index component at all.

    Pins the one relationship `ReadPreparer` depends on to skip a header
    lookup entirely: a chemistry with `supports_target_assignment() == False`
    must never even look for a TGIDX tag, because `tgidx_key` is `None` for
    it rather than a name nothing will ever be found under.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return NO_TARGET_INDEX_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout carrying the UMI but nothing following it."""
        return ReadStructure(build_head_components())


class ChemistryTwoTargets(ChemistryCarmackCustomSeq10):
    """Shipped chemistry layout whose target index whitelist carries two entries.

    Exists to prove a multi-target chemistry constructs cleanly; a later
    sub-task's multi-bucket driver tests may reuse it.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return TWO_TARGETS_CHEMISTRY

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return two whitelist entries instead of the shipped single one."""
        return TWO_TARGET_WHITELIST


class ChemistryTgidxHomopolymerRightAnchor(ChemistryCarmackCustomSeq10):
    """A chemistry whose target index is immediately followed by a homopolymer run.

    Exercises `insert_start`'s homopolymer branch: `tgidx_right_anchor()`
    resolves to a HOMOPOLYMER component, so the matched-arm trim point has to
    be read off the read itself rather than computed from a fixed anchor
    length.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return HOMOPOLYMER_RIGHT_ANCHOR_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout whose target index is followed by a homopolymer run."""
        return ReadStructure(
            [
                *build_head_components(),
                build_polyg_component(),
                ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
                ReadComponent(
                    name="POLYT",
                    type=ReadComponentType.HOMOPOLYMER,
                    homopolymer_base=RIGHT_ANCHOR_BASE,
                    min_run=POLYG_MIN_RUN,
                ),
            ]
        )


class ChemistryTgidxLastComponent(ChemistryCarmackCustomSeq10):
    """A chemistry whose target index is the final component in the read structure.

    Exercises `insert_start`'s `anchor is None` passthrough branch: there is
    no adapter between the target index and the insert, so the trim point is
    exactly the target index span's own end.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return TGIDX_LAST_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout ending on the target index, with nothing after it."""
        return ReadStructure(
            [
                *build_head_components(),
                build_polyg_component(),
                ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
            ]
        )


class ChemistryWithoutUmi(ChemistryCarmackCustomSeq10):
    """A chemistry declaring a target index but no UMI component.

    `ReadPreparer` requires `supports_umi_extraction()`, so this chemistry
    must be rejected at construction regardless of its otherwise-valid target
    index declaration.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return WITHOUT_UMI_CHEMISTRY

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Return a layout carrying the anchor and index but no UMI component."""
        return ReadStructure(
            [
                ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
                build_polyg_component(),
                ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
            ]
        )


class ChemistryEmptyTargetWhitelist(ChemistryCarmackCustomSeq10):
    """Shipped chemistry whose target index whitelist loads no entries.

    `supports_target_assignment()` only checks for a homopolymer-anchored
    target index and knows nothing about the whitelist, so this chemistry
    answers `True` while no read could ever be validly matched against it --
    the gap `ReadPreparer.__init__` has to close for itself.
    """

    @cached_property
    def name(self) -> str:
        """Return the identifier this chemistry is requested under."""
        return EMPTY_WHITELIST_CHEMISTRY

    def tgidx_whitelist(self) -> tuple[str, ...]:
        """Return no whitelist entries while keeping the shipped read structure."""
        return ()


def patch_chemistry(chemistry: ChemistryBase):
    """Return a patcher making the factory answer with the supplied chemistry.

    Mirrors the equivalent helper in the assign-targets test suite: it lets a
    synthetic chemistry be resolved by name without registering it with the
    real, process-wide `ChemistryFactory`.

    Args:
        chemistry: The chemistry instance the preparer should resolve.

    Returns:
        An unstarted `unittest.mock.patch` context manager.
    """
    return mock.patch(
        "carmack.prepare_reads.read_preparer.ChemistryFactory.get_chemistry",
        return_value=chemistry,
    )


def build_preparer(chemistry: ChemistryBase | None = None, **kwargs: int | None) -> ReadPreparer:
    """Return a `ReadPreparer` built without ever touching the filesystem.

    The constructor only wraps its FASTQ paths, so -- unlike `TargetAssigner`'s
    equivalent tests, which stream a real file and so must write one first --
    no read record needs to exist on disk for these tests.

    Args:
        chemistry: A chemistry instance the factory should be patched to
            resolve, or `None` to resolve the real shipped chemistry.
        **kwargs: Forwarded to the constructor (`n_workers`, `batch_size`).

    Returns:
        The constructed `ReadPreparer`.
    """
    if chemistry is None:
        return ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, CHEMISTRY, **kwargs)
    with patch_chemistry(chemistry):
        return ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, chemistry.name, **kwargs)


class TestReadPreparerConstruction:
    """Parameter resolution the constructor performs once, up front."""

    def test_wraps_the_input_paths_and_resolves_the_chemistry(self) -> None:
        preparer = ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, CHEMISTRY)

        assert_that(preparer.r1_fastq).is_instance_of(FastqFile)
        assert_that(preparer.r1_fastq.filename).is_equal_to(DUMMY_R1_FASTQ)
        assert_that(preparer.r2_fastq).is_instance_of(FastqFile)
        assert_that(preparer.r2_fastq.filename).is_equal_to(DUMMY_R2_FASTQ)
        assert_that(preparer.chemistry_name).is_equal_to(CHEMISTRY)
        assert_that(preparer.chemistry.name).is_equal_to(CHEMISTRY)

    def test_caches_the_tgidx_key_right_anchor_umi_name_and_barcode_names(self) -> None:
        """Pin the four chemistry-derived attributes a later driver reads directly.

        Restated as literals, not only derived from the chemistry, so a
        rename of any of them is caught here rather than only downstream.
        Also pins that the cached right anchor is the exact object the
        chemistry itself would return, not a value copied out of it, since it
        must never be recomputed per read.
        """
        preparer = ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, CHEMISTRY)

        assert_that(preparer.tgidx_key).is_equal_to("TGIDX")
        assert_that(preparer.umi_name).is_equal_to("UMI")
        assert_that(preparer.barcode_names).is_equal_to(["BC3", "BC2", "BC1"])
        assert_that(preparer.tgidx_right_anchor).is_same_as(
            preparer.chemistry.tgidx_right_anchor()
        )
        assert_that(preparer.tgidx_right_anchor.name).is_equal_to("ME")

    def test_tgidx_key_and_right_anchor_are_none_without_target_assignment_support(self) -> None:
        """A chemistry with no target index never even names a tag to look for.

        `tgidx_key` being `None` -- not merely an empty string, and not the
        literal `"TGIDX"` on a chemistry that happens to carry no such tag --
        is what lets `prepare_read` skip the header lookup outright.
        """
        chemistry = ChemistryNoTargetIndex()
        preparer = build_preparer(chemistry=chemistry)

        assert_that(chemistry.supports_target_assignment()).is_false()
        assert_that(preparer.tgidx_key).is_none()
        assert_that(preparer.tgidx_right_anchor).is_none()

    def test_two_target_whitelist_chemistry_constructs_and_resolves_tgidx_key(self) -> None:
        """A chemistry with more than one whitelisted target still constructs cleanly."""
        chemistry = ChemistryTwoTargets()

        assert_that(chemistry.tgidx_whitelist()).is_length(2)

        preparer = build_preparer(chemistry=chemistry)

        assert_that(preparer.tgidx_key).is_equal_to("TGIDX")

    def test_n_workers_and_batch_size_default(self) -> None:
        preparer = ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, CHEMISTRY)

        assert_that(preparer.n_workers).is_equal_to(1)
        assert_that(preparer.batch_size).is_equal_to(MAX_READS_PER_BATCH)

    def test_n_workers_and_batch_size_pass_through(self) -> None:
        preparer = ReadPreparer(
            DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, CHEMISTRY, n_workers=4, batch_size=10
        )

        assert_that(preparer.n_workers).is_equal_to(4)
        assert_that(preparer.batch_size).is_equal_to(10)

    def test_max_reads_per_batch_and_default_max_workers_are_committed_constants(self) -> None:
        """Pin the two module constants restated from assign-targets, not derived from it.

        `DEFAULT_MAX_WORKERS` is explicitly provisional here -- mirrored from
        assign-targets's measured saturation point rather than independently
        measured for this stage -- so this only pins the value carried over,
        not a claim that it has been re-measured.
        """
        assert_that(MAX_READS_PER_BATCH).is_equal_to(2500)
        assert_that(DEFAULT_MAX_WORKERS).is_equal_to(16)

    def test_no_target_sentinel_is_imported_by_identity_from_target_assigner(self) -> None:
        """`NO_TARGET` must be the exact object assign-targets defines, not a restated copy.

        A future refactor that let the two modules' sentinels drift apart
        would corrupt every downstream consumer that keys on the literal
        silently, since the two strings would still compare equal.
        """
        assert_that(read_preparer.NO_TARGET).is_same_as(target_assigner.NO_TARGET)


class TestReadPreparerConstructionValidation:
    """The fail-fast checks that reject a chemistry this stage could never dispatch reads for."""

    def test_chemistry_without_umi_support_raises(self) -> None:
        chemistry = ChemistryWithoutUmi()

        with patch_chemistry(chemistry), pytest.raises(ValueError, match=WITHOUT_UMI_CHEMISTRY):
            ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, WITHOUT_UMI_CHEMISTRY)

    def test_empty_target_whitelist_raises_though_support_check_passes(self) -> None:
        """Pin that the constructor, not the chemistry accessor, closes the empty-whitelist gap.

        `ChemistryBase.supports_target_assignment` asks only whether the
        target index is preceded by a homopolymer anchor, so a chemistry
        whose whitelist loads no entries still answers `True`. No read could
        ever be validly dispatched from it, so rejecting it falls to the
        constructor.
        """
        chemistry = ChemistryEmptyTargetWhitelist()

        assert_that(chemistry.supports_target_assignment()).is_true()
        assert_that(chemistry.tgidx_whitelist()).is_empty()

        with (
            patch_chemistry(chemistry),
            pytest.raises(ValueError, match=EMPTY_WHITELIST_CHEMISTRY),
        ):
            ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, EMPTY_WHITELIST_CHEMISTRY)

    def test_chemistry_without_target_assignment_constructs_with_no_whitelist_check(self) -> None:
        """A chemistry declaring no target index at all has nothing to check a whitelist for.

        `ChemistryNoTargetIndex.tgidx_whitelist()` is empty too, but that
        must never be mistaken for the empty-whitelist failure above: the
        check that raises is gated on `supports_target_assignment()`, so a
        chemistry that never claims to support target assignment in the
        first place constructs successfully regardless of what its whitelist
        holds.
        """
        chemistry = ChemistryNoTargetIndex()

        assert_that(chemistry.supports_target_assignment()).is_false()
        assert_that(chemistry.tgidx_whitelist()).is_empty()

        preparer = build_preparer(chemistry=chemistry)

        assert_that(preparer.tgidx_key).is_none()

    def test_unknown_chemistry_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            ReadPreparer(DUMMY_R1_FASTQ, DUMMY_R2_FASTQ, "does_not_exist")


class TestPrepareReadUnmatched:
    """The scRNA (unmatched) arm: every path that never reaches a real target value."""

    def test_read_with_no_tgidx_tag_is_unmatched(self) -> None:
        name, seq, qual = build_read("notag", {"UMI": DEFAULT_UMI}, "ACGTACGTAC" * 4)
        preparer = build_preparer()
        counts = PrepareCounts()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        assert_that(outcome).is_instance_of(UnmatchedOutcome)
        assert_that(outcome.ann).is_equal_to(ReadAnnotation.parse(name))
        assert_that(outcome.r1_seq).is_equal_to(seq)
        assert_that(outcome.r1_qual).is_equal_to(qual)
        assert_that(counts.total).is_equal_to(1)
        assert_that(counts.unmatched).is_equal_to(1)
        assert_that(counts.target_counts).is_empty()

    def test_read_with_tgidx_none_is_unmatched(self) -> None:
        name, seq, qual = build_read(
            "nonetag", {"UMI": DEFAULT_UMI, "TGIDX": NO_TARGET}, "ACGTACGTAC" * 4
        )
        preparer = build_preparer()
        counts = PrepareCounts()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        assert_that(outcome).is_instance_of(UnmatchedOutcome)
        assert_that(outcome.r1_seq).is_equal_to(seq)
        assert_that(outcome.r1_qual).is_equal_to(qual)
        assert_that(counts.total).is_equal_to(1)
        assert_that(counts.unmatched).is_equal_to(1)
        assert_that(counts.target_counts).is_empty()

    def test_read_on_chemistry_without_target_assignment_ignores_a_stray_tgidx_tag(self) -> None:
        """A `TGIDX=` tag is never even looked up when `tgidx_key` is `None`.

        Proves the dispatch is gated on `self.tgidx_key` itself, not merely
        on whether the tag happens to be absent from a given read: a
        chemistry with no target index must route every read to the
        unmatched arm even when the read carries a value that would
        otherwise look like a real target.
        """
        name, seq, qual = build_read(
            "strayta", {"UMI": DEFAULT_UMI, "TGIDX": "SHOULDBEIGNORED"}, "ACGTACGTAC" * 4
        )
        preparer = build_preparer(chemistry=ChemistryNoTargetIndex())
        counts = PrepareCounts()

        assert_that(preparer.tgidx_key).is_none()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        assert_that(outcome).is_instance_of(UnmatchedOutcome)
        assert_that(outcome.ann.get("TGIDX")).is_equal_to("SHOULDBEIGNORED")
        assert_that(outcome.r1_seq).is_equal_to(seq)
        assert_that(counts.total).is_equal_to(1)
        assert_that(counts.unmatched).is_equal_to(1)
        assert_that(counts.target_counts).is_empty()


class TestPrepareReadMatched:
    """The matched arm across all three of `insert_start`'s dispatch branches.

    Each test drives a different chemistry so that `tgidx_right_anchor`
    resolves to a different kind of anchor -- a fixed-length primer, a
    homopolymer run, or none at all -- and checks that `prepare_read` slices
    at exactly the trim point `insert_start` would compute for that anchor,
    never a hardcoded offset.
    """

    @staticmethod
    def expected_header(preparer: ReadPreparer, name: str) -> str:
        """Render the header independently of the implementation under test.

        Built from the read's own tags and the preparer's own chemistry
        rather than a copied literal, so the assertion cannot pass by a
        hardcoded string drifting in step with a header-rendering change.

        Args:
            preparer: The preparer whose chemistry and UMI name to render with.
            name: The read's rendered header, carrying the tags to read back.

        Returns:
            The header `render_sctip_header` would produce for this read.
        """
        chemistry = preparer.chemistry
        ann = ReadAnnotation.parse(name)
        barcode_components = chemistry.read_structure.get_components_by_type(
            ReadComponentType.BARCODE
        )
        barcodes = {comp.name: ann.get(comp.name) for comp in barcode_components}
        umi = ann.get(preparer.umi_name)
        return render_sctip_header(chemistry, ann.read_id, barcodes, umi)

    def test_fixed_length_right_anchor_trims_past_the_anchors_own_length(self) -> None:
        """The real chemistry's `ME` primer is a fixed-length right anchor.

        `carmack_custom_seq_1_0` never exercises the homopolymer or
        `None`-anchor branches of `insert_start` itself, so this is the one
        real-chemistry case among the three matched-arm tests; the other two
        need a synthetic chemistry.
        """
        preparer = build_preparer()
        anchor = preparer.tgidx_right_anchor
        tgidx_pos_end = 40
        name, seq, qual = build_matched_read(
            "matched", TGIDX_VALUE, tgidx_pos_end, "N" * anchor.length + INSERT_SEQ
        )
        counts = PrepareCounts()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        cut = tgidx_pos_end + anchor.length
        assert_that(outcome).is_instance_of(MatchedOutcome)
        assert_that(outcome.tgidx).is_equal_to(TGIDX_VALUE)
        assert_that(outcome.r1_seq).is_equal_to(seq[cut:])
        assert_that(outcome.r1_seq).is_equal_to(INSERT_SEQ)
        assert_that(outcome.r1_qual).is_equal_to(qual[cut:])
        assert_that(outcome.header).is_equal_to(self.expected_header(preparer, name))
        assert_that(counts.total).is_equal_to(1)
        assert_that(dict(counts.target_counts)).is_equal_to({TGIDX_VALUE: 1})
        assert_that(counts.unmatched).is_equal_to(0)

    def test_homopolymer_right_anchor_trims_at_the_observed_run_end(self) -> None:
        """A HOMOPOLYMER right anchor's real extent is read off the read, not off the chemistry.

        The planted run length is deliberately not the anchor's own
        `min_run`, so a wrongly hardcoded `min_run` in place of a genuine
        forward scan would fail this test rather than pass by coincidence.
        """
        run_length = 5
        tgidx_pos_end = 30
        name, seq, qual = build_matched_read(
            "homopolymer", TGIDX_VALUE, tgidx_pos_end, RIGHT_ANCHOR_BASE * run_length + INSERT_SEQ
        )
        preparer = build_preparer(chemistry=ChemistryTgidxHomopolymerRightAnchor())
        counts = PrepareCounts()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        cut = tgidx_pos_end + run_length
        assert_that(outcome).is_instance_of(MatchedOutcome)
        assert_that(outcome.tgidx).is_equal_to(TGIDX_VALUE)
        assert_that(outcome.r1_seq).is_equal_to(seq[cut:])
        assert_that(outcome.r1_seq).is_equal_to(INSERT_SEQ)
        assert_that(outcome.r1_qual).is_equal_to(qual[cut:])
        assert_that(outcome.header).is_equal_to(self.expected_header(preparer, name))
        assert_that(dict(counts.target_counts)).is_equal_to({TGIDX_VALUE: 1})
        assert_that(counts.unmatched).is_equal_to(0)

    def test_none_right_anchor_trims_at_the_target_index_span_end(self) -> None:
        """No adapter between the target index and the insert means no offset at all."""
        tgidx_pos_end = 30
        name, seq, qual = build_matched_read(
            "lastcomponent", TGIDX_VALUE, tgidx_pos_end, INSERT_SEQ
        )
        preparer = build_preparer(chemistry=ChemistryTgidxLastComponent())
        counts = PrepareCounts()

        assert_that(preparer.tgidx_right_anchor).is_none()

        outcome = preparer.prepare_read(name, seq, qual, counts)

        assert_that(outcome).is_instance_of(MatchedOutcome)
        assert_that(outcome.tgidx).is_equal_to(TGIDX_VALUE)
        assert_that(outcome.r1_seq).is_equal_to(seq[tgidx_pos_end:])
        assert_that(outcome.r1_seq).is_equal_to(INSERT_SEQ)
        assert_that(outcome.r1_qual).is_equal_to(qual[tgidx_pos_end:])
        assert_that(outcome.header).is_equal_to(self.expected_header(preparer, name))
        assert_that(dict(counts.target_counts)).is_equal_to({TGIDX_VALUE: 1})
        assert_that(counts.unmatched).is_equal_to(0)

    def test_matched_read_missing_tgidx_pos_raises(self) -> None:
        """A matched read with no span for its target index is a corrupt input, not an outcome.

        Unlike an unmatched read, which is a normal and expected result, a
        read that carries a real target value but no position span could
        never have come from a correctly run assign-targets stage, so this
        has to be fatal rather than tallied into any counter.
        """
        name, seq, qual = build_read(
            "nospan", {**DEFAULT_BARCODES, "UMI": DEFAULT_UMI, "TGIDX": TGIDX_VALUE}, "A" * 40
        )
        preparer = build_preparer()

        with pytest.raises(ValueError) as excinfo:
            preparer.prepare_read(name, seq, qual, PrepareCounts())

        assert_that(str(excinfo.value)).contains("nospan", "TGIDX_POS")

    def test_matched_read_missing_umi_raises(self) -> None:
        """A matched read with no UMI tag is a corrupt input, not an outcome.

        Mirrors `test_matched_read_missing_tgidx_pos_raises`: a read carrying a
        real target value and its `TGIDX_POS` span but no `UMI` tag at all could
        never have come from a correctly run extract-umis/assign-targets chain,
        so this has to be fatal rather than silently rendering a `UR=None`
        header.
        """
        tags = {
            **DEFAULT_BARCODES,
            "TGIDX": TGIDX_VALUE,
            "TGIDX_POS": format_span(20, 20 + TGIDX_LENGTH),
        }
        name, seq, qual = build_read("noumi", tags, "A" * 40)
        preparer = build_preparer()

        with pytest.raises(ValueError) as excinfo:
            preparer.prepare_read(name, seq, qual, PrepareCounts())

        assert_that(str(excinfo.value)).contains("noumi", "UMI")


class TestPrepareReadCountsAccumulate:
    """Repeated calls fold into the same accumulator instead of resetting it."""

    def test_prepare_read_accumulates_into_the_counts_it_is_handed(self) -> None:
        """The body mutates the accumulator it is given rather than returning a fresh one.

        Mirrors `TargetAssigner.assign_read`'s equivalent guarantee, which is
        what keeps a production run from allocating one outcome-tally object
        per read.
        """
        preparer = build_preparer()
        anchor = preparer.tgidx_right_anchor
        name, seq, qual = build_matched_read(
            "matched", TGIDX_VALUE, 30, "N" * anchor.length + INSERT_SEQ
        )
        counts = PrepareCounts()
        calls = 3

        for _ in range(calls):
            preparer.prepare_read(name, seq, qual, counts)

        assert_that(counts.total).is_equal_to(calls)
        assert_that(dict(counts.target_counts)).is_equal_to({TGIDX_VALUE: calls})
        assert_that(counts.unmatched).is_equal_to(0)

    def test_outcome_invariant_holds_after_every_call_across_a_mix_of_outcomes(self) -> None:
        """`unmatched + sum(target_counts.values()) == total` after every single call.

        Checked call by call rather than once at the end: a read tallied
        twice and a read never tallied cancel out in a final total while
        leaving the invariant broken after either of the two calls that
        caused it.
        """
        preparer = build_preparer()
        anchor = preparer.tgidx_right_anchor
        reads = [
            build_read("u1", {}, "A" * 20),
            build_matched_read("m1", TGIDX_VALUE, 30, "N" * anchor.length + INSERT_SEQ),
            build_read("u2", {"TGIDX": NO_TARGET}, "A" * 20),
            build_matched_read("m2", TGIDX_VALUE, 30, "N" * anchor.length + INSERT_SEQ),
        ]
        counts = PrepareCounts()

        for expected_total, (name, seq, qual) in enumerate(reads, start=1):
            preparer.prepare_read(name, seq, qual, counts)

            assert_that(counts.total).is_equal_to(expected_total)
            assert_that(counts.unmatched + sum(counts.target_counts.values())).is_equal_to(
                counts.total
            )


class CountingReads:
    """A read (or pair) source that counts how many items have been pulled out of it.

    Mirrors the equivalent helper in the assign-targets test suite, one level
    up: here it is single reads or paired-read tuples being counted, whichever
    the batcher or pairer under test is pulling from.
    """

    def __init__(self, items) -> None:
        """Wrap a sequence of items.

        Args:
            items: Items to hand out one at a time.
        """
        self.items = items
        self.pulled = 0

    def __iter__(self):
        """Yield items, counting each one as it leaves.

        Yields:
            The next item.
        """
        for item in self.items:
            self.pulled += 1
            yield item


def make_paired_reads(count: int) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Return `count` distinctly identified, matching-id R1/R2 read lists.

    Args:
        count: Number of read pairs to build.

    Returns:
        The R1 reads and the R2 reads, in index order, each read id shared
        between its R1 and R2 half but the remainder of the header varying,
        the way real sequencer output does.
    """
    r1_reads = [(f"read{i} 1:N:0", f"R1SEQ{i}", f"R1QUAL{i}") for i in range(count)]
    r2_reads = [(f"read{i} 2:N:0", f"R2SEQ{i}", f"R2QUAL{i}") for i in range(count)]
    return r1_reads, r2_reads


class TestIterPairedReads:
    """Zipping an R1 stream with its R2 stream while validating every pulled pair."""

    def test_matching_pairs_stream_through_unchanged_and_in_order(self) -> None:
        r1_reads, r2_reads = make_paired_reads(4)

        pairs = list(iter_paired_reads(iter(r1_reads), iter(r2_reads)))

        assert_that(pairs).is_equal_to(list(zip(r1_reads, r2_reads)))

    def test_mismatch_on_the_first_pair_raises_before_pulling_anything_further(self) -> None:
        """No read beyond the first mismatched pair is ever pulled from either source.

        Proven with a counting wrapper rather than taken on trust: a
        generator that validated eagerly across the whole stream before
        yielding anything would defeat the laziness this function promises.
        """
        r1_reads = CountingReads([("bad1 1:N", "AAAA", "IIII"), ("read1 1:N", "CCCC", "IIII")])
        r2_reads = CountingReads([("bad2 2:N", "TTTT", "IIII"), ("read1 2:N", "GGGG", "IIII")])

        with pytest.raises(ValueError):
            list(iter_paired_reads(r1_reads, r2_reads))

        assert_that(r1_reads.pulled).is_equal_to(1)
        assert_that(r2_reads.pulled).is_equal_to(1)

    def test_mismatch_after_several_matches_raises_at_exactly_that_point(self) -> None:
        """Pairs already yielded before a later mismatch are unaffected by it."""
        r1_reads = [
            ("read0 1:N", "A0", "I0"),
            ("read1 1:N", "A1", "I1"),
            ("mismatched1 1:N", "A2", "I2"),
        ]
        r2_reads = [
            ("read0 2:N", "T0", "J0"),
            ("read1 2:N", "T1", "J1"),
            ("mismatched2 2:N", "T2", "J2"),
        ]

        yielded = []
        with pytest.raises(ValueError):
            for pair in iter_paired_reads(iter(r1_reads), iter(r2_reads)):
                yielded.append(pair)

        assert_that(yielded).is_equal_to(list(zip(r1_reads[:2], r2_reads[:2])))

    @pytest.mark.parametrize(
        "r1_reads,r2_reads",
        [
            (
                [("a 1:N", "A", "I"), ("b 1:N", "A", "I"), ("c 1:N", "A", "I")],
                [("a 2:N", "T", "I"), ("b 2:N", "T", "I")],
            ),
            (
                [("a 1:N", "A", "I"), ("b 1:N", "A", "I")],
                [("a 2:N", "T", "I"), ("b 2:N", "T", "I"), ("c 2:N", "T", "I")],
            ),
        ],
        ids=["r1_longer", "r2_longer"],
    )
    def test_length_mismatch_raises(
        self, r1_reads: list[tuple[str, str, str]], r2_reads: list[tuple[str, str, str]]
    ) -> None:
        with pytest.raises(ValueError):
            list(iter_paired_reads(iter(r1_reads), iter(r2_reads)))

    def test_both_empty_yields_nothing(self) -> None:
        assert_that(list(iter_paired_reads(iter([]), iter([])))).is_empty()

    def test_mismatch_error_names_both_read_ids(self) -> None:
        r1_reads = [("alpha 1:N", "AAAA", "IIII")]
        r2_reads = [("beta 2:N", "TTTT", "IIII")]

        with pytest.raises(ValueError) as excinfo:
            list(iter_paired_reads(iter(r1_reads), iter(r2_reads)))

        assert_that(str(excinfo.value)).contains("alpha", "beta")


# Batch shape the batcher tests drive: small enough that the synthetic input
# stays short, several batches deep so laziness is observable across it.
BATCH_SIZE = 3
FULL_BATCHES = 3


def make_pairs(count: int) -> list[tuple[tuple[str, str, str], tuple[str, str, str]]]:
    """Return `count` distinctly identified synthetic `(r1_read, r2_read)` pairs.

    Args:
        count: Number of pairs to build.

    Returns:
        One pair per index, in index order.
    """
    r1_reads, r2_reads = make_paired_reads(count)
    return list(zip(r1_reads, r2_reads))


class TestIterReadPairBatches:
    """Lazily grouping a paired-read stream into same-length `(r1_batch, r2_batch)` batches."""

    def test_full_batches_then_a_short_final_batch_correspond_index_for_index(self) -> None:
        pairs = make_pairs(BATCH_SIZE * FULL_BATCHES + 1)

        batches = list(iter_read_pair_batches(iter(pairs), BATCH_SIZE))

        r1_lengths = [len(r1_batch) for r1_batch, _ in batches]
        r2_lengths = [len(r2_batch) for _, r2_batch in batches]
        assert_that(r1_lengths).is_equal_to([BATCH_SIZE] * FULL_BATCHES + [1])
        assert_that(r2_lengths).is_equal_to(r1_lengths)

        flattened = [
            (r1, r2) for r1_batch, r2_batch in batches for r1, r2 in zip(r1_batch, r2_batch)
        ]
        assert_that(flattened).is_equal_to(pairs)

    def test_empty_input_yields_nothing(self) -> None:
        assert_that(list(iter_read_pair_batches(iter([]), BATCH_SIZE))).is_empty()

    def test_exact_multiple_yields_no_trailing_empty_batch(self) -> None:
        pairs = make_pairs(BATCH_SIZE * FULL_BATCHES)

        batches = list(iter_read_pair_batches(iter(pairs), BATCH_SIZE))

        assert_that(batches).is_length(FULL_BATCHES)
        assert_that([len(r1_batch) for r1_batch, _ in batches]).is_equal_to(
            [BATCH_SIZE] * FULL_BATCHES
        )

    def test_pulls_only_one_batch_of_pairs_before_yielding_it(self) -> None:
        source = CountingReads(make_pairs(BATCH_SIZE * FULL_BATCHES + 1))

        r1_batch, _ = next(iter_read_pair_batches(source, BATCH_SIZE))

        assert_that(r1_batch).is_length(BATCH_SIZE)
        assert_that(source.pulled).is_equal_to(BATCH_SIZE)

    def test_never_reads_more_than_one_batch_ahead(self) -> None:
        pairs = make_pairs(BATCH_SIZE * FULL_BATCHES + 1)
        source = CountingReads(pairs)

        batched = 0
        for r1_batch, _ in iter_read_pair_batches(source, BATCH_SIZE):
            batched += len(r1_batch)
            assert_that(source.pulled - batched).is_less_than_or_equal_to(BATCH_SIZE)

        assert_that(batched).is_equal_to(len(pairs))
        assert_that(source.pulled).is_equal_to(len(pairs))


# ---------------------------------------------------------------------------
# prepare_reads(): whole-stage behavioural tests, pool wiring, module worker
# plumbing and the r1_batches_with_r2_sidecar helper.
#
# None of `ReadPreparer.prepare_reads`, `read_preparer.WORKER_PREPARER`,
# `read_preparer.init_prepare_worker`, `read_preparer.prepare_read_batch` or
# `read_preparer.r1_batches_with_r2_sidecar` exist yet, so every test below is
# expected to fail -- most with an AttributeError raised the moment the
# missing attribute is looked up, since importing them by name at module
# level here would break collection of the whole file, including the tests
# above this line. That failure is the correct, expected state for this
# stage of the work.
# ---------------------------------------------------------------------------

OUT_PREFIX = "out"


def build_full_r1_read(
    read_id: str,
    tgidx: str | None,
    polyg_run_length: int,
    after_polyg: str,
    barcodes: dict[str, str] | None = None,
    umi: str = DEFAULT_UMI,
) -> tuple[str, str, str]:
    """Build a fully positioned R1 read, as a real extract-umis run (plus,
    when `tgidx` is given, a real assign-targets run) would emit it.

    Unlike `build_read`/`build_matched_read` above -- which set only the
    header tags `prepare_read` itself reads -- this builder also writes each
    barcode's and the UMI's own `*_POS` span, since a whole `prepare_reads()`
    run additionally exercises `ScrnaWriter`, which slices the synthesized
    barcodes FASTQ off those spans rather than off the tag values.

    Args:
        read_id: Read identifier, the header's first token.
        tgidx: Value written to the TGIDX tag, or `None` to omit the tag
            entirely -- the shape a chemistry with no target index support
            emits, as opposed to an explicit `NO_TARGET` value.
        polyg_run_length: Number of `POLYG_BASE` bases written immediately
            after the UMI, standing in for the observed anchor run. Zero for
            a chemistry whose read structure carries no such component.
        after_polyg: Sequence written immediately after the POLYG run -- the
            insert alone for an unmatched read, or a TGIDX-length filler plus
            a right-anchor-length filler plus the insert for a matched one.
        barcodes: Barcode tag values, defaulting to `DEFAULT_BARCODES`.
        umi: UMI tag value.

    Returns:
        The rendered header, the assembled sequence, and a position-varying
        quality string of the same length.
    """
    bc = barcodes if barcodes is not None else DEFAULT_BARCODES
    bc3, bc2, bc1 = bc["BC3"], bc["BC2"], bc["BC1"]

    bc3_start, bc3_end = 0, len(bc3)
    bc2_start, bc2_end = bc3_end, bc3_end + len(bc2)
    bc1_start, bc1_end = bc2_end, bc2_end + len(bc1)
    umi_start, umi_end = bc1_end, bc1_end + len(umi)
    polyg_end = umi_end + polyg_run_length

    seq = bc3 + bc2 + bc1 + umi + (POLYG_BASE * polyg_run_length) + after_polyg

    ann = ReadAnnotation(read_id=read_id)
    ann.set("BC3", bc3)
    ann.set("BC3_POS", format_span(bc3_start, bc3_end))
    ann.set("BC2", bc2)
    ann.set("BC2_POS", format_span(bc2_start, bc2_end))
    ann.set("BC1", bc1)
    ann.set("BC1_POS", format_span(bc1_start, bc1_end))
    ann.set("UMI", umi)
    ann.set("UMI_POS", format_span(umi_start, umi_end))
    if tgidx is not None:
        ann.set("TGIDX", tgidx)
        ann.set("TGIDX_POS", format_span(polyg_end, polyg_end + TGIDX_LENGTH))

    return ann.render(), seq, make_qual(len(seq))


def build_r2_read(read_id: str) -> tuple[str, str, str]:
    """Build a synthetic R2 mate for `read_id`, carrying content distinguishable per read.

    Args:
        read_id: The read id shared with its R1 mate.

    Returns:
        The rendered header, the sequence, and a matching quality string.
    """
    seq = f"R2SEQ{read_id}"
    return f"{read_id} 2:N:0:1", seq, make_qual(len(seq))


def make_prepare_records(
    ids_and_targets: list[tuple[str, str | None]], anchor_length: int
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Build paired R1/R2 records for a whole `prepare_reads()` run.

    Args:
        ids_and_targets: `(read_id, tgidx)` pairs, in the order both streams
            should carry them. `tgidx=None` builds an unmatched read carrying
            no TGIDX tag at all; any other value builds a matched read whose
            TGIDX span is immediately followed by an `anchor_length` filler
            and then the shared insert marker.
        anchor_length: Length of the chemistry's TGIDX right anchor, sizing
            the filler a matched read plants after its TGIDX span so the
            insert begins exactly where `insert_start` will look for it.

    Returns:
        The R1 records and their paired R2 records, in the given order.
    """
    r1_records = [
        build_full_r1_read(
            read_id,
            tgidx=tgidx,
            polyg_run_length=POLYG_MIN_RUN,
            after_polyg=(
                INSERT_SEQ
                if tgidx is None
                else ("A" * TGIDX_LENGTH) + ("N" * anchor_length) + INSERT_SEQ
            ),
        )
        for read_id, tgidx in ids_and_targets
    ]
    r2_records = [build_r2_read(read_id) for read_id, _ in ids_and_targets]
    return r1_records, r2_records


def write_fastq(path: Path, records: list[tuple[str, str, str]]) -> None:
    """Write `(header, seq, qual)` records to a gzipped FASTQ file.

    Args:
        path: Destination path for the gzipped FASTQ.
        records: The records to write, in the order they should appear.
    """
    with gzip.open(path, "wt") as handle:
        for header, seq, qual in records:
            handle.write(f"@{header}\n{seq}\n+\n{qual}\n")


def read_fastq(path: Path) -> list[tuple[str, str, str, str]]:
    """Read a gzipped FASTQ file back into `(header, seq, plus, qual)` records.

    Args:
        path: Path of the gzipped FASTQ to read.

    Returns:
        One four-line tuple per read, in file order.
    """
    with gzip.open(path, "rt") as handle:
        lines = handle.read().splitlines()
    return [tuple(lines[i : i + 4]) for i in range(0, len(lines), 4)]


def extract_read_id(header: str) -> str:
    """Return the read id from an output FASTQ header, matched or unmatched alike.

    A matched header is `read_id|CB=...|UR=...`; an unmatched one is the bare
    read id with nothing appended.

    Args:
        header: The output header line, without its leading `@`.

    Returns:
        The read id portion of the header.
    """
    return header.split("|", 1)[0]


def none_r1_path(directory: Path, prefix: str) -> Path:
    """Return the path of the unmatched arm's trimmed R1 file for `prefix`."""
    return directory / f"{prefix}.none.r1.fastq.gz"


def none_r2_path(directory: Path, prefix: str) -> Path:
    """Return the path of the unmatched arm's passthrough R2 file for `prefix`."""
    return directory / f"{prefix}.none.r2.fastq.gz"


def none_barcodes_path(directory: Path, prefix: str) -> Path:
    """Return the path of the unmatched arm's synthesized barcodes file for `prefix`."""
    return directory / f"{prefix}.none.barcodes.fastq.gz"


def target_r1_path(directory: Path, prefix: str, tgidx: str) -> Path:
    """Return the path of one matched target bucket's R1 file for `prefix`."""
    return directory / f"{prefix}.{tgidx}.r1.fastq.gz"


def target_r2_path(directory: Path, prefix: str, tgidx: str) -> Path:
    """Return the path of one matched target bucket's R2 file for `prefix`."""
    return directory / f"{prefix}.{tgidx}.r2.fastq.gz"


def prepare_stats_path(directory: Path, prefix: str) -> Path:
    """Return the path of the stats report `prepare_reads` writes for `prefix`."""
    return directory / f"{prefix}.prepare_stats.txt"


@pytest.fixture
def build_full_run_preparer(tmp_path: Path) -> Callable[..., ReadPreparer]:
    """Return a factory that writes paired R1/R2 FASTQs and builds a `ReadPreparer` over them.

    Args:
        tmp_path: Directory the synthetic FASTQs are written into.

    Returns:
        A callable taking the R1 records, the R2 records, optional filenames,
        an optional chemistry instance the factory should be made to resolve,
        and any further constructor arguments, returning the preparer built
        over the written files.
    """

    def build(
        r1_records: list[tuple[str, str, str]],
        r2_records: list[tuple[str, str, str]],
        r1_name: str = "input.r1.fastq.gz",
        r2_name: str = "input.r2.fastq.gz",
        chemistry: ChemistryBase | None = None,
        **preparer_kwargs: int | None,
    ) -> ReadPreparer:
        r1_path = tmp_path / r1_name
        r2_path = tmp_path / r2_name
        write_fastq(r1_path, r1_records)
        write_fastq(r2_path, r2_records)
        if chemistry is None:
            return ReadPreparer(str(r1_path), str(r2_path), CHEMISTRY, **preparer_kwargs)
        with patch_chemistry(chemistry):
            return ReadPreparer(str(r1_path), str(r2_path), chemistry.name, **preparer_kwargs)

    return build


# ---------------------------------------------------------------------------
# Recording stand-ins for prepare_reads()'s pool, writers and progress bar.
#
# Adapted from test_target_assigner.py's RecordingWriteStream/RecordingGzipFile/
# RecordingAssignFuture/RecordingAssignExecutor/RecordingProgressBar/PoolHarness,
# generalised to the variable, whitelist-sized number of writers
# `prepare_reads()` opens through a `contextlib.ExitStack` rather than a
# fixed `with (...)` tuple: the recording gzip file and write stream record
# the same open/close events regardless of how many times, or through what
# mechanism, they are entered, so no new mechanism is needed here -- only
# more instances of the same one.
# ---------------------------------------------------------------------------

PREPARE_STREAM_OPEN_EVENT = "stream_open"
PREPARE_STREAM_CLOSE_EVENT = "stream_close"
PREPARE_EXECUTOR_ENTER_EVENT = "executor_enter"
PREPARE_EXECUTOR_EXIT_EVENT = "executor_exit"
PREPARE_EXECUTOR_SHUTDOWN_EVENT = "executor_shutdown"
PREPARE_PROGRESS_ENTER_EVENT = "progress_enter"
PREPARE_PROGRESS_EXIT_EVENT = "progress_exit"


class RecordingPrepareWriteStream:
    """A gzip write stream that reports its open and close to a shared event log.

    Wraps the real stream rather than standing in for it, so `prepare_reads()`
    still writes genuine gzipped FASTQs while the moment each stream opens
    and closes becomes observable.
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
        self.events.append(PREPARE_STREAM_OPEN_EVENT)
        return self.stream.__enter__()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the underlying stream, noting the close first.

        Args:
            exc_type: Type of any exception leaving the block.
            exc: The exception leaving the block, if any.
            tb: Traceback of that exception, if any.
        """
        self.events.append(PREPARE_STREAM_CLOSE_EVENT)
        self.stream.__exit__(exc_type, exc, tb)


class RecordingPrepareGzipFile:
    """A `GzipFile` stand-in handing out streams that report their lifecycle."""

    def __init__(self, filename: str, events: list[str]) -> None:
        """Note the file to compress into.

        Args:
            filename: Path the real stream compresses into.
            events: Shared log the stream appends its lifecycle events to.
        """
        self.filename = filename
        self.events = events

    def open_write_stream(self) -> RecordingPrepareWriteStream:
        """Open the real compressor, wrapped so its lifecycle is recorded.

        Returns:
            The recording wrapper around a real write stream.
        """
        return RecordingPrepareWriteStream(
            GzipFile(self.filename).open_write_stream(), self.events
        )


class RecordingPrepareFuture:
    """A future stand-in that runs its batch on submission and holds the outcome.

    The work runs eagerly and its outcome is held until the result is asked
    for, which is how a real future looks from the driver's side: a worker's
    exception surfaces where the result is retrieved, never where the batch
    was submitted.
    """

    def __init__(self, worker: Callable[..., object], batch: list[tuple[str, str, str]]) -> None:
        """Run the worker over the batch and hold what it did.

        Args:
            worker: Single-argument callable applied to the batch.
            batch: Batch of R1 reads handed to the worker.
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


class RecordingPrepareExecutor:
    """An executor stand-in recording how the stage created, used and closed it.

    Submissions run eagerly in this process, so nothing read back here
    depends on timing. It accepts and keeps the pool's keyword arguments,
    because what the stage passes for `initializer` and `initargs` is the
    contract that gives each worker its preparer, and it runs that
    initializer once, exactly as a real pool runs it once per worker process
    before handing it any batch.
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

    def __enter__(self) -> "RecordingPrepareExecutor":
        """Note that the pool has been entered.

        Returns:
            This executor, as a real pool hands back itself.
        """
        self.events.append(PREPARE_EXECUTOR_ENTER_EVENT)
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
        self.events.append(PREPARE_EXECUTOR_EXIT_EVENT)
        self.shutdown()
        return False

    def submit(
        self, worker: Callable[..., object], batch: list[tuple[str, str, str]]
    ) -> RecordingPrepareFuture:
        """Accept a batch and run its worker straight away.

        Args:
            worker: Callable to apply to the batch.
            batch: Batch to hand to the worker.

        Returns:
            A future stand-in holding the outcome of the call.
        """
        self.submitted_batches.append(batch)
        return RecordingPrepareFuture(worker, batch)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Note that the pool has been shut down.

        Args:
            wait: Accepted for signature compatibility and ignored.
            cancel_futures: Accepted for signature compatibility and ignored.
        """
        self.events.append(PREPARE_EXECUTOR_SHUTDOWN_EVENT)


class RecordingPrepareProgressBar:
    """A progress bar stand-in recording its task, its advances and its entry.

    Records how many batches had already been submitted at the moment it was
    entered, which is the property the stage's layout turns on: the driver
    submits its opening window when it is called, and that first submit is
    when a process pool forks its workers, so the driver call has to come
    before this bar starts.
    """

    def __init__(self, harness: "PreparePoolHarness") -> None:
        """Attach to the harness whose executor the entry is measured against.

        Args:
            harness: Harness holding the shared event log and the executor.
        """
        self.harness = harness
        self.tasks: list[tuple[str, int | None]] = []
        self.updates: list[tuple[str, int]] = []
        self.batches_submitted_on_entry: int | None = None

    def __enter__(self) -> "RecordingPrepareProgressBar":
        """Note the entry and how much work had already been submitted by then.

        Returns:
            This progress bar, as a real one hands back itself.
        """
        self.batches_submitted_on_entry = len(self.harness.submitted_batches)
        self.harness.events.append(PREPARE_PROGRESS_ENTER_EVENT)
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
        self.harness.events.append(PREPARE_PROGRESS_EXIT_EVENT)
        return False

    def add_task(self, description: str, total: int | None) -> str:
        """Record a task, keeping the total exactly as it was given.

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


class PrepareDriverCall(NamedTuple):
    """One call the stage made to the in-order driver.

    Attributes:
        executor: Executor the batches were submitted to.
        worker: Callable each batch was to be handed to.
        n_workers: Window argument, which sizes the driver's in-flight window.
    """

    executor: object
    worker: Callable[..., object]
    n_workers: int


class PreparePoolHarness:
    """Stands in for the executor, the driver, the progress bar and the writers.

    What `prepare_reads()`'s `with`/`ExitStack` block has to get right is the
    order between the writers, the pool and the progress bar: every writer is
    opened before the pool, the pool is torn down before any writer closes,
    and the pool is started before the progress bar begins refreshing.
    Recording all of them into one shared event log is what makes that order
    assertable rather than assumed, at whatever number of writers a given
    chemistry's whitelist happens to open.
    """

    def __init__(self) -> None:
        """Start with nothing created and nothing recorded."""
        self.events: list[str] = []
        self.executors: list[RecordingPrepareExecutor] = []
        self.progress_bars: list[RecordingPrepareProgressBar] = []
        self.driver_calls: list[PrepareDriverCall] = []

    def executor_factory(
        self,
        max_workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> RecordingPrepareExecutor:
        """Create a recording executor, keeping it for the tests to read.

        Args:
            max_workers: Pool width the stage asked for.
            initializer: Callable the stage wants run once per worker process.
            initargs: Arguments that initializer is to be called with.

        Returns:
            The recording executor the stage will use.
        """
        executor = RecordingPrepareExecutor(
            self.events, max_workers, initializer=initializer, initargs=initargs
        )
        self.executors.append(executor)
        return executor

    def gzip_file_factory(self, filename: str) -> RecordingPrepareGzipFile:
        """Create a recording gzip file over a real compressor.

        Args:
            filename: Path the stage means to compress into.

        Returns:
            The recording gzip file the stage will open a stream on.
        """
        return RecordingPrepareGzipFile(filename, self.events)

    def progress_bar_factory(self, unit: str) -> RecordingPrepareProgressBar:
        """Create a recording progress bar, keeping it for the tests to read.

        Args:
            unit: Unit the real bar would label its counts with, ignored here.

        Returns:
            The recording progress bar the stage will use.
        """
        progress = RecordingPrepareProgressBar(self)
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

        Args:
            executor: Executor the driver submits to.
            worker: Callable applied to each batch.
            batches: Batches to map over.
            n_workers: Window argument sizing the in-flight window.

        Returns:
            The real driver's iterator over the batch results.
        """
        self.driver_calls.append(PrepareDriverCall(executor, worker, n_workers))
        return map_batches_in_order(executor, worker, batches, n_workers)

    @property
    def submitted_batches(self) -> list[list[tuple[str, str, str]]]:
        """Return the batches submitted so far, over every executor created."""
        return [batch for executor in self.executors for batch in executor.submitted_batches]


@pytest.fixture
def prepare_pool_harness(monkeypatch: pytest.MonkeyPatch) -> PreparePoolHarness:
    """Install the recording pool, driver, progress bar and gzip file into `read_preparer`.

    The worker preparer is reset through `monkeypatch` before anything runs,
    so the module global the pool initializer writes is restored when the
    test ends and no test leaks its preparer into another.

    Args:
        monkeypatch: Patcher installing the stand-ins and undoing them after.

    Returns:
        The harness every stand-in records into.
    """
    harness = PreparePoolHarness()
    monkeypatch.setattr(read_preparer, "WORKER_PREPARER", None)
    monkeypatch.setattr(read_preparer, "ProcessPoolExecutor", harness.executor_factory)
    monkeypatch.setattr(read_preparer, "GzipFile", harness.gzip_file_factory)
    monkeypatch.setattr(read_preparer, "progress_bar", harness.progress_bar_factory)
    monkeypatch.setattr(read_preparer, "map_batches_in_order", harness.driver)
    return harness


# Behavioural whole-stage tests run over the recording pool rather than a real
# one, mirroring test_target_assigner.py's `substitute_pool`: the harness
# substitutes only the fork, wrapping and calling through to the real driver,
# running the real per-batch worker in this process and opening real
# compressors, so batching, dispatch, ordering and gzip output are all still
# exercised.
substitute_prepare_pool = pytest.mark.usefixtures("prepare_pool_harness")


@substitute_prepare_pool
class TestReadPreparerOutputs:
    """Whole `prepare_reads()` runs over real, small synthetic FASTQ fixtures."""

    def test_every_read_is_written_exactly_once_across_output_files(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """Every input read id appears in exactly one output R1 file, no more, no less."""
        chemistry = ChemistryTwoTargets()
        target_a, target_b = TWO_TARGET_WHITELIST
        anchor_length = chemistry.tgidx_right_anchor().length
        ids_and_targets = [
            ("u1", None),
            ("u2", None),
            ("ma1", target_a),
            ("ma2", target_a),
            ("mb1", target_b),
        ]
        r1_records, r2_records = make_prepare_records(ids_and_targets, anchor_length)
        preparer = build_full_run_preparer(r1_records, r2_records, chemistry=chemistry)

        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        written_ids = [
            extract_read_id(record[0][1:])
            for record in read_fastq(none_r1_path(tmp_path, OUT_PREFIX))
        ]
        for target in TWO_TARGET_WHITELIST:
            written_ids += [
                extract_read_id(record[0][1:])
                for record in read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target))
            ]

        assert_that(sorted(written_ids)).is_equal_to(
            sorted(read_id for read_id, _ in ids_and_targets)
        )

    def test_unmatched_arm_files_match_a_direct_dispatch_and_writer_call(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """The unmatched arm's three files hold exactly what ScrnaWriter would write directly.

        `prepare_read` and `ScrnaWriter.write_read` are each independently
        tested elsewhere, so calling them directly here is an oracle for what
        the whole-run wiring under test should have produced, rather than a
        hand-computed trim point that could drift out of step with either.
        """
        chemistry = ChemistryTwoTargets()
        read_id = "unmatched1"
        name, seq, qual = build_full_r1_read(
            read_id, tgidx=None, polyg_run_length=POLYG_MIN_RUN, after_polyg=INSERT_SEQ
        )
        r2_name, r2_seq, r2_qual = build_r2_read(read_id)
        preparer = build_full_run_preparer(
            [(name, seq, qual)], [(r2_name, r2_seq, r2_qual)], chemistry=chemistry
        )

        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        outcome = preparer.prepare_read(name, seq, qual, PrepareCounts())
        assert_that(outcome).is_instance_of(UnmatchedOutcome)

        expected_r1, expected_r2, expected_barcodes = io.BytesIO(), io.BytesIO(), io.BytesIO()
        ScrnaWriter(preparer.chemistry).write_read(
            outcome.ann,
            outcome.r1_seq,
            outcome.r1_qual,
            r2_name,
            r2_seq,
            r2_qual,
            expected_r1,
            expected_r2,
            expected_barcodes,
        )

        assert_that(read_fastq(none_r1_path(tmp_path, OUT_PREFIX))).is_equal_to(
            [tuple(expected_r1.getvalue().decode("UTF-8").splitlines())]
        )
        assert_that(read_fastq(none_r2_path(tmp_path, OUT_PREFIX))).is_equal_to(
            [tuple(expected_r2.getvalue().decode("UTF-8").splitlines())]
        )
        assert_that(read_fastq(none_barcodes_path(tmp_path, OUT_PREFIX))).is_equal_to(
            [tuple(expected_barcodes.getvalue().decode("UTF-8").splitlines())]
        )

    def test_matched_arm_files_match_a_direct_dispatch_and_writer_call_across_two_buckets(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """Each matched bucket's two files hold exactly what write_sctip_read would write directly."""
        chemistry = ChemistryTwoTargets()
        target_a, target_b = TWO_TARGET_WHITELIST
        anchor_length = chemistry.tgidx_right_anchor().length
        ids_and_targets = [("ma", target_a), ("mb", target_b)]
        r1_records, r2_records = make_prepare_records(ids_and_targets, anchor_length)
        preparer = build_full_run_preparer(r1_records, r2_records, chemistry=chemistry)

        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        for (name, seq, qual), (_, r2_seq, r2_qual) in zip(r1_records, r2_records):
            outcome = preparer.prepare_read(name, seq, qual, PrepareCounts())
            assert_that(outcome).is_instance_of(MatchedOutcome)

            writers = {outcome.tgidx: SctipBucketWriters(r1=io.BytesIO(), r2=io.BytesIO())}
            write_sctip_read(
                writers,
                outcome.tgidx,
                outcome.header,
                outcome.r1_seq,
                outcome.r1_qual,
                r2_seq,
                r2_qual,
            )
            expected_r1 = tuple(writers[outcome.tgidx].r1.getvalue().decode("UTF-8").splitlines())
            expected_r2 = tuple(writers[outcome.tgidx].r2.getvalue().decode("UTF-8").splitlines())

            assert_that(read_fastq(target_r1_path(tmp_path, OUT_PREFIX, outcome.tgidx))).contains(
                expected_r1
            )
            assert_that(read_fastq(target_r2_path(tmp_path, OUT_PREFIX, outcome.tgidx))).contains(
                expected_r2
            )

    def test_target_bucket_with_no_matched_reads_still_gets_empty_output_files(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """A whitelisted bucket that receives no read is still opened, not created lazily."""
        chemistry = ChemistryTwoTargets()
        target_a, target_b = TWO_TARGET_WHITELIST
        anchor_length = chemistry.tgidx_right_anchor().length
        r1_records, r2_records = make_prepare_records([("ma1", target_a)], anchor_length)
        preparer = build_full_run_preparer(r1_records, r2_records, chemistry=chemistry)

        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(target_r1_path(tmp_path, OUT_PREFIX, target_b).exists()).is_true()
        assert_that(target_r2_path(tmp_path, OUT_PREFIX, target_b).exists()).is_true()
        assert_that(read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target_b))).is_empty()
        assert_that(read_fastq(target_r2_path(tmp_path, OUT_PREFIX, target_b))).is_empty()

    def test_chemistry_without_target_assignment_writes_only_the_scrna_files(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """A chemistry with no target index writes no per-target files at all.

        Every read here carries no TGIDX tag whatsoever, the shape real
        extract-umis output takes for such a chemistry -- distinct from a
        `TGIDX=NONE`-tagged read, which an assign-targets-shaped input would
        carry instead.
        """
        chemistry = ChemistryNoTargetIndex()
        ids = ["a", "b", "c"]
        r1_records = [
            build_full_r1_read(read_id, tgidx=None, polyg_run_length=0, after_polyg=INSERT_SEQ)
            for read_id in ids
        ]
        r2_records = [build_r2_read(read_id) for read_id in ids]
        preparer = build_full_run_preparer(r1_records, r2_records, chemistry=chemistry)

        stats = preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.unmatched_written).is_equal_to(len(ids))
        assert_that(stats.target_written).is_empty()

        assert_that(none_r1_path(tmp_path, OUT_PREFIX).exists()).is_true()
        assert_that(none_r2_path(tmp_path, OUT_PREFIX).exists()).is_true()
        assert_that(none_barcodes_path(tmp_path, OUT_PREFIX).exists()).is_true()
        assert_that(prepare_stats_path(tmp_path, OUT_PREFIX).exists()).is_true()

        input_names = {"input.r1.fastq.gz", "input.r2.fastq.gz"}
        produced_files = sorted(
            path.name for path in tmp_path.iterdir() if path.name not in input_names
        )
        assert_that(produced_files).is_equal_to(
            sorted(
                [
                    f"{OUT_PREFIX}.none.r1.fastq.gz",
                    f"{OUT_PREFIX}.none.r2.fastq.gz",
                    f"{OUT_PREFIX}.none.barcodes.fastq.gz",
                    f"{OUT_PREFIX}.prepare_stats.txt",
                ]
            )
        )

        written_ids = [
            extract_read_id(record[0][1:])
            for record in read_fastq(none_r1_path(tmp_path, OUT_PREFIX))
        ]
        assert_that(sorted(written_ids)).is_equal_to(sorted(ids))

    def test_prepare_stats_reconciles_and_matches_the_written_report(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        """The returned stats reconcile, and the stats file on disk matches them exactly."""
        chemistry = ChemistryTwoTargets()
        target_a, target_b = TWO_TARGET_WHITELIST
        anchor_length = chemistry.tgidx_right_anchor().length
        ids_and_targets = [
            ("u1", None),
            ("u2", None),
            ("ma1", target_a),
            ("ma2", target_a),
            ("mb1", target_b),
        ]
        r1_records, r2_records = make_prepare_records(ids_and_targets, anchor_length)
        preparer = build_full_run_preparer(r1_records, r2_records, chemistry=chemistry)

        stats = preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(len(ids_and_targets))
        assert_that(stats.unmatched_written + stats.matched_written).is_equal_to(stats.total_reads)
        assert_that(stats.unmatched_written).is_equal_to(2)
        assert_that(stats.target_written).is_equal_to({target_a: 2, target_b: 1})

        report_on_disk = strip_report_run_details(
            prepare_stats_path(tmp_path, OUT_PREFIX).read_text()
        )
        assert_that(report_on_disk).is_equal_to(strip_report_run_details(stats.get_report()))

    def test_prefix_defaults_to_the_r1_input_filename(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        chemistry = ChemistryNoTargetIndex()
        read_id = "a"
        name, seq, qual = build_full_r1_read(
            read_id, tgidx=None, polyg_run_length=0, after_polyg=INSERT_SEQ
        )
        r2_name, r2_seq, r2_qual = build_r2_read(read_id)
        r1_filename = "SAMPLE123.r1.fastq.gz"
        preparer = build_full_run_preparer(
            [(name, seq, qual)],
            [(r2_name, r2_seq, r2_qual)],
            r1_name=r1_filename,
            chemistry=chemistry,
        )

        preparer.prepare_reads(output_dir=str(tmp_path))

        expected_prefix = get_prefix(r1_filename)
        assert_that(none_r1_path(tmp_path, expected_prefix).exists()).is_true()
        assert_that(none_r2_path(tmp_path, expected_prefix).exists()).is_true()
        assert_that(none_barcodes_path(tmp_path, expected_prefix).exists()).is_true()
        assert_that(prepare_stats_path(tmp_path, expected_prefix).exists()).is_true()

    def test_empty_input_writes_every_output_file_empty_and_a_zero_count_report(
        self, build_full_run_preparer: Callable[..., ReadPreparer], tmp_path: Path
    ) -> None:
        chemistry = ChemistryTwoTargets()
        preparer = build_full_run_preparer([], [], chemistry=chemistry)

        stats = preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(0)
        assert_that(stats.unmatched_written).is_equal_to(0)
        assert_that(stats.target_written).is_empty()

        assert_that(read_fastq(none_r1_path(tmp_path, OUT_PREFIX))).is_empty()
        assert_that(read_fastq(none_r2_path(tmp_path, OUT_PREFIX))).is_empty()
        assert_that(read_fastq(none_barcodes_path(tmp_path, OUT_PREFIX))).is_empty()
        for target in TWO_TARGET_WHITELIST:
            assert_that(read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target))).is_empty()
            assert_that(read_fastq(target_r2_path(tmp_path, OUT_PREFIX, target))).is_empty()

        assert_that(prepare_stats_path(tmp_path, OUT_PREFIX).read_text()).contains(
            "Total reads: 0"
        )


PREPARE_POOL_WORKERS = 3
PREPARE_POOL_BATCH_SIZE = 2


class TestReadPreparerPoolWiring:
    """How `prepare_reads()` creates, orders and feeds the pool it runs its worker on.

    Uses `ChemistryTwoTargets`, so every run in this class opens seven
    writers -- three fixed plus two per target bucket -- more than either
    `assign_targets`'s or barcode extraction's own pool-wiring tests
    exercise, which is the point: it proves the ordering invariant holds at a
    writer count no existing precedent covers.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryTwoTargets:
        """Return the two-target chemistry every test in this class runs over."""
        return ChemistryTwoTargets()

    @pytest.fixture
    def preparer(
        self,
        build_full_run_preparer: Callable[..., ReadPreparer],
        chemistry: ChemistryTwoTargets,
    ) -> ReadPreparer:
        """Return a preparer over five reads, three workers and two per batch."""
        anchor_length = chemistry.tgidx_right_anchor().length
        target_a, target_b = TWO_TARGET_WHITELIST
        ids_and_targets = [
            ("r0", None),
            ("r1", target_a),
            ("r2", None),
            ("r3", target_b),
            ("r4", target_a),
        ]
        r1_records, r2_records = make_prepare_records(ids_and_targets, anchor_length)
        return build_full_run_preparer(
            r1_records,
            r2_records,
            chemistry=chemistry,
            n_workers=PREPARE_POOL_WORKERS,
            batch_size=PREPARE_POOL_BATCH_SIZE,
        )

    def test_all_writer_opens_precede_the_executor_enter_and_all_closes_follow_the_executor_exit_and_shutdown(
        self, preparer: ReadPreparer, prepare_pool_harness: PreparePoolHarness, tmp_path: Path
    ) -> None:
        """The pool is gone before any compressor is asked to finish, at every one of the seven writers.

        Generalises `test_target_assigner.py`'s
        `test_executor_is_torn_down_before_the_output_stream_is_closed` from
        a fixed pair of streams to a whitelist-sized set of them, since
        `contextlib.ExitStack` only closes cleanly if every one of them --
        not merely the first or the last -- opens before the pool and closes
        after it.
        """
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        events = prepare_pool_harness.events
        open_indices = [i for i, event in enumerate(events) if event == PREPARE_STREAM_OPEN_EVENT]
        close_indices = [
            i for i, event in enumerate(events) if event == PREPARE_STREAM_CLOSE_EVENT
        ]
        enter_indices = [
            i for i, event in enumerate(events) if event == PREPARE_EXECUTOR_ENTER_EVENT
        ]
        exit_indices = [
            i for i, event in enumerate(events) if event == PREPARE_EXECUTOR_EXIT_EVENT
        ]
        shutdown_indices = [
            i for i, event in enumerate(events) if event == PREPARE_EXECUTOR_SHUTDOWN_EVENT
        ]

        assert_that(open_indices).is_length(7)
        assert_that(close_indices).is_length(7)
        assert_that(enter_indices).is_length(1)
        assert_that(exit_indices).is_length(1)
        assert_that(shutdown_indices).is_length(1)
        assert_that(max(open_indices)).is_less_than(min(enter_indices))
        assert_that(max(exit_indices)).is_less_than(min(close_indices))
        assert_that(max(shutdown_indices)).is_less_than(min(close_indices))

    def test_executor_is_constructed_with_the_worker_initializer_and_the_parent_preparer(
        self, preparer: ReadPreparer, prepare_pool_harness: PreparePoolHarness, tmp_path: Path
    ) -> None:
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(prepare_pool_harness.executors).is_length(1)
        executor = prepare_pool_harness.executors[0]
        assert_that(executor.max_workers).is_equal_to(PREPARE_POOL_WORKERS)
        assert_that(executor.initializer).is_same_as(read_preparer.init_prepare_worker)
        assert_that(executor.initargs).is_length(1)
        assert_that(executor.initargs[0]).is_same_as(preparer)

    def test_driver_is_called_with_the_batch_worker_and_the_stages_worker_count(
        self, preparer: ReadPreparer, prepare_pool_harness: PreparePoolHarness, tmp_path: Path
    ) -> None:
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(prepare_pool_harness.driver_calls).is_length(1)
        call = prepare_pool_harness.driver_calls[0]
        assert_that(call.executor).is_same_as(prepare_pool_harness.executors[0])
        assert_that(call.worker).is_same_as(read_preparer.prepare_read_batch)
        assert_that(call.n_workers).is_equal_to(PREPARE_POOL_WORKERS)

    def test_first_batch_is_submitted_before_the_progress_bar_is_entered(
        self, preparer: ReadPreparer, prepare_pool_harness: PreparePoolHarness, tmp_path: Path
    ) -> None:
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(prepare_pool_harness.progress_bars).is_length(1)
        submitted_on_entry = prepare_pool_harness.progress_bars[0].batches_submitted_on_entry
        assert_that(submitted_on_entry).is_not_none()
        assert_that(submitted_on_entry).is_greater_than_or_equal_to(1)

    def test_batch_results_are_folded_inside_the_block_that_keeps_the_pool_alive(
        self, preparer: ReadPreparer, prepare_pool_harness: PreparePoolHarness, tmp_path: Path
    ) -> None:
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        events = prepare_pool_harness.events
        assert_that(events).contains(PREPARE_PROGRESS_ENTER_EVENT, PREPARE_PROGRESS_EXIT_EVENT)
        assert_that(events.index(PREPARE_EXECUTOR_ENTER_EVENT)).is_less_than(
            events.index(PREPARE_PROGRESS_ENTER_EVENT)
        )
        assert_that(events.index(PREPARE_PROGRESS_EXIT_EVENT)).is_less_than(
            events.index(PREPARE_EXECUTOR_EXIT_EVENT)
        )

    def test_first_pair_mismatch_raises_before_any_executor_or_writer_is_created(
        self,
        build_full_run_preparer: Callable[..., ReadPreparer],
        chemistry: ChemistryTwoTargets,
        prepare_pool_harness: PreparePoolHarness,
        tmp_path: Path,
    ) -> None:
        """A mismatch on the very first pair leaves no executor, no event and no file behind."""
        name, seq, qual = build_full_r1_read(
            "a", tgidx=None, polyg_run_length=POLYG_MIN_RUN, after_polyg=INSERT_SEQ
        )
        r2_name, r2_seq, r2_qual = build_r2_read("mismatched")
        preparer = build_full_run_preparer(
            [(name, seq, qual)], [(r2_name, r2_seq, r2_qual)], chemistry=chemistry
        )

        with pytest.raises(ValueError):
            preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(prepare_pool_harness.executors).is_empty()
        assert_that(prepare_pool_harness.events).is_empty()
        assert_that(none_r1_path(tmp_path, OUT_PREFIX).exists()).is_false()
        assert_that(none_r2_path(tmp_path, OUT_PREFIX).exists()).is_false()
        assert_that(none_barcodes_path(tmp_path, OUT_PREFIX).exists()).is_false()
        assert_that(prepare_stats_path(tmp_path, OUT_PREFIX).exists()).is_false()
        for target in TWO_TARGET_WHITELIST:
            assert_that(target_r1_path(tmp_path, OUT_PREFIX, target).exists()).is_false()
            assert_that(target_r2_path(tmp_path, OUT_PREFIX, target).exists()).is_false()

    def test_mismatch_after_several_matches_still_tears_down_the_pool_and_writers_cleanly(
        self,
        build_full_run_preparer: Callable[..., ReadPreparer],
        chemistry: ChemistryTwoTargets,
        prepare_pool_harness: PreparePoolHarness,
        tmp_path: Path,
    ) -> None:
        """A mismatch reached partway through the stream still tears everything down cleanly.

        The exception happens inside the `with`/`ExitStack` block, so this
        pins that the cleanup-on-exception path does not itself deadlock --
        exactly the class of bug this design has to rule out.
        """
        anchor_length = chemistry.tgidx_right_anchor().length
        ok_r1, ok_r2 = make_prepare_records(
            [("r0", None), ("r1", TWO_TARGET_WHITELIST[0])], anchor_length
        )
        bad_name, bad_seq, bad_qual = build_full_r1_read(
            "r2", tgidx=None, polyg_run_length=POLYG_MIN_RUN, after_polyg=INSERT_SEQ
        )
        bad_r2_name, bad_r2_seq, bad_r2_qual = build_r2_read("mismatched")
        r1_records = ok_r1 + [(bad_name, bad_seq, bad_qual)]
        r2_records = ok_r2 + [(bad_r2_name, bad_r2_seq, bad_r2_qual)]
        preparer = build_full_run_preparer(
            r1_records, r2_records, chemistry=chemistry, n_workers=1, batch_size=2
        )

        with pytest.raises(ValueError):
            preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        events = prepare_pool_harness.events
        assert_that(events).contains(PREPARE_EXECUTOR_EXIT_EVENT, PREPARE_EXECUTOR_SHUTDOWN_EVENT)
        assert_that(events.count(PREPARE_STREAM_CLOSE_EVENT)).is_equal_to(7)


class TestReadPreparerModuleWorkerPlumbing:
    """The once-per-process handover of the preparer, and the batch worker it backs."""

    def test_worker_preparer_starts_out_unset(self) -> None:
        """No preparer is installed until a worker process is initialised.

        Every test that installs one restores it via `monkeypatch`, so a
        failure here is either a module declaring the global already filled
        in or another test leaking its own preparer into this one.
        """
        assert_that(read_preparer.WORKER_PREPARER).is_none()

    def test_init_prepare_worker_installs_the_preparer_it_is_handed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker's preparer is the very object the initializer was given."""
        preparer = build_preparer()
        monkeypatch.setattr(read_preparer, "WORKER_PREPARER", None)

        read_preparer.init_prepare_worker(preparer)

        assert_that(read_preparer.WORKER_PREPARER).is_same_as(preparer)

    def test_prepare_read_batch_takes_the_batch_alone(self) -> None:
        """One parameter, so the driver can submit the worker as it stands."""
        assert_that(list(signature(read_preparer.prepare_read_batch).parameters)).is_length(1)

    def test_prepare_read_batch_reads_the_preparer_from_the_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The answer follows whichever preparer the process has installed.

        The same read is matched under a chemistry with a real target index
        and unmatched under one with none at all, so the outcome type
        changing with nothing but the module global is what pins where the
        worker reads its preparer from.
        """
        matched_preparer = build_preparer()
        anchor = matched_preparer.tgidx_right_anchor
        name, seq, qual = build_matched_read(
            "swap", TGIDX_VALUE, 30, "N" * anchor.length + INSERT_SEQ
        )

        monkeypatch.setattr(read_preparer, "WORKER_PREPARER", matched_preparer)
        matched_outcomes, _ = read_preparer.prepare_read_batch([(name, seq, qual)])

        unmatched_preparer = build_preparer(chemistry=ChemistryNoTargetIndex())
        monkeypatch.setattr(read_preparer, "WORKER_PREPARER", unmatched_preparer)
        unmatched_outcomes, _ = read_preparer.prepare_read_batch([(name, seq, qual)])

        assert_that(matched_outcomes[0]).is_instance_of(MatchedOutcome)
        assert_that(unmatched_outcomes[0]).is_instance_of(UnmatchedOutcome)

    def test_prepare_read_batch_matches_direct_per_read_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker computes exactly what calling `prepare_read` per read would.

        Asserted against the per-read body rather than against copied
        literals, so this pins the worker as the same computation as the
        serial pass it was lifted from, not merely a plausible-looking one.
        """
        preparer = build_preparer()
        anchor = preparer.tgidx_right_anchor
        reads = [
            build_read("u1", {}, "A" * 20),
            build_matched_read("m1", TGIDX_VALUE, 30, "N" * anchor.length + INSERT_SEQ),
            build_read("u2", {"TGIDX": NO_TARGET}, "A" * 20),
        ]
        monkeypatch.setattr(read_preparer, "WORKER_PREPARER", preparer)

        expected_counts = PrepareCounts()
        expected_outcomes = [
            preparer.prepare_read(name, seq, qual, expected_counts) for name, seq, qual in reads
        ]

        outcomes, counts = read_preparer.prepare_read_batch(reads)

        assert_that(outcomes).is_equal_to(expected_outcomes)
        assert_that(counts).is_equal_to(expected_counts)

    def test_prepare_read_batch_maps_an_empty_batch_to_no_reads_and_zero_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty batch is answered, not refused: nothing in, nothing out."""
        monkeypatch.setattr(read_preparer, "WORKER_PREPARER", build_preparer())

        outcomes, counts = read_preparer.prepare_read_batch([])

        assert_that(outcomes).is_empty()
        assert_that(counts).is_equal_to(PrepareCounts())


class TestR1BatchesWithR2Sidecar:
    """Direct unit tests of `r1_batches_with_r2_sidecar`'s r1/r2 alignment guarantee.

    No pool is involved anywhere in this class: these tests exercise the
    generator in isolation, since the alignment property the whole
    `prepare_reads()` design depends on -- that popping `r2_sidecar` once per
    drained result reconstructs the correct `r2_batch` for that result -- has
    to hold on its own, before it is ever combined with `map_batches_in_order`.
    """

    def test_sidecar_aligns_with_iter_read_pair_batches_across_full_and_short_batches(
        self,
    ) -> None:
        pairs = make_pairs(BATCH_SIZE * FULL_BATCHES + 1)
        expected = list(iter_read_pair_batches(iter(pairs), BATCH_SIZE))

        r2_sidecar: deque = deque()
        r1_batches = list(
            read_preparer.r1_batches_with_r2_sidecar(iter(pairs), BATCH_SIZE, r2_sidecar)
        )

        assert_that(r1_batches).is_equal_to([r1_batch for r1_batch, _ in expected])
        assert_that(list(r2_sidecar)).is_equal_to([r2_batch for _, r2_batch in expected])

    def test_sidecar_entry_is_appended_before_its_batch_is_yielded(self) -> None:
        """Each pull of the generator grows the sidecar by one before handing back its batch.

        Proven one pull at a time rather than via a final `list(...)`, since
        that is the only way to observe that the append happens strictly
        before the yield -- which is what makes it safe for a caller like
        `map_batches_in_order` to submit a batch and only later pop its
        sidecar entry back off.
        """
        pairs = make_pairs(BATCH_SIZE * FULL_BATCHES + 1)
        r2_sidecar: deque = deque()
        generator = read_preparer.r1_batches_with_r2_sidecar(iter(pairs), BATCH_SIZE, r2_sidecar)

        for expected_length in range(1, FULL_BATCHES + 2):
            next(generator)
            assert_that(len(r2_sidecar)).is_equal_to(expected_length)

    def test_empty_input_yields_nothing_and_appends_nothing(self) -> None:
        r2_sidecar: deque = deque()

        result = list(read_preparer.r1_batches_with_r2_sidecar(iter([]), BATCH_SIZE, r2_sidecar))

        assert_that(result).is_empty()
        assert_that(r2_sidecar).is_empty()


# ---------------------------------------------------------------------------
# A real forked ProcessPoolExecutor regression test.
#
# Every other prepare_reads() test in this file substitutes the executor (see
# PoolHarness above), so none of them exercise the interaction between forked
# pool workers and this stage's gzip writer subprocesses -- the interaction
# whose wrong ordering caused the historical extract_barcodes deadlock
# (fixed by commit a4e56f25 and pinned there by
# TestBarcodeExtractorRealProcessPool) that this stage's ExitStack,
# writers-before-executor design exists to avoid at a larger, dynamic writer
# count. This class runs a real prepare_reads() over a real forked pool
# instead, using the real, ChemistryFactory-registered chemistry named by
# CHEMISTRY. That chemistry's single whitelisted target plus its three fixed
# scRNA files opens five concurrent gzip writers -- more concurrent writers
# than any existing precedent -- with no chemistry mocking anywhere in this
# class, since the fork happens only after ReadPreparer.__init__ has already
# resolved the chemistry, inside the child's own target() function.
# ---------------------------------------------------------------------------

REAL_POOL_READS = 24
REAL_POOL_TIMEOUT_S = 180
REAL_POOL_PREFIX = "real_pool"
REAL_POOL_OUTPUTS = (
    f"{REAL_POOL_PREFIX}.none.r1.fastq.gz",
    f"{REAL_POOL_PREFIX}.none.r2.fastq.gz",
    f"{REAL_POOL_PREFIX}.none.barcodes.fastq.gz",
    f"{REAL_POOL_PREFIX}.{TGIDX_VALUE}.r1.fastq.gz",
    f"{REAL_POOL_PREFIX}.{TGIDX_VALUE}.r2.fastq.gz",
    f"{REAL_POOL_PREFIX}.prepare_stats.txt",
)

# The gzip FASTQ outputs among REAL_POOL_OUTPUTS, excluding the plain-text
# stats report -- the five streamed writers whose pipe lifecycle this class
# exercises.
REAL_POOL_GZIP_OUTPUTS = tuple(name for name in REAL_POOL_OUTPUTS if name.endswith(".fastq.gz"))


def run_prepare_reads_in_process_group(
    r1_path: Path, r2_path: Path, output_dir: Path, prefix: str
) -> None:
    """Run a real prepare_reads() in a forked child that leads its own process group.

    A teardown-order or drain-order regression deadlocks rather than fails, so
    the run is given a deadline. The child leads its own process group so that
    killing it takes the pool workers with it -- that releases the inherited
    pipe write ends, letting the stranded gzip writers see EOF and exit
    instead of lingering for the rest of the session.

    Args:
        r1_path: The UMI- and target-annotated R1 FASTQ to prepare.
        r2_path: The paired R2 FASTQ.
        output_dir: Directory the run writes its output files into.
        prefix: Prefix for the output file names.

    Raises:
        AssertionError: If the child misses its deadline, or exits non-zero.
    """

    def target() -> None:
        os.setsid()
        preparer = ReadPreparer(str(r1_path), str(r2_path), CHEMISTRY, n_workers=2)
        preparer.prepare_reads(str(output_dir), prefix)

    proc = multiprocessing.get_context("fork").Process(target=target)
    proc.start()
    proc.join(REAL_POOL_TIMEOUT_S)

    if proc.is_alive():
        os.killpg(proc.pid, signal.SIGKILL)
        proc.join(REAL_POOL_TIMEOUT_S)
        pytest.fail(
            f"prepare_reads did not finish within {REAL_POOL_TIMEOUT_S}s - the gzip "
            "writers are most likely blocked waiting on EOF for pipes still held open "
            "by pool workers"
        )

    assert_that(proc.exitcode).is_equal_to(0)


class TestReadPreparerRealProcessPool:
    """
    End-to-end prepare_reads() driven by a real forked worker pool.

    Every other prepare_reads() test substitutes the executor, so none of
    them exercise the interaction between forked pool workers and the five
    gzip writer subprocesses this run opens. Workers are forked on first
    submit and inherit the writers' pipe write ends, so tearing the two down
    in the wrong order strands gzip on an EOF that never arrives.
    """

    @pytest.fixture(scope="class")
    def extraction_output(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Run one real-pool prepare_reads() and hand its output directory to the tests."""
        work_dir = tmp_path_factory.mktemp("real_pool")
        anchor_length = ChemistryFactory.get_chemistry(CHEMISTRY).tgidx_right_anchor().length

        half = REAL_POOL_READS // 2
        ids_and_targets = [(f"u{i}", None) for i in range(half)] + [
            (f"m{i}", TGIDX_VALUE) for i in range(REAL_POOL_READS - half)
        ]
        r1_records, r2_records = make_prepare_records(ids_and_targets, anchor_length)

        r1_path = work_dir / "real_pool.r1.fastq.gz"
        r2_path = work_dir / "real_pool.r2.fastq.gz"
        write_fastq(r1_path, r1_records)
        write_fastq(r2_path, r2_records)

        output_dir = work_dir / "out"
        output_dir.mkdir()
        run_prepare_reads_in_process_group(r1_path, r2_path, output_dir, REAL_POOL_PREFIX)

        return output_dir

    def test_prepare_reads_completes_and_writes_every_output_file(
        self, extraction_output: Path
    ) -> None:
        """A real-pool run reaches finalisation, so every output file is written."""
        for name in REAL_POOL_OUTPUTS:
            assert_that((extraction_output / name).exists()).described_as(name).is_true()

    def test_streamed_gzip_outputs_are_complete(self, extraction_output: Path) -> None:
        """The gzip writers terminate cleanly, so their streams decompress whole.

        A writer left stranded on a pipe that never sees EOF is torn down
        with its stream still open, which would either hang this test (were
        the deadlock still present) or leave a truncated gzip member behind
        that fails to decompress or ends mid-record. Decompressing here
        raises on the former; the multiple-of-four line count rules out the
        latter.
        """
        for name in REAL_POOL_GZIP_OUTPUTS:
            with gzip.open(extraction_output / name, "rt") as handle:
                lines = handle.read().splitlines()
            assert_that(len(lines) % 4).described_as(name).is_equal_to(0)

    def test_reads_are_conserved_across_the_real_pool_run(self, extraction_output: Path) -> None:
        """Every read the run saw is written to exactly one arm, and the stats report agrees."""
        none_records = read_fastq(extraction_output / f"{REAL_POOL_PREFIX}.none.r1.fastq.gz")
        matched_records = read_fastq(
            extraction_output / f"{REAL_POOL_PREFIX}.{TGIDX_VALUE}.r1.fastq.gz"
        )

        assert_that(len(none_records) + len(matched_records)).is_equal_to(REAL_POOL_READS)
        assert_that(
            (extraction_output / f"{REAL_POOL_PREFIX}.prepare_stats.txt").read_text()
        ).contains(f"Total reads: {REAL_POOL_READS}")


# ---------------------------------------------------------------------------
# The prepare-reads command line interface's wiring into the carmack command
# line: that "prepare-reads" is registered on the command group, sits in the
# right place in the grouped help listing, and passes its options through to
# ReadPreparer correctly.
# ---------------------------------------------------------------------------

# The name the stage is exposed under and the metavars its two positional
# FASTQ arguments render as. All three are user-facing vocabulary rather than
# internals, so they are pinned literally: renaming any of them is a breaking
# change to anyone with a pipeline script or a documented command line.
PREPARE_READS_COMMAND_NAME = "prepare-reads"
PREPARE_READS_R1_METAVAR = "<r1_annotated_fastq>"
PREPARE_READS_R2_METAVAR = "<r2_fastq>"

# The help group the command belongs to and the two commands it sits between.
# The stage runs after assign-targets and before fastq-filter, and the
# grouped help listing is the only place a user reads that order off, so the
# position is part of the contract and not a cosmetic detail.
PREPARE_READS_USER_COMMAND_GROUP = "Commands for users"
PREPARE_READS_PRECEDING_COMMAND = "assign-targets"
PREPARE_READS_FOLLOWING_COMMAND = "fastq-filter"

# The parameter name the pool width is bound to and the two spellings it is
# offered under -- the same convention assign-targets, extract-barcodes and
# split-bam already use.
PREPARE_READS_CPU_COUNT_OPTION = "cpu_count"
PREPARE_READS_CPU_COUNT_OPTS = ["-n", "--cpu_count"]

# Pool width handed to the command to check the option reaches the
# constructor. It differs from both the resolved default and from one, so a
# wiring that ignored the option and passed either would still fail.
PREPARE_READS_CLI_WORKERS = 3

# The width the command runs at when no -n is given: DEFAULT_MAX_WORKERS here
# is read_preparer's own constant (already imported above), not
# assign-targets's -- the two are semantically distinct per-stage constants
# that happen to share a value today.
PREPARE_READS_DEFAULT_CPU_COUNT = min(DEFAULT_MAX_WORKERS, get_cpu_count())

PREPARE_READS_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def prepare_reads_flatten_help(output: str) -> str:
    """Collapse a rendered help screen into one lowercase line.

    rich_click lays the options out in a bordered table, so a help string
    longer than its column wraps over several rows with a rule character at
    each end. Dropping the rules and collapsing the whitespace puts the
    string back together as the user reads it, which is what an assertion on
    a phrase needs.

    Colour codes are stripped first, and that is not cosmetic. rich_click
    styles the help whenever the terminal accepts colour, which puts escape
    sequences between the words of a wrapped phrase and inside a metavar's
    angle brackets. Collapsing whitespace alone leaves those sequences behind
    as tokens, so an assertion on a phrase passes only on a terminal that
    refused colour -- the tests would pass in CI and fail for anyone running
    them locally.

    Args:
        output: The help screen as the runner captured it.

    Returns:
        The same text as a single lowercase line of space-separated words.
    """
    plain = PREPARE_READS_ANSI_ESCAPE.sub("", output)
    return " ".join(plain.replace("│", " ").split()).lower()


def prepare_reads_user_command_names() -> list[str]:
    """Return the commands listed in the user-facing help group, in help order.

    Returns:
        The command names of the ``Commands for users`` group, in the order
        the grouped help listing renders them.
    """
    groups = click.rich_click.COMMAND_GROUPS["carmack"]
    return next(
        group["commands"] for group in groups if group["name"] == PREPARE_READS_USER_COMMAND_GROUP
    )


def prepare_reads_command_option(command_name: str, option_name: str) -> click.Parameter:
    """Return one declared option of a command on the real carmack command group.

    Args:
        command_name: Name the command is registered under.
        option_name: Parameter name the option binds to.

    Returns:
        The Click parameter, so its names, default and rendering flags can be
        read off the declaration itself rather than inferred from help text.
    """
    command = carmack.__main__.carmack_cli.commands[command_name]
    declared = [param for param in command.params if param.name == option_name]
    assert_that(declared).described_as(f"{command_name} option '{option_name}'").is_length(1)
    return declared[0]


class TestPrepareReadsCli:
    """The prepare-reads command's wiring into the carmack command line."""

    @pytest.mark.parametrize(
        "prefix_args, expected_prefix",
        [
            pytest.param([], None, id="prefix-defaults-to-none"),
            pytest.param(["--prefix", OUT_PREFIX], OUT_PREFIX, id="prefix-passed-through"),
        ],
    )
    def test_command_constructs_the_preparer_and_prepares_reads(
        self, tmp_path: Path, prefix_args: list[str], expected_prefix: str | None
    ) -> None:
        runner = CliRunner()
        with mock.patch("carmack.__main__.ReadPreparer", autospec=True) as mock_preparer:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [
                    PREPARE_READS_COMMAND_NAME,
                    DUMMY_R1_FASTQ,
                    DUMMY_R2_FASTQ,
                    "--chemistry",
                    CHEMISTRY,
                    "--output_dir",
                    str(tmp_path),
                    *prefix_args,
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_preparer.assert_called_once_with(
            DUMMY_R1_FASTQ,
            DUMMY_R2_FASTQ,
            CHEMISTRY,
            n_workers=PREPARE_READS_DEFAULT_CPU_COUNT,
        )
        mock_preparer.return_value.prepare_reads.assert_called_once_with(
            str(tmp_path), expected_prefix
        )

    def test_command_is_listed_in_the_top_level_help(self) -> None:
        runner = CliRunner()

        result = runner.invoke(carmack.__main__.carmack_cli, ["--help"])

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(result.output).contains(PREPARE_READS_COMMAND_NAME)

    def test_command_group_places_it_between_assign_targets_and_fastq_filter(self) -> None:
        commands = prepare_reads_user_command_names()

        assert_that(commands).contains(PREPARE_READS_COMMAND_NAME)
        assert_that(commands.index(PREPARE_READS_COMMAND_NAME)).is_equal_to(
            commands.index(PREPARE_READS_PRECEDING_COMMAND) + 1
        )
        assert_that(commands.index(PREPARE_READS_FOLLOWING_COMMAND)).is_equal_to(
            commands.index(PREPARE_READS_COMMAND_NAME) + 1
        )

    def test_chemistry_option_is_required(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with mock.patch("carmack.__main__.ReadPreparer", autospec=True) as mock_preparer:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [
                    PREPARE_READS_COMMAND_NAME,
                    DUMMY_R1_FASTQ,
                    DUMMY_R2_FASTQ,
                    "--output_dir",
                    str(tmp_path),
                ],
            )

        assert_that(result.exit_code).is_not_equal_to(0)
        assert_that(result.output).contains("--chemistry")
        mock_preparer.assert_not_called()

    def test_command_help_documents_its_arguments_and_chemistry_option(self) -> None:
        runner = CliRunner()

        result = runner.invoke(
            carmack.__main__.carmack_cli, [PREPARE_READS_COMMAND_NAME, "--help"]
        )
        rendered = prepare_reads_flatten_help(result.output)

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(rendered).contains(PREPARE_READS_COMMAND_NAME)
        assert_that(rendered).contains(PREPARE_READS_R1_METAVAR)
        assert_that(rendered).contains(PREPARE_READS_R2_METAVAR)
        assert_that(rendered).contains("--chemistry")


class TestPrepareReadsCliWorkerCount:
    """The option the stage's pool width is set through, and its capped default.

    Both tests that invoke the command patch the constructor rather than run
    a real prepare_reads(): the pool forks, so an object patched in the
    parent records nothing a child did, and a test that asserted on one would
    pass whether or not the wiring worked.
    """

    @pytest.mark.parametrize("option", PREPARE_READS_CPU_COUNT_OPTS)
    def test_worker_count_reaches_the_preparer_as_n_workers(
        self, tmp_path: Path, option: str
    ) -> None:
        """A worker count given on the command line is the width the stage runs at.

        Args:
            tmp_path: Output directory the command is pointed at.
            option: Spelling of the worker option the count is passed under.
        """
        runner = CliRunner()
        with mock.patch("carmack.__main__.ReadPreparer", autospec=True) as mock_preparer:
            result = runner.invoke(
                carmack.__main__.carmack_cli,
                [
                    PREPARE_READS_COMMAND_NAME,
                    DUMMY_R1_FASTQ,
                    DUMMY_R2_FASTQ,
                    "--chemistry",
                    CHEMISTRY,
                    "--output_dir",
                    str(tmp_path),
                    option,
                    str(PREPARE_READS_CLI_WORKERS),
                ],
            )

        assert_that(result.exit_code).is_equal_to(0)
        mock_preparer.assert_called_once_with(
            DUMMY_R1_FASTQ,
            DUMMY_R2_FASTQ,
            CHEMISTRY,
            n_workers=PREPARE_READS_CLI_WORKERS,
        )

    def test_worker_option_defaults_to_the_core_count_capped_at_the_saturation_point(
        self,
    ) -> None:
        """The default is the machine's usable cores, held down to the shared cap.

        Asserted against the cap and the core-count helper rather than a
        literal: on a machine narrower than the cap the correct default is
        the lower number, and a test naming sixteen would be wrong there.
        """
        option = prepare_reads_command_option(
            PREPARE_READS_COMMAND_NAME, PREPARE_READS_CPU_COUNT_OPTION
        )

        assert_that(option.default).is_equal_to(min(DEFAULT_MAX_WORKERS, get_cpu_count()))

    def test_worker_option_follows_the_cpu_count_option_convention(self) -> None:
        """The option is spelled and rendered as assign-targets spells it."""
        option = prepare_reads_command_option(
            PREPARE_READS_COMMAND_NAME, PREPARE_READS_CPU_COUNT_OPTION
        )

        assert_that(option.opts).is_equal_to(PREPARE_READS_CPU_COUNT_OPTS)
        assert_that(option.show_default).is_true()
        assert_that(option.required).is_false()
        assert_that(option.type.name).is_equal_to("integer")

    def test_command_help_renders_the_worker_option_with_its_default(self) -> None:
        """The resolved default survives rendering, so a user reads it off --help."""
        runner = CliRunner()

        result = runner.invoke(
            carmack.__main__.carmack_cli, [PREPARE_READS_COMMAND_NAME, "--help"]
        )
        rendered = prepare_reads_flatten_help(result.output)

        assert_that(result.exit_code).is_equal_to(0)
        assert_that(rendered).contains(*PREPARE_READS_CPU_COUNT_OPTS)
        assert_that(rendered).contains(f"[default: {PREPARE_READS_DEFAULT_CPU_COUNT}]")


# ---------------------------------------------------------------------------
# A hand-built, multi-target golden fixture proving both arms and two target
# buckets end to end.
#
# The real carmack_custom_seq_1_0 chemistry's target index whitelist carries
# exactly one entry (TATAGCCT), so no fixture built from real sequencing data
# could ever exercise more than one scTIP target bucket. tests/test_golden_outputs.py's
# real-data goldens cover the real byte-exact pipeline output; this fixture instead
# covers the property no real library can: that ReadPreparer, handed a real, already
# UMI- and target-annotated FASTQ pair, dispatches correctly across more than one
# matched bucket as well as the unmatched arm. Its provenance is documented in
# tests/data_generators/golden.py alongside the other committed golden inputs.
# ---------------------------------------------------------------------------

MULTI_TARGET_GOLDEN_DIR = Path("tests/data/golden")
MULTI_TARGET_R1_FASTQ = MULTI_TARGET_GOLDEN_DIR / "prepare_reads_multi_target_R1.fastq.gz"
MULTI_TARGET_R2_FASTQ = MULTI_TARGET_GOLDEN_DIR / "prepare_reads_multi_target_R2.fastq.gz"

# Read ids built with no TGIDX tag at all -- the shape a chemistry with no target index
# support emits, as `TestPrepareReadUnmatched.test_read_with_no_tgidx_tag_is_unmatched`
# above covers for a single read in isolation.
MULTI_TARGET_NO_TAG_IDS = ("unmatched_no_tag_1", "unmatched_no_tag_2", "unmatched_no_tag_3")

# Read ids instead carrying an explicit TGIDX=NONE tag with no TGIDX_POS span -- the
# shape a real assign-targets run emits for an unmatched read on a chemistry that DOES
# support target assignment, since `TargetAssigner.assign_read` never writes a span for
# the unassigned sentinel. Covered in isolation by
# `TestPrepareReadUnmatched.test_read_with_tgidx_none_is_unmatched` above.
MULTI_TARGET_EXPLICIT_NONE_IDS = ("unmatched_none_1", "unmatched_none_2", "unmatched_none_3")

# (read_id, tgidx) pairs the fixture is built from: the two unmatched shapes above, plus
# matched reads split across both of ChemistryTwoTargets's whitelist entries. Reused
# directly by the test class below rather than re-derived from the fixture files, so the
# committed fixture and its expected arm/bucket mapping cannot silently drift apart.
MULTI_TARGET_IDS_AND_TARGETS: list[tuple[str, str | None]] = [
    *((read_id, None) for read_id in MULTI_TARGET_NO_TAG_IDS),
    *((read_id, None) for read_id in MULTI_TARGET_EXPLICIT_NONE_IDS),
    *((f"matched_a_{i}", TWO_TARGET_WHITELIST[0]) for i in range(1, 9)),
    *((f"matched_b_{i}", TWO_TARGET_WHITELIST[1]) for i in range(1, 9)),
]


def as_explicit_no_target_read(read: tuple[str, str, str]) -> tuple[str, str, str]:
    """Rewrite a `tgidx=None`-built read to instead carry an explicit TGIDX=NONE tag.

    `build_full_r1_read(tgidx=None)` omits the TGIDX tag entirely. This instead
    reproduces the shape a real assign-targets run emits for an unmatched read on a
    chemistry that supports target assignment: an explicit TGIDX=NONE tag with no
    TGIDX_POS span at all, matching `TargetAssigner.assign_read`, which never writes a
    span for the unassigned sentinel.

    Args:
        read: A `(name, seq, qual)` read built with `tgidx=None`.

    Returns:
        The same read, with an explicit TGIDX=NONE tag added to its header.
    """
    name, seq, qual = read
    ann = ReadAnnotation.parse(name)
    ann.set("TGIDX", NO_TARGET)
    return ann.render(), seq, qual


class TestReadPreparerMultiTargetGoldenEndToEnd:
    """`ReadPreparer` over a hand-built, committed, multi-target FASTQ pair.

    This fixture is deliberately synthetic: no real carmack_custom_seq_1_0 library
    carries more than one confirmed target index, so nothing here claims to prove how a
    real multi-target library would behave. Like `TestAssignTargetsEndToEnd`'s own
    `fast=True` run, it proves one specific, narrower property instead -- here, that
    `ReadPreparer` reads a real, already-annotated FASTQ pair off disk and dispatches
    every read to the correct arm or bucket, including across more than one matched
    bucket, which the real single-entry whitelist could never exercise on its own.
    """

    @pytest.fixture
    def chemistry(self) -> ChemistryTwoTargets:
        """Return the synthetic two-target chemistry the fixture was built under."""
        return ChemistryTwoTargets()

    @pytest.fixture
    def preparer(self, chemistry: ChemistryTwoTargets) -> ReadPreparer:
        """Return a preparer constructed over the committed multi-target fixture files.

        Args:
            chemistry: The synthetic chemistry the factory is patched to resolve.

        Returns:
            The constructed `ReadPreparer`.
        """
        assert_that(MULTI_TARGET_R1_FASTQ.is_file()).described_as(
            str(MULTI_TARGET_R1_FASTQ)
        ).is_true()
        assert_that(MULTI_TARGET_R2_FASTQ.is_file()).described_as(
            str(MULTI_TARGET_R2_FASTQ)
        ).is_true()
        with patch_chemistry(chemistry):
            return ReadPreparer(
                str(MULTI_TARGET_R1_FASTQ), str(MULTI_TARGET_R2_FASTQ), chemistry.name
            )

    def test_every_read_is_written_exactly_once_to_the_correct_arm_or_bucket(
        self, preparer: ReadPreparer, tmp_path: Path
    ) -> None:
        """Every fixture read id lands in exactly one output file, and it is the right one."""
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        expected_unmatched = {
            read_id for read_id, tgidx in MULTI_TARGET_IDS_AND_TARGETS if tgidx is None
        }
        written_unmatched = {
            extract_read_id(record[0][1:])
            for record in read_fastq(none_r1_path(tmp_path, OUT_PREFIX))
        }
        assert_that(written_unmatched).is_equal_to(expected_unmatched)

        for target in TWO_TARGET_WHITELIST:
            expected_ids = {
                read_id for read_id, tgidx in MULTI_TARGET_IDS_AND_TARGETS if tgidx == target
            }
            written_ids = {
                extract_read_id(record[0][1:])
                for record in read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target))
            }
            assert_that(written_ids).is_equal_to(expected_ids)

    def test_both_unmatched_tag_shapes_land_in_the_unmatched_arm(
        self, preparer: ReadPreparer, tmp_path: Path
    ) -> None:
        """A missing TGIDX tag and an explicit TGIDX=NONE tag dispatch identically."""
        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        written_unmatched = {
            extract_read_id(record[0][1:])
            for record in read_fastq(none_r1_path(tmp_path, OUT_PREFIX))
        }
        assert_that(written_unmatched).contains(*MULTI_TARGET_NO_TAG_IDS)
        assert_that(written_unmatched).contains(*MULTI_TARGET_EXPLICIT_NONE_IDS)

    def test_both_arms_and_both_target_buckets_are_exercised(
        self, preparer: ReadPreparer, tmp_path: Path
    ) -> None:
        """The fixture actually drives every arm and bucket it is meant to, not just some.

        Guards the fixture's own composition as much as the stage under test: a fixture
        that accidentally carried zero reads for one arm or bucket would let the other
        assertions in this class pass vacuously.
        """
        target_a, target_b = TWO_TARGET_WHITELIST
        assert_that(any(tgidx is None for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS)).is_true()
        assert_that(any(tgidx == target_a for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS)).is_true()
        assert_that(any(tgidx == target_b for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS)).is_true()

        preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(read_fastq(none_r1_path(tmp_path, OUT_PREFIX))).is_not_empty()
        assert_that(read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target_a))).is_not_empty()
        assert_that(read_fastq(target_r1_path(tmp_path, OUT_PREFIX, target_b))).is_not_empty()

    def test_prepare_stats_reconciles_against_the_known_fixture_composition(
        self, preparer: ReadPreparer, tmp_path: Path
    ) -> None:
        """The reconciling invariant holds, and every count matches the fixture's own design."""
        target_a, target_b = TWO_TARGET_WHITELIST
        expected_unmatched = sum(1 for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS if tgidx is None)
        expected_a = sum(1 for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS if tgidx == target_a)
        expected_b = sum(1 for _, tgidx in MULTI_TARGET_IDS_AND_TARGETS if tgidx == target_b)

        stats = preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        assert_that(stats.total_reads).is_equal_to(len(MULTI_TARGET_IDS_AND_TARGETS))
        assert_that(stats.unmatched_written + stats.matched_written).is_equal_to(stats.total_reads)
        assert_that(stats.unmatched_written).is_equal_to(expected_unmatched)
        assert_that(stats.target_written).is_equal_to({target_a: expected_a, target_b: expected_b})

    def test_prepare_stats_report_on_disk_matches_the_returned_stats(
        self, preparer: ReadPreparer, tmp_path: Path
    ) -> None:
        """The written `.prepare_stats.txt` report matches the stats object byte for byte."""
        stats = preparer.prepare_reads(output_dir=str(tmp_path), prefix=OUT_PREFIX)

        report_on_disk = strip_report_run_details(
            prepare_stats_path(tmp_path, OUT_PREFIX).read_text()
        )
        assert_that(report_on_disk).is_equal_to(strip_report_run_details(stats.get_report()))
