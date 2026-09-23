"""Tests for the linear-dedup module.

The first section below proves that ``tests/data/linear_dedup_fixture.bam`` (plus its ``.bai``
and the small separate missing-CB fixture) exists and is built exactly to the layout documented
below, so the ``LinearDedup`` engine and reporting tests that follow exercise a fixture whose
contents are pinned and independently verified.

Fixture layout
---------------
Two references, both long enough to hold every position used below:

    chr1: 5000bp
    chr2: 5000bp

Every alignment is a plain, ungapped, all-``M`` CIGAR (``reference_end = reference_start +
read length``, no indels/soft-clips); sequence and quality content is arbitrary placeholder
data of the matching length -- nothing in these tests or the future engine reads it. ``CB`` and
``AS`` are set only where the table below gives them a value; a ``None`` means the tag is
omitted from that record entirely (not set to an empty/sentinel value). Only R1 ever carries an
``AS`` tag, matching the engine's R1-only scoring design. Every record is paired
(``is_paired``); ``is_read1``/``is_read2`` match the record's ``mate`` field; a record's
``category`` maps onto pysam flags as ``primary`` -> neither secondary nor supplementary,
``secondary`` -> ``is_secondary``, ``supplementary`` -> ``is_supplementary``. An unmapped record
with a mapped mate still carries the mate's ``chrom``/``reference_start`` (the samtools
coordinate-sort convention for such pairs) but no CIGAR, so it has no ``reference_end``.

``FIXTURE_RECORDS`` (28 primary + 1 supplementary + 1 secondary = 30 reads total) covers, by
QNAME group:

- (a) ``dup_low_as`` / ``dup_high_as`` -- same CB ("CELL_A"), same chrom/strand/position
  (chr1, +, 100); differing R1 ``AS`` (50 vs 90). ``dup_high_as`` is the intended winner.
- (h, supplementary half) ``dup_high_as`` also has a *supplementary* R1 alignment on chr2:50,
  sharing its QNAME with the (surviving) primary -- must never appear in dedup output.
- (b) ``diffcell_A`` / ``diffcell_B`` -- different CB ("CELL_A" / "CELL_B"), identical
  chrom/strand/position (chr1, +, 500). Both must survive (no cross-cell collision).
- (c) ``diffchrom_1`` / ``diffchrom_2`` -- same CB ("CELL_A"), identical strand/position
  (+, 800) but different chrom (chr1 / chr2). Both must survive (chrom is part of the key).
- (h, secondary half) ``diffchrom_1`` also has a *secondary* R1 alignment on chr1:2000, sharing
  its QNAME with the (surviving) primary -- must never appear in dedup output.
- (d) ``revdup_shift`` / ``revdup_base`` -- same CB ("CELL_D"), both R1 reverse-strand on chr1,
  with *different* ``reference_start`` (1200 vs 1230) but the *same* ``reference_end`` (1260,
  from read lengths 60 and 30) -- the same fragment key. ``revdup_base`` (AS=80) is the intended
  winner over ``revdup_shift`` (AS=40). ``filler_between_revdup`` (CB="CELL_X", its own distinct
  key, irrelevant AS=10) has its R1 at chr1:1215 -- strictly between the two revdup R1 starts --
  so in coordinate-sort order the two same-key R1 reads are NOT adjacent, and one of
  ``revdup_base``'s own mate reads (R2 at 1180) sorts even earlier. This is the case that rules
  out a single-pass/windowed design.
- (e) ``noas_loser`` / ``scored_winner`` -- same CB ("CELL_E"), same position (chr1, +, 1500).
  ``noas_loser``'s R1 has no ``AS`` tag; ``scored_winner``'s R1 has ``AS=30`` and must win.
- (f) ``noas_alone`` -- CB "CELL_F", chr1:1700, R1 has no ``AS`` tag and no competitor at its
  key -- must survive alone.
- (i) ``unmapped_r1_mapped_mate`` -- R1 unmapped (chr1:1800, matching its mapped mate's
  position per sort convention), R2 mapped with ``mate_is_unmapped``. ``mapped_r1_unmapped_mate``
  -- R1 mapped (chr1:1900, AS=20) with ``mate_is_unmapped``, R2 unmapped. Both pairs must be
  excluded from grouping and counted, never silently dropped.

A separate, tiny fixture -- ``tests/data/linear_dedup_fixture_missing_cb.bam`` -- holds exactly
one pair, ``missing_cb_r1``: a primary R1 (chr1:100, AS=55) with NO ``CB`` tag at all, and its R2
mate which does carry one ("CELL_Z"). This is kept out of the main fixture so the main fixture
never triggers the engine's fail-fast ``ValueError`` path and stays usable, as-is, by every other
scenario's tests.
"""

import dataclasses
from pathlib import Path

import pysam
import pytest
from assertpy import assert_that

from carmack.linear_dedup.linear_dedup import LinearDedup
from carmack.linear_dedup.linear_dedup_reporting import LinearDedupStats
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME
from carmack.utils import get_prefix
from tests.utils import (
    assert_mqc_payload_file,
    assert_no_mqc_file_bundles_payloads,
    strip_report_run_details,
)

BAM_PATH = "tests/data/linear_dedup_fixture.bam"
BAI_PATH = BAM_PATH + ".bai"

MISSING_CB_BAM_PATH = "tests/data/linear_dedup_fixture_missing_cb.bam"
MISSING_CB_BAI_PATH = MISSING_CB_BAM_PATH + ".bai"

REFERENCES = {"chr1": 5000, "chr2": 5000}

# One entry per alignment record expected in tests/data/linear_dedup_fixture.bam. See the
# module docstring above for the scenario each group covers.
FIXTURE_RECORDS = [
    # --- Scenario (a): same-cell/same-position duplicate pair, differing AS.
    dict(
        qname="dup_low_as",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=100,
        ref_len=50,
        cb="CELL_A",
        as_tag=50,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="dup_low_as",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=250,
        ref_len=50,
        cb="CELL_A",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="dup_high_as",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=100,
        ref_len=50,
        cb="CELL_A",
        as_tag=90,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="dup_high_as",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=260,
        ref_len=50,
        cb="CELL_A",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # (h, supplementary half): shares QNAME with the surviving dup_high_as primary.
    dict(
        qname="dup_high_as",
        mate=1,
        category="supplementary",
        chrom="chr2",
        is_reverse=False,
        ref_start=50,
        ref_len=50,
        cb="CELL_A",
        as_tag=20,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (b): different-cell, same position -- both survive.
    dict(
        qname="diffcell_A",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=500,
        ref_len=50,
        cb="CELL_A",
        as_tag=60,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffcell_A",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=650,
        ref_len=50,
        cb="CELL_A",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffcell_B",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=500,
        ref_len=50,
        cb="CELL_B",
        as_tag=60,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffcell_B",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=650,
        ref_len=50,
        cb="CELL_B",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (c): same-cell, different chromosome -- both survive.
    dict(
        qname="diffchrom_1",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=800,
        ref_len=50,
        cb="CELL_A",
        as_tag=70,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffchrom_1",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=950,
        ref_len=50,
        cb="CELL_A",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # (h, secondary half): shares QNAME with the surviving diffchrom_1 primary.
    dict(
        qname="diffchrom_1",
        mate=1,
        category="secondary",
        chrom="chr1",
        is_reverse=False,
        ref_start=2000,
        ref_len=50,
        cb="CELL_A",
        as_tag=15,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffchrom_2",
        mate=1,
        category="primary",
        chrom="chr2",
        is_reverse=False,
        ref_start=800,
        ref_len=50,
        cb="CELL_A",
        as_tag=70,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="diffchrom_2",
        mate=2,
        category="primary",
        chrom="chr2",
        is_reverse=True,
        ref_start=950,
        ref_len=50,
        cb="CELL_A",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (d): reverse-strand fragment collapsing across non-adjacent positions.
    dict(
        qname="revdup_shift",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1200,
        ref_len=60,
        cb="CELL_D",
        as_tag=40,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="revdup_shift",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1100,
        ref_len=50,
        cb="CELL_D",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="filler_between_revdup",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1215,
        ref_len=50,
        cb="CELL_X",
        as_tag=10,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="filler_between_revdup",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1350,
        ref_len=50,
        cb="CELL_X",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="revdup_base",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1230,
        ref_len=30,
        cb="CELL_D",
        as_tag=80,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="revdup_base",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1180,
        ref_len=50,
        cb="CELL_D",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (e): missing AS loses to a scored competitor at the same key.
    dict(
        qname="noas_loser",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1500,
        ref_len=50,
        cb="CELL_E",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="noas_loser",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1650,
        ref_len=50,
        cb="CELL_E",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="scored_winner",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1500,
        ref_len=50,
        cb="CELL_E",
        as_tag=30,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="scored_winner",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1650,
        ref_len=50,
        cb="CELL_E",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (f): missing AS with no competitor -- survives alone.
    dict(
        qname="noas_alone",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1700,
        ref_len=50,
        cb="CELL_F",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="noas_alone",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=1850,
        ref_len=50,
        cb="CELL_F",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
    # --- Scenario (i): unmapped R1 with a mapped mate, and a mapped R1 with an unmapped mate.
    dict(
        qname="unmapped_r1_mapped_mate",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1800,
        ref_len=None,
        cb="CELL_G",
        as_tag=None,
        unmapped=True,
        mate_unmapped=False,
    ),
    dict(
        qname="unmapped_r1_mapped_mate",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1800,
        ref_len=50,
        cb="CELL_G",
        as_tag=None,
        unmapped=False,
        mate_unmapped=True,
    ),
    dict(
        qname="mapped_r1_unmapped_mate",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1900,
        ref_len=50,
        cb="CELL_H",
        as_tag=20,
        unmapped=False,
        mate_unmapped=True,
    ),
    dict(
        qname="mapped_r1_unmapped_mate",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=1900,
        ref_len=None,
        cb="CELL_H",
        as_tag=None,
        unmapped=True,
        mate_unmapped=False,
    ),
]

# The single pair in tests/data/linear_dedup_fixture_missing_cb.bam -- see the module docstring.
MISSING_CB_RECORDS = [
    dict(
        qname="missing_cb_r1",
        mate=1,
        category="primary",
        chrom="chr1",
        is_reverse=False,
        ref_start=100,
        ref_len=50,
        cb=None,
        as_tag=55,
        unmapped=False,
        mate_unmapped=False,
    ),
    dict(
        qname="missing_cb_r1",
        mate=2,
        category="primary",
        chrom="chr1",
        is_reverse=True,
        ref_start=250,
        ref_len=50,
        cb="CELL_Z",
        as_tag=None,
        unmapped=False,
        mate_unmapped=False,
    ),
]


def record_id(record: dict) -> str:
    """Build a readable pytest id for one FIXTURE_RECORDS entry."""
    return f"{record['qname']}-mate{record['mate']}-{record['category']}"


def load_records(path: str, index_path: str) -> list:
    """Open an indexed BAM and materialize every alignment record into a list."""
    with pysam.AlignmentFile(path, "rb", index_filename=index_path) as bam:
        return list(bam)


def lookup_by_qname_mate_category(records: list) -> dict:
    """Key materialized alignment records by (qname, mate number, primary/secondary/supplementary)."""
    lookup = {}
    for read in records:
        mate = 1 if read.is_read1 else 2
        if read.is_secondary:
            category = "secondary"
        elif read.is_supplementary:
            category = "supplementary"
        else:
            category = "primary"
        lookup[(read.query_name, mate, category)] = read
    return lookup


class TestLinearDedupFixtureExists:
    """Sanity checks that the fixture BAM exists, opens, and is indexed."""

    def test_fixture_bam_opens_and_is_indexed(self):
        records = load_records(BAM_PATH, BAI_PATH)
        assert_that(records).described_as(
            "records read from linear_dedup_fixture.bam"
        ).is_not_empty()

    def test_fixture_bam_has_expected_references(self):
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as bam:
            reference_lengths = dict(zip(bam.references, bam.lengths))
        assert_that(reference_lengths).described_as("fixture BAM references").is_equal_to(
            REFERENCES
        )

    def test_fixture_bam_read_count_matches_designed_layout(self):
        records = load_records(BAM_PATH, BAI_PATH)
        assert_that(len(records)).described_as(
            "total reads in linear_dedup_fixture.bam"
        ).is_equal_to(len(FIXTURE_RECORDS))


class TestLinearDedupFixtureLayout:
    """Per-record structural checks proving every designed record is present as specified."""

    @pytest.fixture(scope="class")
    def lookup(self) -> dict:
        return lookup_by_qname_mate_category(load_records(BAM_PATH, BAI_PATH))

    @pytest.mark.parametrize("record", FIXTURE_RECORDS, ids=record_id)
    def test_record_matches_designed_layout(self, lookup, record):
        key = (record["qname"], record["mate"], record["category"])
        assert_that(lookup).described_as(f"fixture lookup for {record_id(record)}").contains_key(
            key
        )
        read = lookup[key]

        assert_that(read.is_paired).described_as("is_paired").is_true()
        assert_that(read.is_read1).described_as("is_read1").is_equal_to(record["mate"] == 1)
        assert_that(read.is_read2).described_as("is_read2").is_equal_to(record["mate"] == 2)
        assert_that(read.is_secondary).described_as("is_secondary").is_equal_to(
            record["category"] == "secondary"
        )
        assert_that(read.is_supplementary).described_as("is_supplementary").is_equal_to(
            record["category"] == "supplementary"
        )
        assert_that(read.is_unmapped).described_as("is_unmapped").is_equal_to(record["unmapped"])
        assert_that(read.mate_is_unmapped).described_as("mate_is_unmapped").is_equal_to(
            record["mate_unmapped"]
        )
        assert_that(read.is_reverse).described_as("is_reverse").is_equal_to(record["is_reverse"])
        assert_that(read.reference_name).described_as("reference_name").is_equal_to(
            record["chrom"]
        )
        assert_that(read.reference_start).described_as("reference_start").is_equal_to(
            record["ref_start"]
        )

        if record["ref_len"] is not None:
            assert_that(read.reference_end).described_as("reference_end").is_equal_to(
                record["ref_start"] + record["ref_len"]
            )

        if record["cb"] is None:
            assert_that(read.has_tag("CB")).described_as("has CB tag").is_false()
        else:
            assert_that(read.has_tag("CB")).described_as("has CB tag").is_true()
            assert_that(read.get_tag("CB")).described_as("CB tag value").is_equal_to(record["cb"])

        if record["as_tag"] is None:
            assert_that(read.has_tag("AS")).described_as("has AS tag").is_false()
        else:
            assert_that(read.has_tag("AS")).described_as("has AS tag").is_true()
            assert_that(float(read.get_tag("AS"))).described_as("AS tag value").is_equal_to(
                float(record["as_tag"])
            )


class TestLinearDedupFixtureScenarios:
    """Cross-record checks proving each acceptance scenario is actually present in the fixture."""

    @pytest.fixture(scope="class")
    def lookup(self) -> dict:
        return lookup_by_qname_mate_category(load_records(BAM_PATH, BAI_PATH))

    def fragment_key(self, read: pysam.AlignedSegment) -> tuple:
        """Mirror the engine's planned fragment_key so scenarios can be checked without it."""
        return (
            read.get_tag("CB"),
            read.reference_name,
            read.is_reverse,
            read.reference_end if read.is_reverse else read.reference_start,
        )

    def test_scenario_a_same_cell_same_position_duplicate_pair(self, lookup):
        low = lookup[("dup_low_as", 1, "primary")]
        high = lookup[("dup_high_as", 1, "primary")]

        assert_that(self.fragment_key(low)).described_as(
            "dup_low_as vs dup_high_as fragment key"
        ).is_equal_to(self.fragment_key(high))
        assert_that(float(high.get_tag("AS"))).described_as(
            "dup_high_as AS should exceed dup_low_as AS"
        ).is_greater_than(float(low.get_tag("AS")))

    def test_scenario_b_different_cell_same_position_both_present(self, lookup):
        cell_a = lookup[("diffcell_A", 1, "primary")]
        cell_b = lookup[("diffcell_B", 1, "primary")]

        assert_that(cell_a.get_tag("CB")).described_as("diffcell_A CB").is_not_equal_to(
            cell_b.get_tag("CB")
        )
        assert_that(
            (cell_a.reference_name, cell_a.is_reverse, cell_a.reference_start)
        ).described_as("diffcell_A/B share position").is_equal_to(
            (cell_b.reference_name, cell_b.is_reverse, cell_b.reference_start)
        )

    def test_scenario_c_same_cell_different_chromosome_both_present(self, lookup):
        chrom_1 = lookup[("diffchrom_1", 1, "primary")]
        chrom_2 = lookup[("diffchrom_2", 1, "primary")]

        assert_that(chrom_1.get_tag("CB")).described_as("diffchrom_1 CB").is_equal_to(
            chrom_2.get_tag("CB")
        )
        assert_that(chrom_1.reference_name).described_as(
            "diffchrom_1/2 reference_name must differ"
        ).is_not_equal_to(chrom_2.reference_name)
        assert_that(self.fragment_key(chrom_1)).described_as(
            "diffchrom_1/2 fragment keys must differ (chrom is part of the key)"
        ).is_not_equal_to(self.fragment_key(chrom_2))

    def test_scenario_d_reverse_strand_same_end_different_start(self, lookup):
        shift = lookup[("revdup_shift", 1, "primary")]
        base = lookup[("revdup_base", 1, "primary")]

        assert_that(shift.is_reverse).described_as("revdup_shift is_reverse").is_true()
        assert_that(base.is_reverse).described_as("revdup_base is_reverse").is_true()
        assert_that(shift.reference_start).described_as(
            "revdup_shift/base reference_start must differ"
        ).is_not_equal_to(base.reference_start)
        assert_that(shift.reference_end).described_as(
            "revdup_shift/base reference_end must match (same fragment key)"
        ).is_equal_to(base.reference_end)
        assert_that(self.fragment_key(shift)).described_as(
            "revdup_shift/base fragment key"
        ).is_equal_to(self.fragment_key(base))
        assert_that(float(base.get_tag("AS"))).described_as(
            "revdup_base AS should exceed revdup_shift AS"
        ).is_greater_than(float(shift.get_tag("AS")))

    def test_scenario_d_duplicate_group_is_non_adjacent_in_coordinate_sort_order(self):
        records = load_records(BAM_PATH, BAI_PATH)
        qnames_in_file_order = [read.query_name for read in records]

        # R1 and R2 share a QNAME, so a QNAME-only lookup can latch onto either mate.
        # LinearDedup only ever operates on R1 records, so isolate each QNAME's R1
        # position specifically.
        shift_index = next(
            index
            for index, read in enumerate(records)
            if read.query_name == "revdup_shift" and read.is_read1
        )
        base_index = next(
            index
            for index, read in enumerate(records)
            if read.query_name == "revdup_base" and read.is_read1
        )
        between = qnames_in_file_order[
            min(shift_index, base_index) + 1 : max(shift_index, base_index)
        ]

        assert_that(between).described_as(
            "records between revdup_shift's and revdup_base's first occurrence in file order"
        ).is_not_empty()
        assert_that("filler_between_revdup" in between).described_as(
            "filler_between_revdup must sort strictly between the two revdup R1 reads"
        ).is_true()

    def test_scenario_e_missing_as_loses_to_scored_competitor(self, lookup):
        loser = lookup[("noas_loser", 1, "primary")]
        winner = lookup[("scored_winner", 1, "primary")]

        assert_that(loser.has_tag("AS")).described_as("noas_loser has no AS tag").is_false()
        assert_that(winner.has_tag("AS")).described_as("scored_winner has an AS tag").is_true()
        assert_that(self.fragment_key(loser)).described_as(
            "noas_loser vs scored_winner fragment key"
        ).is_equal_to(self.fragment_key(winner))

    def test_scenario_f_missing_as_with_no_competitor_survives_alone(self, lookup):
        alone = lookup[("noas_alone", 1, "primary")]
        assert_that(alone.has_tag("AS")).described_as("noas_alone has no AS tag").is_false()

        all_primary_r1_keys = [
            self.fragment_key(read)
            for read in load_records(BAM_PATH, BAI_PATH)
            if read.is_read1 and not read.is_secondary and not read.is_supplementary
            if not read.is_unmapped and not read.mate_is_unmapped
        ]
        assert_that(all_primary_r1_keys.count(self.fragment_key(alone))).described_as(
            "noas_alone's fragment key must be unique across the fixture"
        ).is_equal_to(1)

    def test_scenario_h_supplementary_alignment_shares_qname_with_winner(self, lookup):
        primary = lookup[("dup_high_as", 1, "primary")]
        supplementary = lookup[("dup_high_as", 1, "supplementary")]

        assert_that(supplementary.query_name).described_as(
            "supplementary QNAME matches its winning primary"
        ).is_equal_to(primary.query_name)
        assert_that(supplementary.is_supplementary).described_as("supplementary flag").is_true()
        assert_that(supplementary.is_secondary).described_as(
            "supplementary record must not also be flagged secondary"
        ).is_false()

    def test_scenario_h_secondary_alignment_shares_qname_with_winner(self, lookup):
        primary = lookup[("diffchrom_1", 1, "primary")]
        secondary = lookup[("diffchrom_1", 1, "secondary")]

        assert_that(secondary.query_name).described_as(
            "secondary QNAME matches its winning primary"
        ).is_equal_to(primary.query_name)
        assert_that(secondary.is_secondary).described_as("secondary flag").is_true()
        assert_that(secondary.is_supplementary).described_as(
            "secondary record must not also be flagged supplementary"
        ).is_false()

    def test_scenario_i_unmapped_r1_with_mapped_mate(self, lookup):
        r1 = lookup[("unmapped_r1_mapped_mate", 1, "primary")]
        r2 = lookup[("unmapped_r1_mapped_mate", 2, "primary")]

        assert_that(r1.is_unmapped).described_as("unmapped R1 is_unmapped").is_true()
        assert_that(r2.is_unmapped).described_as("its mapped mate is_unmapped").is_false()
        assert_that(r2.mate_is_unmapped).described_as("mapped mate's mate_is_unmapped").is_true()

    def test_scenario_i_mapped_r1_with_unmapped_mate(self, lookup):
        r1 = lookup[("mapped_r1_unmapped_mate", 1, "primary")]
        r2 = lookup[("mapped_r1_unmapped_mate", 2, "primary")]

        assert_that(r1.is_unmapped).described_as("mapped R1 is_unmapped").is_false()
        assert_that(r1.mate_is_unmapped).described_as("mapped R1's mate_is_unmapped").is_true()
        assert_that(r2.is_unmapped).described_as("its unmapped mate is_unmapped").is_true()

    def test_main_fixture_never_triggers_missing_cb_failure(self):
        """The main fixture must stay usable everywhere: every groupable primary R1 has a CB tag."""
        records = load_records(BAM_PATH, BAI_PATH)
        for read in records:
            if not read.is_read1 or read.is_secondary or read.is_supplementary:
                continue
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            assert_that(read.has_tag("CB")).described_as(
                f"{read.query_name} (a groupable primary R1) must carry a CB tag"
            ).is_true()


class TestLinearDedupMissingCbFixture:
    """Sanity checks for the separate missing-CB fixture (scenario g)."""

    def test_missing_cb_fixture_opens_and_is_indexed(self):
        records = load_records(MISSING_CB_BAM_PATH, MISSING_CB_BAI_PATH)
        assert_that(len(records)).described_as(
            "total reads in linear_dedup_fixture_missing_cb.bam"
        ).is_equal_to(len(MISSING_CB_RECORDS))

    def test_missing_cb_fixture_primary_r1_has_no_cb_tag(self):
        lookup = lookup_by_qname_mate_category(
            load_records(MISSING_CB_BAM_PATH, MISSING_CB_BAI_PATH)
        )
        r1 = lookup[("missing_cb_r1", 1, "primary")]
        r2 = lookup[("missing_cb_r1", 2, "primary")]

        assert_that(r1.is_read1).described_as("is_read1").is_true()
        assert_that(r1.is_secondary).described_as("is_secondary").is_false()
        assert_that(r1.is_supplementary).described_as("is_supplementary").is_false()
        assert_that(r1.is_unmapped).described_as("is_unmapped").is_false()
        assert_that(r1.mate_is_unmapped).described_as("mate_is_unmapped").is_false()
        assert_that(r1.has_tag("CB")).described_as(
            "primary R1 must have no CB tag at all"
        ).is_false()
        assert_that(r2.has_tag("CB")).described_as("its mate does carry a CB tag").is_true()


# ---------------------------------------------------------------------------------------------
# LinearDedup engine tests: pass 1 of the two-pass design.
#
# Everything below this point exercises carmack.linear_dedup.linear_dedup.LinearDedup and
# carmack.linear_dedup.linear_dedup_reporting.LinearDedupStats.
# The scenario labels (a)-(i) match the module docstring above and plans/linear-dedup.md.
# ---------------------------------------------------------------------------------------------

# Reconciling totals for tests/data/linear_dedup_fixture.bam, derived from FIXTURE_RECORDS:
# every mate-1 record (16: 12 groupable primaries + 2 non-primary R1 alignments + 2 primary R1s
# whose pair is unmapped on one side) is scanned in pass 1, and each is classified into exactly
# one of eligible / skipped_non_primary / skipped_unmapped / skipped_unpaired.
FIXTURE_TOTAL_PAIRS = 16
FIXTURE_ELIGIBLE_PAIRS = 12
FIXTURE_SKIPPED_NON_PRIMARY = 2
FIXTURE_SKIPPED_UNMAPPED = 2
FIXTURE_SKIPPED_UNPAIRED = 0

# Per-chromosome breakdown of the 12 eligible primary R1s above, derived from FIXTURE_RECORDS:
# 11 on chr1 (every eligible group except diffchrom_2) and 1 on chr2 (diffchrom_2 alone).
FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME = {"chr1": 11, "chr2": 1}

# Per-chromosome breakdown of the 9 winning fragment-key groups pass 1 resolves from those 12
# eligible reads (dup_high_as, diffcell_A, diffcell_B, diffchrom_1, filler_between_revdup,
# revdup_base, scored_winner, noas_alone on chr1; diffchrom_2 alone on chr2).
FIXTURE_PAIRS_KEPT_BY_CHROMOSOME = {"chr1": 8, "chr2": 1}

# References for small, in-test synthetic BAMs that cover scenarios the committed fixtures
# cannot (an exact AS tie, and a non-default barcode tag) without extending those fixtures.
SYNTHETIC_REFERENCES = {"chr1": 10000}


def build_synthetic_header(references: dict) -> pysam.AlignmentHeader:
    """Build a coordinate-sorted-order header for an in-test synthetic BAM."""
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": chrom, "LN": length} for chrom, length in references.items()],
        }
    )


def make_pair(
    header: pysam.AlignmentHeader,
    qname: str,
    chrom: str,
    r1_pos: int,
    r1_reverse: bool,
    r2_pos: int,
    r2_reverse: bool,
    r1_tags: dict | None = None,
    r2_tags: dict | None = None,
    read_length: int = 50,
) -> tuple[pysam.AlignedSegment, pysam.AlignedSegment]:
    """Build one primary, paired, mapped R1/R2 pair for an in-test synthetic BAM."""

    def build_segment(is_read1, pos, reverse, mate_pos, mate_reverse, tags):
        segment = pysam.AlignedSegment(header)
        segment.query_name = qname
        segment.is_paired = True
        segment.is_read1 = is_read1
        segment.is_read2 = not is_read1
        segment.is_reverse = reverse
        segment.reference_id = header.get_tid(chrom)
        segment.reference_start = pos
        segment.cigartuples = [(0, read_length)]
        segment.mapping_quality = 60
        segment.query_sequence = "A" * read_length
        segment.query_qualities = pysam.qualitystring_to_array("I" * read_length)
        segment.next_reference_id = header.get_tid(chrom)
        segment.next_reference_start = mate_pos
        segment.mate_is_reverse = mate_reverse
        if tags:
            segment.set_tags(list(tags.items()))
        return segment

    r1 = build_segment(True, r1_pos, r1_reverse, r2_pos, r2_reverse, r1_tags)
    r2 = build_segment(False, r2_pos, r2_reverse, r1_pos, r1_reverse, r2_tags)
    return r1, r2


def write_indexed_bam(
    header: pysam.AlignmentHeader, path: Path, segments: list
) -> tuple[str, str]:
    """Write pre-ordered segments (already in valid coordinate-sort order) to an indexed BAM."""
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for segment in segments:
            bam.write(segment)
    pysam.index(str(path))
    return str(path), str(path) + ".bai"


class TestLinearDedupConstruction:
    """The barcode_tag constructor parameter: default and override."""

    def test_barcode_tag_defaults_to_cb(self):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        assert_that(engine.barcode_tag).described_as("default barcode_tag").is_equal_to("CB")

    def test_barcode_tag_can_be_overridden(self):
        engine = LinearDedup(BAM_PATH, BAI_PATH, barcode_tag="BC")
        assert_that(engine.barcode_tag).described_as("overridden barcode_tag").is_equal_to("BC")


class TestFragmentKey:
    """fragment_key(read): (barcode, chrom, is_reverse, pos), strand-aware position."""

    @pytest.fixture(scope="class")
    def engine(self) -> LinearDedup:
        return LinearDedup(BAM_PATH, BAI_PATH)

    @pytest.fixture(scope="class")
    def lookup(self) -> dict:
        return lookup_by_qname_mate_category(load_records(BAM_PATH, BAI_PATH))

    def test_fragment_key_forward_strand_uses_reference_start(self, engine, lookup):
        read = lookup[("dup_high_as", 1, "primary")]
        assert_that(engine.fragment_key(read)).described_as(
            "forward-strand fragment key"
        ).is_equal_to(("CELL_A", "chr1", False, 100))

    def test_fragment_key_reverse_strand_uses_reference_end(self, engine, lookup):
        read = lookup[("revdup_base", 1, "primary")]
        assert_that(engine.fragment_key(read)).described_as(
            "reverse-strand fragment key"
        ).is_equal_to(("CELL_D", "chr1", True, 1260))


class TestReadScore:
    """read_score(read): float(AS) when present, -inf (has_tag-guarded) when absent."""

    @pytest.fixture(scope="class")
    def engine(self) -> LinearDedup:
        return LinearDedup(BAM_PATH, BAI_PATH)

    @pytest.fixture(scope="class")
    def lookup(self) -> dict:
        return lookup_by_qname_mate_category(load_records(BAM_PATH, BAI_PATH))

    def test_read_score_returns_as_tag_as_float(self, engine, lookup):
        read = lookup[("dup_high_as", 1, "primary")]
        assert_that(engine.read_score(read)).described_as("AS-tagged read score").is_equal_to(90.0)

    def test_read_score_returns_negative_infinity_when_as_missing(self, engine, lookup):
        read = lookup[("noas_loser", 1, "primary")]
        assert_that(engine.read_score(read)).described_as("read score with no AS tag").is_equal_to(
            float("-inf")
        )


class TestFindBestReads:
    """find_best_reads(input_bam): winners set + LinearDedupStats, over the main fixture."""

    @pytest.fixture(scope="class")
    def winners_and_stats(self) -> tuple:
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as bam:
            return engine.find_best_reads(bam)

    @pytest.fixture(scope="class")
    def winners(self, winners_and_stats) -> set:
        return winners_and_stats[0]

    @pytest.fixture(scope="class")
    def stats(self, winners_and_stats) -> LinearDedupStats:
        return winners_and_stats[1]

    def test_returns_a_winners_set_and_a_stats_object(self, winners, stats):
        assert_that(isinstance(winners, set)).described_as("winners is a set").is_true()
        assert_that(isinstance(stats, LinearDedupStats)).described_as(
            "stats is a LinearDedupStats"
        ).is_true()

    def test_scenario_a_higher_as_wins(self, winners):
        assert_that(winners).described_as("winners").contains("dup_high_as")
        assert_that(winners).described_as("winners").does_not_contain("dup_low_as")

    def test_scenario_b_different_cell_same_position_both_survive(self, winners):
        assert_that(winners).described_as("winners").contains("diffcell_A")
        assert_that(winners).described_as("winners").contains("diffcell_B")

    def test_scenario_c_same_cell_different_chromosome_both_survive(self, winners):
        assert_that(winners).described_as("winners").contains("diffchrom_1")
        assert_that(winners).described_as("winners").contains("diffchrom_2")

    def test_scenario_d_reverse_strand_collapse_keeps_only_higher_as_winner(self, winners):
        assert_that(winners).described_as("winners").contains("revdup_base")
        assert_that(winners).described_as("winners").does_not_contain("revdup_shift")

    def test_scenario_e_missing_as_loses_to_scored_competitor(self, winners):
        assert_that(winners).described_as("winners").contains("scored_winner")
        assert_that(winners).described_as("winners").does_not_contain("noas_loser")

    def test_scenario_f_missing_as_with_no_competitor_survives_alone(self, winners):
        assert_that(winners).described_as("winners").contains("noas_alone")

    def test_scenario_h_secondary_and_supplementary_are_counted_as_non_primary(
        self, winners, stats
    ):
        assert_that(stats.skipped_non_primary).described_as(
            "skipped_non_primary (one secondary R1, one supplementary R1)"
        ).is_equal_to(FIXTURE_SKIPPED_NON_PRIMARY)

    def test_scenario_h_secondary_and_supplementary_do_not_disturb_the_real_winner(self, winners):
        # dup_high_as's own supplementary R1 sits at chr2:50 with AS=20 -- a lower score than
        # dup_low_as's AS=50. If pass 1 folded it into scoring as an eligible competitor, the
        # weaker of the two real contenders (dup_low_as) could still win on some other basis;
        # instead the true primary pair alone decides the group, exactly as in scenario (a).
        assert_that(winners).described_as("winners").contains("dup_high_as")
        assert_that(winners).described_as("winners").does_not_contain("dup_low_as")

    def test_scenario_i_unmapped_r1_and_mapped_r1_with_unmapped_mate_are_excluded(self, winners):
        assert_that(winners).described_as("winners").does_not_contain("unmapped_r1_mapped_mate")
        assert_that(winners).described_as("winners").does_not_contain("mapped_r1_unmapped_mate")

    def test_scenario_i_unmapped_pairs_are_counted_never_raise(self, stats):
        assert_that(stats.skipped_unmapped).described_as(
            "skipped_unmapped (unmapped R1, and mapped R1 with unmapped mate)"
        ).is_equal_to(FIXTURE_SKIPPED_UNMAPPED)

    def test_stats_counts_reconcile_for_the_fixture(self, stats):
        assert_that(stats.total_pairs).described_as("total_pairs").is_equal_to(FIXTURE_TOTAL_PAIRS)
        assert_that(stats.eligible_pairs).described_as("eligible_pairs").is_equal_to(
            FIXTURE_ELIGIBLE_PAIRS
        )
        assert_that(stats.skipped_non_primary).described_as("skipped_non_primary").is_equal_to(
            FIXTURE_SKIPPED_NON_PRIMARY
        )
        assert_that(stats.skipped_unmapped).described_as("skipped_unmapped").is_equal_to(
            FIXTURE_SKIPPED_UNMAPPED
        )
        assert_that(stats.skipped_unpaired).described_as("skipped_unpaired").is_equal_to(
            FIXTURE_SKIPPED_UNPAIRED
        )

    def test_eligible_pairs_by_chromosome_reconciles_with_fixture(self, stats):
        # Per plans/linear-dedup.md, the per-chromosome breakdown dicts are "counted for free
        # inside the pass-1 loop" -- eligibility is already known per R1 read at that point, so
        # find_best_reads must populate this without any extra pass.
        assert_that(stats.eligible_pairs_by_chromosome).described_as(
            "eligible_pairs_by_chromosome"
        ).is_equal_to(FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME)

    def test_pairs_kept_by_chromosome_reconciles_with_fixture(self, stats):
        # Likewise, pass 1 already determines the winning read per fragment key, so it can bucket
        # each winner's chromosome for free -- no need to defer this to pass 2.
        assert_that(stats.pairs_kept_by_chromosome).described_as(
            "pairs_kept_by_chromosome"
        ).is_equal_to(FIXTURE_PAIRS_KEPT_BY_CHROMOSOME)

    def test_stats_invariant_holds(self, stats):
        assert_that(
            stats.eligible_pairs
            + stats.skipped_unmapped
            + stats.skipped_non_primary
            + stats.skipped_unpaired
        ).described_as(
            "eligible_pairs + skipped_unmapped + skipped_non_primary + skipped_unpaired"
        ).is_equal_to(
            stats.total_pairs
        )


class TestFindBestReadsMissingBarcode:
    """A primary R1 missing its barcode tag raises ValueError, fail-fast, pass 1."""

    def test_missing_default_barcode_tag_raises_value_error(self):
        engine = LinearDedup(MISSING_CB_BAM_PATH, MISSING_CB_BAI_PATH)
        with pysam.AlignmentFile(
            MISSING_CB_BAM_PATH, "rb", index_filename=MISSING_CB_BAI_PATH
        ) as bam:
            with pytest.raises(ValueError):
                engine.find_best_reads(bam)


class TestFindBestReadsUnpaired:
    """A primary R1 record that is not paired is skipped and counted, never raises."""

    def test_unpaired_r1_is_skipped_and_counted(self, tmp_path: Path):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        segment = pysam.AlignedSegment(header)
        segment.query_name = "unpaired_read"
        segment.is_paired = False
        # SAM's read1/read2 flags are only meaningful once is_paired is set, but the
        # scanner keys off is_read1 alone before it ever looks at is_paired, so this
        # exercises the not-is_paired skip branch on an otherwise well-formed record.
        segment.is_read1 = True
        segment.reference_id = header.get_tid("chr1")
        segment.reference_start = 6000
        segment.cigartuples = [(0, 50)]
        segment.mapping_quality = 60
        segment.query_sequence = "A" * 50
        segment.query_qualities = pysam.qualitystring_to_array("I" * 50)
        segment.set_tags([("CB", "CELL_UNPAIRED"), ("AS", 10)])
        bam_path, bai_path = write_indexed_bam(header, tmp_path / "unpaired.bam", [segment])

        engine = LinearDedup(bam_path, bai_path)
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            winners, stats = engine.find_best_reads(bam)

        assert_that(winners).described_as("unpaired winners").does_not_contain("unpaired_read")
        assert_that(stats.skipped_unpaired).described_as("skipped_unpaired").is_equal_to(1)
        assert_that(stats.total_pairs).described_as("total_pairs").is_equal_to(1)


class TestFindBestReadsFullyUnmappedPair:
    """A pair whose R1 and R2 are both unmapped sorts to reference_id -1, past every mapped
    record, in a coordinate-sorted BAM. pysam.AlignmentFile.fetch() with no arguments stops
    before that trailing block unless until_eof=True is passed -- so this pair must still be
    seen, counted, and excluded, not silently invisible to the scan."""

    def test_fully_unmapped_pair_is_seen_counted_and_excluded(self, tmp_path: Path):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        anchor_r1, anchor_r2 = make_pair(
            header,
            "anchor",
            "chr1",
            r1_pos=7000,
            r1_reverse=False,
            r2_pos=7200,
            r2_reverse=True,
            r1_tags={"CB": "CELL_ANCHOR", "AS": 25},
        )

        def build_unmapped_segment(is_read1: bool) -> pysam.AlignedSegment:
            segment = pysam.AlignedSegment(header)
            segment.query_name = "fully_unmapped"
            segment.is_paired = True
            segment.is_read1 = is_read1
            segment.is_read2 = not is_read1
            segment.is_unmapped = True
            segment.mate_is_unmapped = True
            segment.reference_id = -1
            segment.reference_start = -1
            segment.next_reference_id = -1
            segment.next_reference_start = -1
            segment.mapping_quality = 0
            segment.query_sequence = "A" * 50
            segment.query_qualities = pysam.qualitystring_to_array("I" * 50)
            segment.set_tags([("CB", "CELL_UNMAPPED")])
            return segment

        unmapped_r1 = build_unmapped_segment(is_read1=True)
        unmapped_r2 = build_unmapped_segment(is_read1=False)
        # The fully-unmapped pair (reference_id -1) must come after every mapped record for the
        # file to stay in valid coordinate-sort order -- exactly the trailing block that a plain
        # fetch() (no until_eof) fails to reach.
        bam_path, bai_path = write_indexed_bam(
            header,
            tmp_path / "fully_unmapped.bam",
            [anchor_r1, anchor_r2, unmapped_r1, unmapped_r2],
        )

        engine = LinearDedup(bam_path, bai_path)
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            winners, stats = engine.find_best_reads(bam)

        assert_that(stats.total_pairs).described_as(
            "total_pairs (anchor R1 + fully-unmapped R1)"
        ).is_equal_to(2)
        assert_that(stats.skipped_unmapped).described_as(
            "skipped_unmapped (the fully-unmapped pair's R1)"
        ).is_equal_to(1)
        assert_that(stats.eligible_pairs).described_as("eligible_pairs").is_equal_to(1)
        assert_that(winners).described_as("winners").contains("anchor")
        assert_that(winners).described_as("winners").does_not_contain("fully_unmapped")


class TestFindBestReadsTieBreak:
    """Exact AS ties keep whichever read was seen first (strict >, not >=)."""

    def test_find_best_reads_tie_keeps_first_seen_read(self, tmp_path: Path):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        first_r1, first_r2 = make_pair(
            header,
            "tie_first",
            "chr1",
            r1_pos=5000,
            r1_reverse=False,
            r2_pos=5200,
            r2_reverse=True,
            r1_tags={"CB": "CELL_TIE", "AS": 42},
        )
        second_r1, second_r2 = make_pair(
            header,
            "tie_second",
            "chr1",
            r1_pos=5000,
            r1_reverse=False,
            r2_pos=5200,
            r2_reverse=True,
            r1_tags={"CB": "CELL_TIE", "AS": 42},
        )
        # tie_first's R1 is written (and therefore scanned) before tie_second's R1; both R2s
        # sort after both R1s so the file stays in valid, non-decreasing coordinate order.
        bam_path, bai_path = write_indexed_bam(
            header,
            tmp_path / "tie.bam",
            [first_r1, second_r1, first_r2, second_r2],
        )

        engine = LinearDedup(bam_path, bai_path)
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            winners, _ = engine.find_best_reads(bam)

        assert_that(winners).described_as("tie winners").contains("tie_first")
        assert_that(winners).described_as("tie winners").does_not_contain("tie_second")


MULTI_CONTIG_REFERENCES = {"chr1": 5000, "chr2": 5000, "chr3": 5000}


class TestFindBestReadsMultiContigPartitioning:
    """find_best_reads scans one contig at a time internally (see the module docstring) --
    this proves that partitioning never leaks a winner or a stats count across contigs, and
    that a trailing no-coordinate pair is still reconciled into stats without ever winning."""

    def test_winners_and_per_chromosome_stats_do_not_leak_across_contigs(self, tmp_path: Path):
        header = build_synthetic_header(MULTI_CONTIG_REFERENCES)

        # chr1: a duplicate pair -- one winner (the higher-AS read).
        chr1_low_r1, chr1_low_r2 = make_pair(
            header,
            "chr1_low",
            "chr1",
            r1_pos=100,
            r1_reverse=False,
            r2_pos=300,
            r2_reverse=True,
            r1_tags={"CB": "CELL_A", "AS": 10},
        )
        chr1_high_r1, chr1_high_r2 = make_pair(
            header,
            "chr1_high",
            "chr1",
            r1_pos=100,
            r1_reverse=False,
            r2_pos=300,
            r2_reverse=True,
            r1_tags={"CB": "CELL_A", "AS": 99},
        )
        # chr2: a lone read at the same (chrom-agnostic) position/cell as chr1's group --
        # must not collide with it, since chrom is part of the fragment key.
        chr2_alone_r1, chr2_alone_r2 = make_pair(
            header,
            "chr2_alone",
            "chr2",
            r1_pos=100,
            r1_reverse=False,
            r2_pos=300,
            r2_reverse=True,
            r1_tags={"CB": "CELL_A", "AS": 5},
        )
        # chr3: two different cells at the same position -- both winners.
        chr3_cell_c_r1, chr3_cell_c_r2 = make_pair(
            header,
            "chr3_cell_c",
            "chr3",
            r1_pos=100,
            r1_reverse=False,
            r2_pos=300,
            r2_reverse=True,
            r1_tags={"CB": "CELL_C", "AS": 20},
        )
        chr3_cell_d_r1, chr3_cell_d_r2 = make_pair(
            header,
            "chr3_cell_d",
            "chr3",
            r1_pos=100,
            r1_reverse=False,
            r2_pos=300,
            r2_reverse=True,
            r1_tags={"CB": "CELL_D", "AS": 20},
        )

        def build_fully_unmapped_segment(is_read1: bool) -> pysam.AlignedSegment:
            segment = pysam.AlignedSegment(header)
            segment.query_name = "fully_unmapped"
            segment.is_paired = True
            segment.is_read1 = is_read1
            segment.is_read2 = not is_read1
            segment.is_unmapped = True
            segment.mate_is_unmapped = True
            segment.reference_id = -1
            segment.reference_start = -1
            segment.next_reference_id = -1
            segment.next_reference_start = -1
            segment.mapping_quality = 0
            segment.query_sequence = "A" * 50
            segment.query_qualities = pysam.qualitystring_to_array("I" * 50)
            segment.set_tags([("CB", "CELL_UNMAPPED")])
            return segment

        unmapped_r1 = build_fully_unmapped_segment(is_read1=True)
        unmapped_r2 = build_fully_unmapped_segment(is_read1=False)

        bam_path, bai_path = write_indexed_bam(
            header,
            tmp_path / "multi_contig.bam",
            [
                chr1_low_r1,
                chr1_high_r1,
                chr1_low_r2,
                chr1_high_r2,
                chr2_alone_r1,
                chr2_alone_r2,
                chr3_cell_c_r1,
                chr3_cell_d_r1,
                chr3_cell_c_r2,
                chr3_cell_d_r2,
                unmapped_r1,
                unmapped_r2,
            ],
        )

        engine = LinearDedup(bam_path, bai_path)
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            winners, stats = engine.find_best_reads(bam)

        assert_that(winners).described_as("multi-contig winners").is_equal_to(
            {"chr1_high", "chr2_alone", "chr3_cell_c", "chr3_cell_d"}
        )
        assert_that(stats.pairs_kept_by_chromosome).described_as(
            "pairs_kept_by_chromosome must not leak across contigs"
        ).is_equal_to({"chr1": 1, "chr2": 1, "chr3": 2})
        assert_that(stats.eligible_pairs_by_chromosome).described_as(
            "eligible_pairs_by_chromosome must not leak across contigs"
        ).is_equal_to({"chr1": 2, "chr2": 1, "chr3": 2})
        assert_that(stats.total_pairs).described_as("total_pairs").is_equal_to(6)
        assert_that(stats.skipped_unmapped).described_as(
            "the fully-unmapped pair's R1"
        ).is_equal_to(1)
        assert_that(stats.eligible_pairs).described_as("eligible_pairs").is_equal_to(5)


class TestLinearDedupCustomBarcodeTag:
    """barcode_tag is threaded through both fragment_key grouping and the fail-fast check."""

    def test_fragment_key_uses_the_configured_barcode_tag(self, tmp_path: Path):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        r1, _ = make_pair(
            header,
            "bc_single",
            "chr1",
            r1_pos=4000,
            r1_reverse=False,
            r2_pos=4200,
            r2_reverse=True,
            r1_tags={"BC": "CELL_BC2", "AS": 5},
        )
        engine = LinearDedup(BAM_PATH, BAI_PATH, barcode_tag="BC")

        assert_that(engine.fragment_key(r1)).described_as(
            "fragment key computed from the custom BC tag"
        ).is_equal_to(("CELL_BC2", "chr1", False, 4000))

    def test_find_best_reads_groups_by_the_configured_barcode_tag(self, tmp_path: Path):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        low_r1, low_r2 = make_pair(
            header,
            "bc_dup_low",
            "chr1",
            r1_pos=3000,
            r1_reverse=False,
            r2_pos=3200,
            r2_reverse=True,
            r1_tags={"BC": "CELL_BC1", "AS": 10},
        )
        high_r1, high_r2 = make_pair(
            header,
            "bc_dup_high",
            "chr1",
            r1_pos=3000,
            r1_reverse=False,
            r2_pos=3200,
            r2_reverse=True,
            r1_tags={"BC": "CELL_BC1", "AS": 99},
        )
        # Neither read carries a CB tag at all -- if the engine fell back to a hardcoded "CB"
        # lookup instead of self.barcode_tag, this would raise ValueError rather than group.
        bam_path, bai_path = write_indexed_bam(
            header,
            tmp_path / "bc_grouping.bam",
            [low_r1, high_r1, low_r2, high_r2],
        )

        engine = LinearDedup(bam_path, bai_path, barcode_tag="BC")
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            winners, _ = engine.find_best_reads(bam)

        assert_that(winners).described_as("BC-grouped winners").contains("bc_dup_high")
        assert_that(winners).described_as("BC-grouped winners").does_not_contain("bc_dup_low")

    def test_find_best_reads_raises_when_primary_r1_missing_the_configured_tag(
        self, tmp_path: Path
    ):
        header = build_synthetic_header(SYNTHETIC_REFERENCES)
        # This read carries CB, not BC. With barcode_tag="BC" the engine must only look at
        # self.barcode_tag, so a present-but-irrelevant CB tag must not satisfy the check.
        r1, r2 = make_pair(
            header,
            "bc_missing",
            "chr1",
            r1_pos=3500,
            r1_reverse=False,
            r2_pos=3700,
            r2_reverse=True,
            r1_tags={"CB": "CELL_X", "AS": 55},
        )
        bam_path, bai_path = write_indexed_bam(header, tmp_path / "bc_missing_tag.bam", [r1, r2])

        engine = LinearDedup(bam_path, bai_path, barcode_tag="BC")
        with pysam.AlignmentFile(bam_path, "rb", index_filename=bai_path) as bam:
            with pytest.raises(ValueError):
                engine.find_best_reads(bam)


class TestLinearDedupStatsValueObject:
    """LinearDedupStats: bare fields populated by pass 1 (find_best_reads).

    The per-chromosome breakdown dicts (and therefore the derived ``pairs_kept()``) are left at
    their empty-dict default here rather than pinned to the fixture's values -- those are
    exercised against the real fixture in TestFindBestReads instead.
    """

    def make_stats(self, **overrides) -> LinearDedupStats:
        fields = dict(
            total_pairs=FIXTURE_TOTAL_PAIRS,
            eligible_pairs=FIXTURE_ELIGIBLE_PAIRS,
            skipped_unmapped=FIXTURE_SKIPPED_UNMAPPED,
            skipped_non_primary=FIXTURE_SKIPPED_NON_PRIMARY,
            skipped_unpaired=FIXTURE_SKIPPED_UNPAIRED,
            reads_missing_as=2,
        )
        fields.update(overrides)
        return LinearDedupStats(**fields)

    @pytest.mark.parametrize(
        "field_name,new_value",
        [
            ("total_pairs", 1),
            ("eligible_pairs", 1),
            ("skipped_unmapped", 1),
            ("skipped_non_primary", 1),
            ("skipped_unpaired", 1),
            ("reads_missing_as", 1),
            ("eligible_pairs_by_chromosome", {"chr1": 1}),
            ("pairs_kept_by_chromosome", {"chr1": 1}),
        ],
    )
    def test_stats_are_frozen(self, field_name, new_value):
        stats = self.make_stats()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(stats, field_name, new_value)

    def test_eligible_pairs_by_chromosome_defaults_to_empty_dict(self):
        stats = self.make_stats()
        assert_that(stats.eligible_pairs_by_chromosome).described_as(
            "eligible_pairs_by_chromosome default"
        ).is_equal_to({})

    def test_pairs_kept_by_chromosome_defaults_to_empty_dict(self):
        stats = self.make_stats()
        assert_that(stats.pairs_kept_by_chromosome).described_as(
            "pairs_kept_by_chromosome default"
        ).is_equal_to({})


# ---------------------------------------------------------------------------------------------
# LinearDedup pass 2 and full orchestration.
#
# Everything below exercises write_deduplicated_reads and linear_dedup_reads. Every test here
# reuses find_best_reads' real winners set over the main fixture rather than a hand-written one,
# so the two passes are proven to compose correctly end to end, exactly as plans/linear-dedup.md's
# two-pass design requires.
# ---------------------------------------------------------------------------------------------

# The nine fragment-key winners find_best_reads resolves from tests/data/linear_dedup_fixture.bam
# (see TestFindBestReads' per-scenario assertions above), pinned here once so every test in
# TestWriteDeduplicatedReads can assert against a known, named set rather than re-deriving it.
WINNER_QNAMES = frozenset(
    {
        "dup_high_as",
        "diffcell_A",
        "diffcell_B",
        "diffchrom_1",
        "diffchrom_2",
        "filler_between_revdup",
        "revdup_base",
        "scored_winner",
        "noas_alone",
    }
)

# The losing QNAME from each duplicate-group scenario in the fixture -- each has a primary R1/R2
# pair present in the file, but must be entirely absent from write_deduplicated_reads' output.
LOSER_QNAMES = frozenset({"dup_low_as", "revdup_shift", "noas_loser"})

# Scenario (i): a QNAME with an unmapped R1 or an unmapped mate never reaches the winners set at
# all, so both of its mates must be entirely absent from the written output.
UNMAPPED_QNAMES = frozenset({"unmapped_r1_mapped_mate", "mapped_r1_unmapped_mate"})


class TestWriteDeduplicatedReads:
    """write_deduplicated_reads(input_bam, output_bam, winners): pass 2 of the two-pass design.

    Uses find_best_reads' real winners set over the main fixture as input, composing the two
    pass-1/pass-2 methods together rather than exercising write_deduplicated_reads against a
    fabricated winners set.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> LinearDedup:
        return LinearDedup(BAM_PATH, BAI_PATH)

    @pytest.fixture(scope="class")
    def winners(self, engine) -> set:
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as bam:
            winners, _ = engine.find_best_reads(bam)
        return winners

    @pytest.fixture(scope="class")
    def written(self, engine, winners, tmp_path_factory) -> tuple:
        """Run pass 2 once for the whole class and return (written_count, written_records)."""
        output_path = tmp_path_factory.mktemp("write-deduplicated-reads") / "written.bam"
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as input_bam:
            with pysam.AlignmentFile(
                str(output_path), "wb", header=input_bam.header
            ) as output_bam:
                written_count = engine.write_deduplicated_reads(input_bam, output_bam, winners)
        return written_count, load_records(str(output_path), None)

    @pytest.fixture(scope="class")
    def written_count(self, written) -> int:
        return written[0]

    @pytest.fixture(scope="class")
    def written_records(self, written) -> list:
        return written[1]

    @pytest.fixture(scope="class")
    def written_lookup(self, written_records) -> dict:
        return lookup_by_qname_mate_category(written_records)

    @pytest.fixture(scope="class")
    def input_lookup(self) -> dict:
        return lookup_by_qname_mate_category(load_records(BAM_PATH, BAI_PATH))

    def test_winners_set_matches_designed_layout(self, winners):
        # Guards every test below against silent fixture/engine drift: if find_best_reads ever
        # resolved a different winner set, the assertions that follow would be exercising the
        # wrong QNAMEs without warning.
        assert_that(winners).described_as(
            "find_best_reads winners over the main fixture"
        ).is_equal_to(set(WINNER_QNAMES))

    @pytest.mark.parametrize("qname", sorted(WINNER_QNAMES), ids=sorted(WINNER_QNAMES))
    def test_both_mates_of_every_winner_are_written(self, written_lookup, qname):
        assert_that(written_lookup).described_as(f"{qname} mate 1 primary written").contains_key(
            (qname, 1, "primary")
        )
        assert_that(written_lookup).described_as(f"{qname} mate 2 primary written").contains_key(
            (qname, 2, "primary")
        )

    @pytest.mark.parametrize("qname", sorted(LOSER_QNAMES), ids=sorted(LOSER_QNAMES))
    def test_both_mates_of_every_loser_are_absent(self, written_records, qname):
        written_qnames = {read.query_name for read in written_records}
        assert_that(written_qnames).described_as(
            f"{qname} (a duplicate-group loser) must be entirely absent from the output"
        ).does_not_contain(qname)

    @pytest.mark.parametrize("qname", sorted(UNMAPPED_QNAMES), ids=sorted(UNMAPPED_QNAMES))
    def test_unmapped_pairs_are_absent(self, written_records, qname):
        written_qnames = {read.query_name for read in written_records}
        assert_that(written_qnames).described_as(
            f"{qname} (unmapped R1 or unmapped mate) must be entirely absent from the output"
        ).does_not_contain(qname)

    def test_supplementary_alignment_sharing_winner_qname_is_excluded(self, written_lookup):
        # dup_high_as's own supplementary R1 alignment (scenario h) shares its QNAME with the
        # winning primary -- it must never reach the output, even though the primary does.
        assert_that(written_lookup).described_as(
            "dup_high_as's own primary must still be present"
        ).contains_key(("dup_high_as", 1, "primary"))
        assert_that(written_lookup).described_as(
            "dup_high_as's supplementary alignment must be excluded"
        ).does_not_contain_key(("dup_high_as", 1, "supplementary"))

    def test_secondary_alignment_sharing_winner_qname_is_excluded(self, written_lookup):
        # diffchrom_1's own secondary R1 alignment (scenario h) shares its QNAME with the winning
        # primary -- it must never reach the output, even though the primary does.
        assert_that(written_lookup).described_as(
            "diffchrom_1's own primary must still be present"
        ).contains_key(("diffchrom_1", 1, "primary"))
        assert_that(written_lookup).described_as(
            "diffchrom_1's secondary alignment must be excluded"
        ).does_not_contain_key(("diffchrom_1", 1, "secondary"))

    @pytest.mark.parametrize("mate", [1, 2], ids=["mate1", "mate2"])
    def test_no_tag_or_flag_is_rewritten_on_a_written_read(
        self, input_lookup, written_lookup, mate
    ):
        key = ("dup_high_as", mate, "primary")
        input_read = input_lookup[key]
        output_read = written_lookup[key]

        assert_that(output_read.flag).described_as(f"{key} flag byte").is_equal_to(input_read.flag)
        assert_that(sorted(output_read.get_tags())).described_as(f"{key} tags").is_equal_to(
            sorted(input_read.get_tags())
        )
        assert_that(output_read.to_string()).described_as(f"{key} full SAM record").is_equal_to(
            input_read.to_string()
        )

    def test_return_value_equals_actual_records_written(self, written_count, written_records):
        assert_that(written_count).described_as(
            "write_deduplicated_reads return value vs. actual records in the output"
        ).is_equal_to(len(written_records))

    def test_return_value_equals_two_per_surviving_pair(self, written_count):
        assert_that(written_count).described_as(
            "2 written records per winning QNAME (both mates, no losers/non-primaries)"
        ).is_equal_to(len(WINNER_QNAMES) * 2)


class TestLinearDedupReadsOrchestration:
    """linear_dedup_reads(output_dir, prefix): composes both passes end to end.

    This class covers the indexed output BAM; the plain-text report and MultiQC payloads it also
    writes are covered separately by TestLinearDedupReadsReporting below.
    """

    def test_produces_indexed_output_bam(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="orchestrated")

        bam_path = tmp_path / "orchestrated.linear_dedup.bam"
        bai_path = tmp_path / "orchestrated.linear_dedup.bam.bai"
        assert_that(bam_path.exists()).described_as("linear_dedup.bam exists").is_true()
        assert_that(bai_path.exists()).described_as("linear_dedup.bam.bai exists").is_true()

    def test_output_bam_is_coordinate_sorted(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="sorted_check")

        bam_path = tmp_path / "sorted_check.linear_dedup.bam"
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            assert_that(bam.header.get("HD", {}).get("SO")).described_as(
                "output BAM header sort order (SO)"
            ).is_equal_to("coordinate")

            last_key = None
            for read in bam:
                key = (read.reference_id, read.reference_start)
                if last_key is not None:
                    assert_that(key >= last_key).described_as(
                        f"(reference_id, reference_start) at {read.query_name} vs. previous record"
                    ).is_true()
                last_key = key

    def test_output_bam_contains_only_primary_records(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="primary_only")

        bam_path = tmp_path / "primary_only.linear_dedup.bam"
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam)

        assert_that(records).described_as("output BAM record count").is_not_empty()
        for read in records:
            assert_that(read.is_secondary).described_as(
                f"{read.query_name} is_secondary"
            ).is_false()
            assert_that(read.is_supplementary).described_as(
                f"{read.query_name} is_supplementary"
            ).is_false()

    def test_output_read_set_matches_direct_two_pass_composition(self, tmp_path: Path):
        orchestrated_dir = tmp_path / "orchestrated"
        orchestrated_dir.mkdir()
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(orchestrated_dir), prefix="composed")

        with pysam.AlignmentFile(str(orchestrated_dir / "composed.linear_dedup.bam"), "rb") as bam:
            orchestrated_records = sorted(
                (read.query_name, read.flag, read.reference_start) for read in bam
            )

        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as input_bam:
            winners, _ = engine.find_best_reads(input_bam)

        direct_path = tmp_path / "direct.bam"
        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as input_bam:
            with pysam.AlignmentFile(
                str(direct_path), "wb", header=input_bam.header
            ) as output_bam:
                engine.write_deduplicated_reads(input_bam, output_bam, winners)
        with pysam.AlignmentFile(str(direct_path), "rb") as bam:
            direct_records = sorted(
                (read.query_name, read.flag, read.reference_start) for read in bam
            )

        assert_that(orchestrated_records).described_as(
            "linear_dedup_reads' output vs. find_best_reads + write_deduplicated_reads called directly"
        ).is_equal_to(direct_records)

    def test_returns_stats_with_pass_one_fields_populated(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        stats = engine.linear_dedup_reads(str(tmp_path), prefix="stats_check")

        assert_that(isinstance(stats, LinearDedupStats)).described_as(
            "linear_dedup_reads return type"
        ).is_true()
        assert_that(stats.total_pairs).described_as("total_pairs").is_equal_to(FIXTURE_TOTAL_PAIRS)
        assert_that(stats.eligible_pairs).described_as("eligible_pairs").is_equal_to(
            FIXTURE_ELIGIBLE_PAIRS
        )
        assert_that(stats.skipped_non_primary).described_as("skipped_non_primary").is_equal_to(
            FIXTURE_SKIPPED_NON_PRIMARY
        )
        assert_that(stats.skipped_unmapped).described_as("skipped_unmapped").is_equal_to(
            FIXTURE_SKIPPED_UNMAPPED
        )
        assert_that(stats.skipped_unpaired).described_as("skipped_unpaired").is_equal_to(
            FIXTURE_SKIPPED_UNPAIRED
        )
        assert_that(stats.eligible_pairs_by_chromosome).described_as(
            "eligible_pairs_by_chromosome"
        ).is_equal_to(FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME)
        assert_that(stats.pairs_kept_by_chromosome).described_as(
            "pairs_kept_by_chromosome"
        ).is_equal_to(FIXTURE_PAIRS_KEPT_BY_CHROMOSOME)

    def test_prefix_defaults_to_bam_filename_prefix(self, tmp_path: Path):
        # Mirrors UmiExtractor.extract_umis' `prefix or get_prefix(self.fastq.filename)`
        # convention (carmack/umi/umi_extractor.py): a None prefix falls back to get_prefix
        # applied to the input BAM's own filename.
        bam_copy = tmp_path / "SK462.bam"
        bai_copy = tmp_path / "SK462.bam.bai"
        bam_copy.write_bytes(Path(BAM_PATH).read_bytes())
        bai_copy.write_bytes(Path(BAI_PATH).read_bytes())

        engine = LinearDedup(str(bam_copy), str(bai_copy))
        engine.linear_dedup_reads(str(tmp_path))

        expected_prefix = get_prefix(str(bam_copy))
        assert_that(expected_prefix).described_as(
            "expected prefix derived from the BAM filename"
        ).is_equal_to("SK462")
        assert_that((tmp_path / f"{expected_prefix}.linear_dedup.bam").exists()).described_as(
            "output BAM uses the prefix derived from the input BAM filename"
        ).is_true()
        assert_that((tmp_path / f"{expected_prefix}.linear_dedup.bam.bai").exists()).described_as(
            "output BAM index uses the same derived prefix"
        ).is_true()


class TestLinearDedupOutputOrderMatchesFilteredInput:
    """linear_dedup_reads no longer runs an external sort on its output (see the module's
    linear_dedup_reads docstring) -- this proves the assumption that makes that safe: the
    output is exactly the filtered input stream, in the same relative order, not merely
    non-decreasing by (reference_id, reference_start)."""

    def test_output_order_equals_filtered_input_order(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="order_check")

        with pysam.AlignmentFile(str(tmp_path / "order_check.linear_dedup.bam"), "rb") as bam:
            output_records = list(bam)

        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as input_bam:
            winners, _ = LinearDedup(BAM_PATH, BAI_PATH).find_best_reads(input_bam)

        with pysam.AlignmentFile(BAM_PATH, "rb", index_filename=BAI_PATH) as input_bam:
            expected_order = [
                read
                for read in input_bam.fetch(until_eof=True)
                if not read.is_secondary
                and not read.is_supplementary
                and read.query_name in winners
            ]

        assert_that(len(output_records)).described_as(
            "output record count vs. the filtered input stream"
        ).is_equal_to(len(expected_order))
        for position, (actual, expected) in enumerate(zip(output_records, expected_order)):
            assert_that(actual.to_string()).described_as(
                f"record at output position {position} vs. its filtered-input counterpart"
            ).is_equal_to(expected.to_string())


# ---------------------------------------------------------------------------------------------
# LinearDedupStats reporting: plain-text report and MultiQC payloads.
#
# Everything below exercises get_report, to_mqc_general_stats, to_mqc_breakdown,
# to_mqc_chromosome_breakdown, and linear_dedup_reads' reporting-writing behaviour, none of
# which exist yet on LinearDedupStats / LinearDedup. Percentages are computed against two
# different denominators, matching what each number is a fraction of:
#   - eligible / skipped_unmapped / skipped_non_primary / skipped_unpaired: percentage of
#     total_pairs (every one of these is a disjoint slice of the reads pass 1 scanned).
#   - pairs kept / pairs removed (duplicates) / reads missing AS: percentage of eligible_pairs
#     (every one of these is a fact about the reads that actually entered grouping).
# "Pairs kept" has no dedicated top-level field -- it is the sum of pairs_kept_by_chromosome's
# values; "pairs removed" (the duplication count) is eligible_pairs minus that sum.
# ---------------------------------------------------------------------------------------------

# Total winning fragment-key groups and duplicate pairs removed for the main fixture, derived
# from FIXTURE_PAIRS_KEPT_BY_CHROMOSOME / FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME above.
FIXTURE_PAIRS_KEPT = sum(FIXTURE_PAIRS_KEPT_BY_CHROMOSOME.values())
FIXTURE_PAIRS_REMOVED = FIXTURE_ELIGIBLE_PAIRS - FIXTURE_PAIRS_KEPT

# Of the 12 eligible primary R1s in the fixture, noas_loser and noas_alone carry no AS tag --
# see TestLinearDedupStatsValueObject.make_stats, which pins the same value.
FIXTURE_READS_MISSING_AS = 2

# The MultiQC bundle keys a linear-dedup payload would have nested its charts under, mirroring
# UMI_BUNDLE_KEYS in tests/test_umi_extractor.py.
LINEAR_DEDUP_BUNDLE_KEYS = ("general_stats", "breakdown", "chromosome_breakdown")


def build_fixture_stats(**overrides) -> LinearDedupStats:
    """Return a LinearDedupStats matching the main fixture's reconciling totals, overridable."""
    fields = dict(
        total_pairs=FIXTURE_TOTAL_PAIRS,
        eligible_pairs=FIXTURE_ELIGIBLE_PAIRS,
        skipped_unmapped=FIXTURE_SKIPPED_UNMAPPED,
        skipped_non_primary=FIXTURE_SKIPPED_NON_PRIMARY,
        skipped_unpaired=FIXTURE_SKIPPED_UNPAIRED,
        reads_missing_as=FIXTURE_READS_MISSING_AS,
        eligible_pairs_by_chromosome=dict(FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME),
        pairs_kept_by_chromosome=dict(FIXTURE_PAIRS_KEPT_BY_CHROMOSOME),
    )
    fields.update(overrides)
    return LinearDedupStats(**fields)


def build_all_zero_stats(**overrides) -> LinearDedupStats:
    """Return a LinearDedupStats with every count at zero, overridable -- for zero-guard checks."""
    fields = dict(
        total_pairs=0,
        eligible_pairs=0,
        skipped_unmapped=0,
        skipped_non_primary=0,
        skipped_unpaired=0,
        reads_missing_as=0,
    )
    fields.update(overrides)
    return LinearDedupStats(**fields)


class TestLinearDedupStatsGetReport:
    """get_report(): plain-text report in the same # comment-header + Key: count (pct%) style
    as UmiExtractionStats.get_report()."""

    def test_report_includes_the_carmack_and_section_headers(self):
        report = build_fixture_stats().get_report()

        assert_that(report).contains("# Carmack version:")
        assert_that(report).contains("# Report generated at:")
        assert_that(report).contains("# Linear Dedup Stats")

    def test_report_reconciles_against_hand_computed_fixture_counts_and_percentages(self):
        # total_pairs=16, eligible_pairs=12, skipped_unmapped=2, skipped_non_primary=2,
        # skipped_unpaired=0 -> percentages of total_pairs: 75.00 / 12.50 / 12.50 / 0.00.
        # pairs_kept=9, pairs_removed=3, reads_missing_as=2 -> percentages of eligible_pairs:
        # 75.00 / 25.00 / 16.67.
        report = build_fixture_stats().get_report()

        assert_that(report).contains(f"Total pairs: {FIXTURE_TOTAL_PAIRS}")
        assert_that(report).contains(f"Eligible: {FIXTURE_ELIGIBLE_PAIRS} (75.00%)")
        assert_that(report).contains(f"Skipped (unmapped): {FIXTURE_SKIPPED_UNMAPPED} (12.50%)")
        assert_that(report).contains(
            f"Skipped (non_primary): {FIXTURE_SKIPPED_NON_PRIMARY} (12.50%)"
        )
        assert_that(report).contains(f"Skipped (unpaired): {FIXTURE_SKIPPED_UNPAIRED} (0.00%)")
        assert_that(report).contains(f"Pairs kept: {FIXTURE_PAIRS_KEPT} (75.00%)")
        assert_that(report).contains(
            f"Pairs removed (duplicates): {FIXTURE_PAIRS_REMOVED} (25.00%)"
        )
        assert_that(report).contains(f"Missing AS: {FIXTURE_READS_MISSING_AS} (16.67%)")

    def test_report_handles_all_zero_stats_without_error(self):
        report = build_all_zero_stats().get_report()

        assert_that(report).contains("Total pairs: 0")
        assert_that(report).contains("Eligible: 0 (0.00%)")
        assert_that(report).contains("Skipped (unmapped): 0 (0.00%)")
        assert_that(report).contains("Skipped (non_primary): 0 (0.00%)")
        assert_that(report).contains("Skipped (unpaired): 0 (0.00%)")
        assert_that(report).contains("Pairs kept: 0 (0.00%)")
        assert_that(report).contains("Pairs removed (duplicates): 0 (0.00%)")
        assert_that(report).contains("Missing AS: 0 (0.00%)")

    def test_get_report_returns_a_string(self):
        assert_that(build_fixture_stats().get_report()).is_instance_of(str)


class TestLinearDedupStatsToMqcGeneralStats:
    """to_mqc_general_stats(prefix): generalstats payload with pct duplication and pct missing-AS."""

    SAMPLE_PREFIX = "SK462"

    def test_has_generalstats_plot_type_and_id(self):
        payload = build_fixture_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("generalstats")
        assert_that(payload["id"]).is_equal_to("carmack_linear_dedup_general_stats")

    def test_attributes_its_columns_with_a_namespace_not_parent_id(self):
        # namespace is what attributes a generalstats payload to Carmack -- parent_id is inert
        # for that plot type (carmack/mqc_report.py), matching UmiExtractionStats' convention.
        payload = build_fixture_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)

        assert_that(payload).contains_entry({"namespace": CARMACK_PARENT_NAME})
        assert_that(payload).does_not_contain_key("parent_id")
        assert_that(payload).does_not_contain_key("parent_name")

    def test_computes_pct_duplication_as_removed_over_eligible(self):
        # duplication rate is the fraction of eligible pairs that were removed as duplicates:
        # (eligible_pairs - pairs_kept) / eligible_pairs = 3 / 12 = 25.0.
        payload = build_fixture_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_duplication"]).is_equal_to(25.0)

    def test_computes_pct_missing_as_over_eligible(self):
        # 2 / 12 = 16.666...
        payload = build_fixture_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_missing_as"]).is_close_to(100 * 2 / 12, 1e-9)

    def test_on_zero_eligible_pairs_returns_zero_percentages(self):
        payload = build_all_zero_stats().to_mqc_general_stats(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(data["pct_duplication"]).is_equal_to(0.0)
        assert_that(data["pct_missing_as"]).is_equal_to(0.0)

    def test_data_is_keyed_by_the_given_prefix(self):
        payload = build_fixture_stats().to_mqc_general_stats("SK999")

        assert_that(payload["data"]).contains_key("SK999")
        assert_that(payload["data"]).does_not_contain_key(self.SAMPLE_PREFIX)


class TestLinearDedupStatsToMqcBreakdown:
    """to_mqc_breakdown(prefix): bargraph payload with raw kept/removed/skipped counts."""

    SAMPLE_PREFIX = "SK462"

    def test_has_bargraph_plot_type_id_and_parent(self):
        payload = build_fixture_stats().to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["plot_type"]).is_equal_to("bargraph")
        assert_that(payload["id"]).is_equal_to("carmack_linear_dedup_breakdown")
        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    def test_data_matches_raw_kept_removed_skipped_counts(self):
        payload = build_fixture_stats().to_mqc_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(
            {
                "kept": FIXTURE_PAIRS_KEPT,
                "removed": FIXTURE_PAIRS_REMOVED,
                "skipped_unmapped": FIXTURE_SKIPPED_UNMAPPED,
                "skipped_non_primary": FIXTURE_SKIPPED_NON_PRIMARY,
                "skipped_unpaired": FIXTURE_SKIPPED_UNPAIRED,
            }
        )

    def test_breakdown_counts_reconcile_with_total_pairs(self):
        # kept + removed together equal eligible_pairs; adding the three skip categories back on
        # top must reconcile to total_pairs, exactly like the stats invariant itself.
        payload = build_fixture_stats().to_mqc_breakdown(self.SAMPLE_PREFIX)
        data = payload["data"][self.SAMPLE_PREFIX]

        assert_that(sum(data.values())).described_as(
            "kept + removed + every skipped category must equal total_pairs"
        ).is_equal_to(FIXTURE_TOTAL_PAIRS)


class TestLinearDedupStatsToMqcChromosomeBreakdown:
    """to_mqc_chromosome_breakdown(prefix): bargraph of duplicates removed per chromosome."""

    SAMPLE_PREFIX = "SK462"

    def test_has_bargraph_plot_type_id_and_parent(self):
        payload = build_fixture_stats().to_mqc_chromosome_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload).is_not_none()
        assert_that(payload["plot_type"]).is_equal_to("bargraph")
        assert_that(payload["id"]).is_equal_to("carmack_linear_dedup_chromosome_breakdown")
        assert_that(payload["parent_id"]).is_equal_to(CARMACK_PARENT_ID)
        assert_that(payload["parent_name"]).is_equal_to(CARMACK_PARENT_NAME)

    def test_data_holds_duplicates_removed_per_chromosome(self):
        # chr1: 11 eligible - 8 kept = 3 removed. chr2: 1 eligible - 1 kept = 0 removed.
        payload = build_fixture_stats().to_mqc_chromosome_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to(
            {
                "chr1": FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME["chr1"]
                - FIXTURE_PAIRS_KEPT_BY_CHROMOSOME["chr1"],
                "chr2": FIXTURE_ELIGIBLE_PAIRS_BY_CHROMOSOME["chr2"]
                - FIXTURE_PAIRS_KEPT_BY_CHROMOSOME["chr2"],
            }
        )

    def test_returns_none_when_both_chromosome_dicts_are_empty(self):
        # The established rule: no data to report -> None, so write_mqc_payloads skips it
        # rather than emitting an empty chart.
        stats = build_all_zero_stats(eligible_pairs_by_chromosome={}, pairs_kept_by_chromosome={})

        assert_that(stats.to_mqc_chromosome_breakdown(self.SAMPLE_PREFIX)).is_none()

    def test_a_chromosome_absent_from_pairs_kept_defaults_to_fully_removed(self):
        # find_best_reads itself can never produce a chromosome present in
        # eligible_pairs_by_chromosome but absent from pairs_kept_by_chromosome (any chromosome
        # with an eligible read yields at least one winner) -- this only exercises the payload
        # builder's own defaulting on a directly constructed LinearDedupStats.
        stats = build_all_zero_stats(
            total_pairs=5,
            eligible_pairs=5,
            eligible_pairs_by_chromosome={"chrX": 5},
            pairs_kept_by_chromosome={},
        )

        payload = stats.to_mqc_chromosome_breakdown(self.SAMPLE_PREFIX)

        assert_that(payload["data"][self.SAMPLE_PREFIX]).is_equal_to({"chrX": 5})


class TestLinearDedupReadsReporting:
    """linear_dedup_reads(output_dir, prefix) also writes the plain-text report and MultiQC
    payloads, via the shared write_mqc_payloads(output_dir, prefix, payloads) writer."""

    def test_writes_a_stats_txt_file_matching_get_report(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        stats = engine.linear_dedup_reads(str(tmp_path), prefix="report_check")

        stats_path = tmp_path / "report_check.linear_dedup_stats.txt"
        assert_that(stats_path.exists()).described_as("linear_dedup_stats.txt exists").is_true()
        # The run-detail lines (carmack version, generated-at timestamp) are volatile between
        # two separate get_report() calls, so both sides are stripped of them before comparing,
        # mirroring tests/utils.py's strip_report_run_details / golden-report convention.
        assert_that(strip_report_run_details(stats_path.read_text())).described_as(
            "written report vs. the returned stats' own get_report()"
        ).is_equal_to(strip_report_run_details(stats.get_report()))

    def test_writes_the_expected_mqc_payload_files(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="mqc_check")

        assert_mqc_payload_file(
            tmp_path / "mqc_check.linear_dedup_general_stats_mqc.json",
            "carmack_linear_dedup_general_stats",
        )
        assert_mqc_payload_file(
            tmp_path / "mqc_check.linear_dedup_breakdown_mqc.json",
            "carmack_linear_dedup_breakdown",
        )
        assert_mqc_payload_file(
            tmp_path / "mqc_check.linear_dedup_chromosome_breakdown_mqc.json",
            "carmack_linear_dedup_chromosome_breakdown",
        )

    def test_no_mqc_file_bundles_more_than_one_payload(self, tmp_path: Path):
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="bundle_check")

        assert_no_mqc_file_bundles_payloads(tmp_path, LINEAR_DEDUP_BUNDLE_KEYS)

    def test_every_output_file_for_the_prefix_is_accounted_for(self, tmp_path: Path):
        # The main fixture always yields a non-empty per-chromosome breakdown, so every one of
        # the three payloads is written -- none is skipped as None here.
        engine = LinearDedup(BAM_PATH, BAI_PATH)
        engine.linear_dedup_reads(str(tmp_path), prefix="complete_check")

        assert_that(sorted(p.name for p in tmp_path.glob("complete_check.*"))).is_equal_to(
            [
                "complete_check.linear_dedup.bam",
                "complete_check.linear_dedup.bam.bai",
                "complete_check.linear_dedup_breakdown_mqc.json",
                "complete_check.linear_dedup_chromosome_breakdown_mqc.json",
                "complete_check.linear_dedup_general_stats_mqc.json",
                "complete_check.linear_dedup_stats.txt",
            ]
        )
