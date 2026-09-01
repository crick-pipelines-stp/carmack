import csv
import os

import pysam
import pytest
from assertpy import assert_that

from carmack.tag_dedup.tag_dedup import TagDedup
from carmack.utils import get_prefix

BAM_PATHS = [
    "tests/data/hydrop_scatac_1_S1_R1.sorted.bam",
    "tests/data/hydrop_scatac_1_S1_R1_dup.sorted.bam",
]
BAI_PATHS = [
    "tests/data/hydrop_scatac_1_S1_R1.sorted.bam.bai",
    "tests/data/hydrop_scatac_1_S1_R1_dup.sorted.bam.bai",
]
CSV_PATHS = ["tests/data/bc_valid.csv", "tests/data/bc_valid_dup.csv"]
CUSTOM_INDEX = 1  # Easier to run tests on the custom files (much smaller)

# Hand-crafted "dup" fixture used for the UMI-aware tag/dedup tests.
DUP_BAM = "tests/data/hydrop_scatac_1_S1_R1_dup.sorted.bam"
DUP_BAI = "tests/data/hydrop_scatac_1_S1_R1_dup.sorted.bam.bai"
DUP_CSV = "tests/data/bc_valid_dup.csv"
DUP_GOLDEN_DEDUP = "tests/data/hydrop_scatac_1_S1_R1_dup.dedup.tagged.bam"

# Read names (BAM query names) present in the dup fixture.
READ01 = "READ01:NONDUP:GRP1:SAMEBARCODE"
READ02 = "READ02:XXXDUP:GRP1:SAMEBARCODE"
READ05 = "READ05:NONDUP:GRP2:FULLYUNIQUE"
READ06 = "READ06:NONDUP:GRP2:FULLYUNIQUE"
READ07 = "READ07:NONDUP:GRP3:DIFFBARCODE"
READ08 = "READ08:XXXDUP:GRP3:DIFFBARCODE"

# Barcodes (as loaded from bc_valid_dup.csv) keyed by read name.
BC_SHARED = "AAGTGTAGAATGAACCTTACCACTGGTGGT"  # READ01, READ02 and READ08
BC_READ05 = "CGTTATACGTTTGCGAATATCTATTAGGCT"
BC_READ06 = "TTCATGTCCTAGCTTGAGAGAGGTTAGAGC"
BC_READ07 = "TCTACAGTGCGACAAGTGGATAGTTCTTGA"


class TestTagDedup:
    def _get_files(self, out_dir, bam_path):
        """
        Get paths to output files.
        """
        prefix = get_prefix(bam_path)
        file_paths = {
            "bam_tagged_file": os.path.join(out_dir, prefix + ".tagged.bam"),
            "bai_tagged_file": os.path.join(out_dir, prefix + ".tagged.bam.bai"),
            "bam_dedup_file": os.path.join(out_dir, prefix + ".dedup.tagged.bam"),
            "bai_dedup_file": os.path.join(out_dir, prefix + ".dedup.tagged.bam.bai"),
            "multiqc_log": os.path.join(out_dir, prefix + ".dedup.stats_mqc.log"),
        }
        return file_paths

    def _check_output(self, file_dict, dedup=True):
        """
        Assert statements to check for the presence of output files.
        """
        assert os.path.exists(file_dict["bam_tagged_file"])
        assert os.path.exists(file_dict["bai_tagged_file"])
        if dedup:
            assert os.path.exists(file_dict["bam_dedup_file"])
            assert os.path.exists(file_dict["bai_dedup_file"])
            assert os.path.exists(file_dict["multiqc_log"])

    def test_tag_only(self, tmpdir):
        """
        Test if barcode and tags are added to the reads in the BAM file.
        """
        # Init
        bam_path = BAM_PATHS[CUSTOM_INDEX]
        bai_path = BAI_PATHS[CUSTOM_INDEX]
        csv_path = CSV_PATHS[CUSTOM_INDEX]
        files = self._get_files(tmpdir, bam_path)

        # Work
        tag_dedup = TagDedup(bam_path, bai_path, csv_path)
        tag_dedup.tag_dedup_reads(dedup=False, output_dir=tmpdir)

        # Tests
        ## Presence of output files
        self._check_output(files, dedup=False)

        ## Check the BAM file
        with pysam.AlignmentFile(files["bam_tagged_file"], "rb") as bam:
            for read in bam:
                assert read.has_tag("CB")
                assert read.has_tag("DU")

                # CR mirrors CB until a distinct raw cell barcode is plumbed through.
                assert read.has_tag("CR")
                assert read.get_tag("CR") == read.get_tag("CB")

                # The legacy BC SAM tag must no longer be written.
                assert not read.has_tag("BC")

    @pytest.mark.parametrize("bam_path, bai_path, csv_path", zip(BAM_PATHS, BAI_PATHS, CSV_PATHS))
    def test_tag_dedup(self, tmpdir, bam_path, bai_path, csv_path):
        """
        Tests for the tagging and deduplication.
        """
        # Init
        files = self._get_files(tmpdir, bam_path)
        unique_counter = {"bam_tagged": 0, "bam_dedup": 0, "mqc_log": 0}

        ## Load barcodes
        with open(csv_path, "r") as valid_barcodes:
            csv_reader = csv.reader(valid_barcodes)
            bc_dict = {line[0].split(" ", 1)[0]: line[1] for line in csv_reader}

        # Work
        tag_dedup = TagDedup(bam_path, bai_path, csv_path)
        tag_dedup.tag_dedup_reads(dedup=True, output_dir=tmpdir)

        # Tests
        ## Presence of output files
        self._check_output(files)

        ## Check reads in tagged BAM
        with pysam.AlignmentFile(files["bam_tagged_file"], "rb") as bam_tagged:
            for read in bam_tagged:
                ### Check for presence of tags
                assert read.has_tag("CB")
                assert read.has_tag("DU")

                ### Check if the barcodes are correctly assigned
                read_name = read.query_name
                barcode = read.get_tag("CB")
                assert bc_dict[read_name] == barcode

                if read.get_tag("DU") == 0:
                    unique_counter["bam_tagged"] += 1

        ## Check reads in deduplicated BAM
        with pysam.AlignmentFile(files["bam_dedup_file"], "rb") as bam_dedup:
            for read in bam_dedup:
                ### Check for presence of tags
                assert read.has_tag("CB")
                assert read.has_tag("DU")

                ### Check if all DU==0
                assert read.get_tag("DU") == 0

                ### Get duplicate count
                unique_counter["bam_dedup"] += 1

        ## Check if unique read counts are same across files
        ### Read unique count from multiqc log
        with open(files["multiqc_log"], "r") as multiqc_log:
            csv_reader = csv.reader(multiqc_log)
            unique_counter["mqc_log"] = int(list(csv_reader)[1][0])

        ### Check
        assert len(set(unique_counter.values())) == 1

        ## Custom-file specific tests
        if bam_path == BAM_PATHS[CUSTOM_INDEX]:
            ### Check if all unique read counts are the same and equal to 10
            assert all([count == 10 for count in unique_counter.values()])

    # ------------------------------------------------------------------ #
    # UMI-aware tag/dedup (SI-5)
    # ------------------------------------------------------------------ #
    def _write_umi_map(self, path, rows):
        """
        Write a TAB-separated UMI map (read_id, barcode, UR, UB) with no header.
        """
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            for read_id, barcode, ur, ub in rows:
                writer.writerow([read_id, barcode, ur, ub])

    def _dedup_summary(self, bam_path):
        """
        Summarise a dedup BAM into comparable (name, start, tlen, CB, DU) tuples.
        """
        summary = []
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                summary.append(
                    (
                        read.query_name,
                        read.reference_start,
                        read.template_length,
                        read.get_tag("CB"),
                        int(read.get_tag("DU")),
                    )
                )
        return summary

    def test_tag_dedup_no_umi_map_matches_golden(self, tmpdir):
        """
        Without a UMI map the tag/dedup output must match today's golden behaviour.
        """
        files = self._get_files(tmpdir, DUP_BAM)

        TagDedup(DUP_BAM, DUP_BAI, DUP_CSV).tag_dedup_reads(dedup=True, output_dir=tmpdir)

        # Deduplicated reads (and their BC/DU tags) match the committed golden.
        produced = self._dedup_summary(files["bam_dedup_file"])
        golden = self._dedup_summary(DUP_GOLDEN_DEDUP)
        assert_that(produced).is_equal_to(golden)

        # No UMI tags are set when no UMI map is present.
        with pysam.AlignmentFile(files["bam_tagged_file"], "rb") as bam:
            for read in bam:
                assert_that(read.has_tag("UR")).is_false()
                assert_that(read.has_tag("UB")).is_false()

        # MultiQC log is unchanged: 10 unique, 2 duplicate reads.
        with open(files["multiqc_log"], "r") as multiqc_log:
            rows = list(csv.reader(multiqc_log))
        assert_that(rows).is_equal_to([["unique_reads", "duplicate_reads"], ["10", "2"]])

    def test_tag_dedup_different_corrected_umi_keeps_reads(self, tmpdir):
        """
        Reads sharing (pos, barcode) but with different corrected UMIs stay distinct.
        """
        umi_map = os.path.join(tmpdir, "umi_map.tsv")
        self._write_umi_map(
            umi_map,
            [
                (READ01, BC_SHARED, "AAAAAAAAAA", "AAAAAAAAAA"),
                (READ02, BC_SHARED, "CCCCCCCCCC", "CCCCCCCCCC"),
                (READ05, BC_READ05, "GGGGGGGGGG", "GGGGGGGGGG"),
                (READ06, BC_READ06, "TTTTTTTTTT", "TTTTTTTTTT"),
                (READ07, BC_READ07, "ACACACACAC", "ACACACACAC"),
                (READ08, BC_SHARED, "GTGTGTGTGT", "GTGTGTGTGT"),
            ],
        )
        files = self._get_files(tmpdir, DUP_BAM)

        TagDedup(DUP_BAM, DUP_BAI, DUP_CSV, umi_map=umi_map).tag_dedup_reads(
            dedup=True, output_dir=tmpdir
        )

        names = [row[0] for row in self._dedup_summary(files["bam_dedup_file"])]
        # READ02 differs from READ01 only by corrected UMI, so it now survives at
        # both of its mate positions and nothing collapses.
        assert_that(names.count(READ02)).is_equal_to(2)
        assert_that(names).is_length(12)

    def test_tag_dedup_same_corrected_umi_collapses_reads(self, tmpdir):
        """
        Reads with the same corrected UMI (UB) collapse even if the raw UMI (UR) differs.
        """
        umi_map = os.path.join(tmpdir, "umi_map.tsv")
        self._write_umi_map(
            umi_map,
            [
                # READ01 / READ02 share a corrected UMI (UB) but differ in raw UMI (UR).
                (READ01, BC_SHARED, "AAAAAAAAAA", "AAAAAAAAAA"),
                (READ02, BC_SHARED, "AAAAAAAATA", "AAAAAAAAAA"),
                (READ05, BC_READ05, "GGGGGGGGGG", "GGGGGGGGGG"),
                (READ06, BC_READ06, "TTTTTTTTTT", "TTTTTTTTTT"),
                (READ07, BC_READ07, "ACACACACAC", "ACACACACAC"),
                (READ08, BC_SHARED, "GTGTGTGTGT", "GTGTGTGTGT"),
            ],
        )
        files = self._get_files(tmpdir, DUP_BAM)

        TagDedup(DUP_BAM, DUP_BAI, DUP_CSV, umi_map=umi_map).tag_dedup_reads(
            dedup=True, output_dir=tmpdir
        )

        names = [row[0] for row in self._dedup_summary(files["bam_dedup_file"])]
        # Deduplication keys on the corrected UMI (UB), not the raw UMI (UR).
        assert_that(names).does_not_contain(READ02)
        assert_that(names).is_length(10)

    def test_umi_tags_set_from_map(self, tmpdir):
        """
        UR/UB tags are written onto tagged reads from the UMI map.
        """
        umi_map = os.path.join(tmpdir, "umi_map.tsv")
        self._write_umi_map(
            umi_map,
            [
                (READ01, BC_SHARED, "AAAAAAAAAA", "AAAAAAAAAG"),
                (READ02, BC_SHARED, "CCCCCCCCCC", "CCCCCCCCCT"),
                (READ05, BC_READ05, "GGGGGGGGGG", "GGGGGGGGGA"),
                (READ06, BC_READ06, "TTTTTTTTTT", "TTTTTTTTTA"),
                (READ07, BC_READ07, "ACACACACAC", "ACACACACAG"),
                (READ08, BC_SHARED, "GTGTGTGTGT", "GTGTGTGTGA"),
            ],
        )
        files = self._get_files(tmpdir, DUP_BAM)

        TagDedup(DUP_BAM, DUP_BAI, DUP_CSV, umi_map=umi_map).tag_dedup_reads(
            dedup=False, output_dir=tmpdir
        )

        expected = {
            READ01: ("AAAAAAAAAA", "AAAAAAAAAG"),
            READ05: ("GGGGGGGGGG", "GGGGGGGGGA"),
        }
        with pysam.AlignmentFile(files["bam_tagged_file"], "rb") as bam:
            for read in bam:
                assert_that(read.has_tag("UR")).is_true()
                assert_that(read.has_tag("UB")).is_true()
                if read.query_name in expected:
                    ur, ub = expected[read.query_name]
                    assert_that(read.get_tag("UR")).is_equal_to(ur)
                    assert_that(read.get_tag("UB")).is_equal_to(ub)

    def test_umi_map_missing_read_tagged_without_umi_tags(self, tmpdir):
        """
        Reads absent from the UMI map are still tagged and deduped, but without UR/UB.
        """
        umi_map = os.path.join(tmpdir, "umi_map.tsv")
        # READ05 is deliberately omitted from the map.
        self._write_umi_map(
            umi_map,
            [
                (READ01, BC_SHARED, "AAAAAAAAAA", "AAAAAAAAAA"),
                (READ02, BC_SHARED, "CCCCCCCCCC", "CCCCCCCCCC"),
                (READ06, BC_READ06, "TTTTTTTTTT", "TTTTTTTTTT"),
                (READ07, BC_READ07, "ACACACACAC", "ACACACACAC"),
                (READ08, BC_SHARED, "GTGTGTGTGT", "GTGTGTGTGT"),
            ],
        )
        files = self._get_files(tmpdir, DUP_BAM)

        TagDedup(DUP_BAM, DUP_BAI, DUP_CSV, umi_map=umi_map).tag_dedup_reads(
            dedup=True, output_dir=tmpdir
        )

        seen_read05 = False
        with pysam.AlignmentFile(files["bam_tagged_file"], "rb") as bam:
            for read in bam:
                if read.query_name == READ05:
                    seen_read05 = True
                    assert_that(read.has_tag("CB")).is_true()
                    assert_that(read.has_tag("DU")).is_true()
                    assert_that(read.has_tag("UR")).is_false()
                    assert_that(read.has_tag("UB")).is_false()
        assert_that(seen_read05).is_true()

        # READ05 (missing from the map) still survives dedup.
        names = [row[0] for row in self._dedup_summary(files["bam_dedup_file"])]
        assert_that(names).contains(READ05)
