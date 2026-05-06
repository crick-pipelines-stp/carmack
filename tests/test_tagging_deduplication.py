import csv
import os

import pysam
import pytest

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
                assert read.has_tag("BC")
                assert read.has_tag("DU")

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
                assert read.has_tag("BC")
                assert read.has_tag("DU")

                ### Check if the barcodes are correctly assigned
                read_name = read.query_name
                barcode = read.get_tag("BC")
                assert bc_dict[read_name] == barcode

                if read.get_tag("DU") == 0:
                    unique_counter["bam_tagged"] += 1

        ## Check reads in deduplicated BAM
        with pysam.AlignmentFile(files["bam_dedup_file"], "rb") as bam_dedup:
            for read in bam_dedup:
                ### Check for presence of tags
                assert read.has_tag("BC")
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
