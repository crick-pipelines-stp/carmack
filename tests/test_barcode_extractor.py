# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest

import numpy as np
import pytest

import carmack.utils as utils
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile
from tests.utils import with_temporary_folder


R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"
R2_PATH = "tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz"
CB_PATH = "tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz"


class TestBarcodeExtractor(unittest.TestCase):
    # ------------------------------------------------------------------------------ #
    # calc_raw_barcode_match_counts
    # ------------------------------------------------------------------------------ #

    @with_temporary_folder
    def test_bcext_calc_raw_barcode_match_counts_hydrop(self, temp_path):
        """Test calculation of raw barcode match counts."""

        # Setup
        expected_hash = "21f54e9372a0b5b304064cb88cff39f6"
        test_file = os.path.join(temp_path, "barcode_counts.txt")
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")

        # Test
        bc_counts = barcode_ext.calc_raw_barcode_match_counts()

        # Assert
        with open(test_file, "w") as out_file:
            for bc_count_set in bc_counts:
                sorted_bc_set = sorted(bc_count_set)
                for bc in sorted_bc_set:
                    line = bc + "-" + str(bc_count_set[bc])
                    out_file.write(line + "\n")

        utils.validate_file_md5(test_file, expected_hash)

    @with_temporary_folder
    def test_bcext_calc_raw_barcode_match_distribution_hydrop(self, temp_path):
        """Test calculation of raw barcode match distribution."""

        # Setup
        expected_hash = "3b2ee579aaa8e472c635cfc04e6cdeca"
        test_file = os.path.join(temp_path, "barcode_dist.txt")
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")

        # Test
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Assert
        with open(test_file, "w") as out_file:
            for bc_count_set in bc_dist:
                sorted_bc_set = sorted(bc_count_set)
                for bc in sorted_bc_set:
                    line = bc + "-" + str(bc_count_set[bc])
                    out_file.write(line + "\n")

        utils.validate_file_md5(test_file, expected_hash)

    @with_temporary_folder
    def test_bcext_extract_cell_barcodes_md5(self, temp_path):
        """Test cell barcode extraction"""

        expected_hash_file_all = "880f4312a753e63d9d8a81f1e7010f6a"
        expected_hash_file_valid = "55a3edbc0adf3a4552f9cdf5cfaeeb4d"
        expected_hash_file_bc_stats = "287dddba1fa60acdbfcffd8b18d691a0"
        expected_hash_file_bc_counts = "b49cdb0ad6b3fc4178a10f4e75e8b4be"

        # Init
        max_corrections = 2
        log_freq = 100
        prefix = utils.get_prefix(R1_PATH)
        print_stats = True
        test_file_all = os.path.join(temp_path, prefix + ".bc_all.csv")
        test_file_valid = os.path.join(temp_path, prefix + ".bc_valid.csv")
        test_file_bc_stats = os.path.join(temp_path, prefix + ".bc_counts_stats.csv")
        test_file_bc_counts = os.path.join(temp_path, prefix + ".bc_counts.csv")
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")

        # Run extract_cell_barcodes
        barcode_ext.extract_cell_barcodes(
            output_dir=temp_path,
            max_corrections=max_corrections,
            log_freq=log_freq,
            print_stats=print_stats,
            prefix=prefix,
        )

        # Check md5
        utils.validate_file_md5(test_file_all, expected_hash_file_all)
        utils.validate_file_md5(test_file_valid, expected_hash_file_valid)
        utils.validate_file_md5(test_file_bc_stats, expected_hash_file_bc_stats)
        utils.validate_file_md5(test_file_bc_counts, expected_hash_file_bc_counts)

    @with_temporary_folder
    def test_bcext_extract_cell_barcodes_prefix(self, temp_path):
        """Test barcode extraction using either no or a user-specified prefix"""
        # Init
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        max_corrections = 2
        log_freq = 100
        print_stats = True

        # Run extract_cell_barcodes
        barcode_ext.extract_cell_barcodes(
            output_dir=temp_path,
            max_corrections=max_corrections,
            log_freq=log_freq,
            print_stats=print_stats,
            prefix="hydrop_scatac_1_S2_R2_001",
        )

        # Get files
        files = os.listdir(temp_path)

        # assert prefix in all filenames
        assert all("hydrop_scatac_1_S2_R2_001" in filename for filename in files)

    @with_temporary_folder
    def test_bcext_extract_cell_barcodes_no_prefix(self, temp_path):
        """Test barcode extraction using either no or a user-specified prefix"""
        # Init
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        max_corrections = 2
        log_freq = 100
        print_stats = True

        # Run extract_cell_barcodes
        barcode_ext.extract_cell_barcodes(
            output_dir=temp_path,
            max_corrections=max_corrections,
            log_freq=log_freq,
            print_stats=print_stats,
        )

        # Get files
        files = os.listdir(temp_path)
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        prefix = barcode_ext.read1.rsplit("/", 1)[-1].split(".", 1)[0]

        # assert prefix in all filenames
        assert all(prefix in filename for filename in files)


class TestBarcodeExtractorFixtures:

    # ------------------------------------------------------------------------------ #
    # gen_nearby_seqs
    # ------------------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "maxdist,seq",
        [
            (1, "TGTAGCAAGN"),
            (2, "TGTAGCAAGN"),
            (3, "TGTAGCAANN"),
            (4, "TGTANCANNN"),
            (4, "NGTAGCANNN"),
        ],
    )
    def test_bcext_gen_nearby_seqs_withn(self, maxdist, seq):
        """Test generation of nearby sequences."""

        # Setup
        qs = np.full(len(seq), 30)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_sets = chemistry.load_barcode_set()

        # Test
        seqs, error = zip(
            *BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist)
        )  # pylint: disable=unused-variable

        # Assert
        assert len(seqs) == 1

    @pytest.mark.parametrize("maxdist", [1, 2])
    @pytest.mark.parametrize("seq", ["AGTTNNN", "AGCTNNNNNNN", "AGCTNNNN", "AGCTNNN", "AGCTNNNN"])
    def test_bcext_gen_nearby_seqs_withn_none(self, maxdist, seq):
        """Test generation of nearby sequences."""

        # Setup
        qs = np.full(len(seq), 30)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_sets = chemistry.load_barcode_set()

        # Test
        nearby_seqs = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

        # Assert
        assert len(nearby_seqs) == 0

    @pytest.mark.parametrize(
        "maxdist,seq,exp",
        [(1, "TGTAGCAAGC", 1), (3, "TGTAGCAAGG", 1), (5, "TGTAGCAAGC", 12), (4, "AGCTGGCCGG", 2)],
    )
    def test_bcext_gen_nearby_seqs_no_n(self, maxdist, seq, exp):
        """Test generation of nearby sequences."""

        # Setup
        qs = np.full(len(seq), 30)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_sets = chemistry.load_barcode_set()

        # Test
        seqs, error = zip(
            *BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist)
        )  # pylint: disable=unused-variable

        # Assert
        assert len(seqs) == exp

    @pytest.mark.parametrize(
        "maxdist,seq,expected",
        [
            (1, "TGTAGCAAGN", 1),
            (3, "TGTAGCAAGN", 1),
            (5, "TGTAGCAAGN", 7),
            (5, "TGTAGCAANN", 5),
            (5, "TGTANCAANN", 3),
        ],
    )
    def test_bcext_gen_nearby_seqs_expected(self, maxdist, seq, expected):
        """Test generation of nearby sequences."""

        # Init
        qs = np.full(len(seq), 30)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_sets = chemistry.load_barcode_set()

        # Generate nearby sequences
        seqs, error = zip(
            *BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist)
        )  # pylint: disable=unused-variable

        for seq in list(seqs):
            assert seq in barcode_sets[0]

        assert len(seqs) == expected

    # ------------------------------------------------------------------------------ #
    # gen_indel_set
    # ------------------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "seq,target_len,expected",
        [("TGTAGCAAGC", 10, 1), ("TGTAGCAAGCG", 11, 1), ("TGTAGCAAGCGG", 12, 1)],
    )
    def test_bcext_gen_indel_set_correct_length(self, seq, target_len, expected):
        """Test generation of indel sets when input sequence has correct length."""

        # Init
        qs = np.full(len(seq), 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)

        assert len(seq_set) == expected
        assert len(qs_set) == expected

    @pytest.mark.parametrize(
        "seq,target_len,expected",
        [("TGTAGCAGC", 10, 10), ("TGTAGCAAGC", 11, 11), ("TGTAGCAGC", 11, 55), ("TGTGC", 10, 0)],
    )
    def test_bcext_gen_indel_set_deletions(self, seq, target_len, expected):
        """Test generation of indel sets when input sequence has one or more deletions."""

        # Init
        qs = np.full(len(seq), 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)

        assert len(seq_set) == expected
        assert len(qs_set) == expected

    @pytest.mark.parametrize(
        "seq,target_len,expected",
        [
            ("TGTAGCAGCGC", 10, 11),
            ("TGTAGCAGCCC", 10, 9),
            ("TGTAGCAGCGCA", 10, 63),
            ("TGTAGCACGCCCGCGCA", 10, 0),
        ],
    )
    def test_bcext_gen_indel_set_insertions(self, seq, target_len, expected):
        """Test generation of indel sets when input sequence has one or more insertions."""

        # Init
        qs = np.full(len(seq), 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)

        assert len(seq_set) == expected
        assert len(qs_set) == expected

    @pytest.mark.parametrize(
        "seq,target_len", [("TGTAGCAGC", 10), ("TGTAGCAAGC", 11), ("TGTAGCAGC", 11)]
    )
    def test_bcext_gen_indel_set_qs_deletions(self, seq, target_len):
        """Test generation of indel sets with high quality score for position N."""

        # Init
        qs = np.full(len(seq), 30)
        expected_qs = np.full(target_len, 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(
            seq, qs, target_len, 2
        )  # pylint: disable=unused-variable

        for qs_seq in qs_set:
            assert not all([a == b for a, b in zip(qs_seq, expected_qs)])

    # ------------------------------------------------------------------------------ #
    # correct_barcode_chunk
    # ------------------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "seq, qs, max_corrections, target_len, dist_updates, expected_seq",
        [
            (
                "TGTAGCAAGT",
                [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
                1,
                10,
                {"TGTAGCAAGT": 1000000},
                "TGTAGCAAGT",
            ),  # Sequence matches barcode and has a high quality score - should return the original sequence
            (
                "TGTAGCAAGT",
                [22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                1,
                10,
                {"TGTAGCAAGT": 1000000},
                "TGTAGCAAGT",
            ),  # Sequence matches barcode and has a low quality score - should return the original sequence as the max dist is not high enough to generate other possible barcodes
            (
                "TGTAGCAAGT",
                [22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                5,
                10,
                {"TGTAGCAAGT": 1000000},
                "TGTAGCAAGT",
            ),  # Sequence matches barcode and has a low quality score - should return the original sequence as the prior distribution is weighted to the matched barcode
            (
                "TGTAGCAAGT",
                [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                5,
                10,
                {"TGAATCCACC": 1000000},
                None,
            ),  # Sequence matches barcode and has a low quality score - should nothing as the qual score is so poor
            (
                "TGTAGCAAGT",
                [22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                4,
                10,
                {"CATTGCGAGT": 10000000000, "TGTAGCAAGT": 0},
                "CATTGCGAGT",
            ),  # Sequence matches barcode and has a low quality score - should return the a diff sequence as the prior distribution is weighted to another sequence
            (
                "TGTAGCAAGTT",
                [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
                2,
                10,
                {},
                "TGTAGCAAGT",
            ),  # Insertion - ends up with 3 copies of the same barcode generated in different ways
            (
                "TGTAGCAAGTTT",
                [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
                3,
                10,
                {},
                "TGTAGCAAGT",
            ),  # Insertion
            (
                "TGTAGCAGT",
                [30, 30, 30, 30, 30, 30, 30, 30, 30],
                3,
                10,
                {},
                None,
            ),  # Deletion - no real dominating prior means we cant decide with confidence
            ("TGTAGCGT", [30, 30, 30, 30, 30, 30, 30, 30], 3, 10, {}, "TGTAGCAAGT"),  # Deletion
        ],
    )
    def test_bcext_correct_barcode_chunk(
        self, seq, qs, max_corrections, target_len, dist_updates, expected_seq
    ):
        """Test generation of barcode correction."""

        # Init
        np_qs = np.asarray(qs)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_sets = chemistry.load_barcode_set()

        # Get bc counts
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_counts = barcode_ext.calc_raw_barcode_match_counts()

        # Set either very high or very low counts for target barcodes
        for key in dist_updates.keys():
            bc_counts[0][key] = dist_updates[key]

        # Inject the altered numbers and calc the distribution
        barcode_ext.bc_counts = bc_counts
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Correct barcode
        corr_seq, match_candidates, unnorm_posterior, posterior = (
            BarcodeExtractor.correct_barcode_chunk(
                seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist[0]
            )
        )  # pylint: disable=unused-variable
        # print("")
        # print(match_candidates)
        # print(unnorm_posterior)
        # print(posterior)
        # print(corr_seq)
        assert corr_seq == expected_seq

    # ------------------------------------------------------------------------------ #
    # correct_barcode
    # ------------------------------------------------------------------------------ #

    # Refactoring testing
    @pytest.mark.parametrize(
        "seq, expected_seq, expected_msg",
        [
            (
                "CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT",
                "GAACAGTAGTACGGTGGACTCAGTGTGGAA",
                "OK|WL_MATCH",
            ),  # Full whitelist match
            (
                "TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA",
                None,
                "FAIL|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:INDL_9:CORRFAIL|BC3:CORROK",
            ),  # Correction fail on chunk 3
            (
                "GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC",
                None,
                "FAIL|NIM|SUBSET:OK|BC1:CORRFAIL|BC2:CORROK|BC3:CORROK",
            ),  # Correction fail on chunk 1 but no indels
            (
                "CATGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT",
                "GAACAGTAGTACGGTGGACTCAGTGTGGAA",
                "OK|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:CORROK|BC3:INDL_9:CORROK",
            ),  # Indel but correction ok
            (
                "TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT",
                None,
                "FAIL|NIM|SUBSET:SPC2_NOTFND",
            ),  # Fail because spacer not found
            (
                "TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC",
                None,
                "FAIL|NIM|SUBSET:INDL|BC1:INDL_17:CORRFAIL|BC2:INDL_3:CORRFAIL|BC3:CORROK",
            ),  # Fail because too many indels
        ],
    )
    def test_bcext_correct_barcode_perm(self, seq, expected_seq, expected_msg):
        """Test correction of whole barcode read"""
        print("")

        # Init
        qs = np.full(len(seq), 30)
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)

        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()
        barcode_chunk = chemistry.subset_barcode_chunks(seq, qs)

        # Test
        bc, msg = barcode_ext.correct_barcode(
            seq, barcode_chunk, barcode_wl, barcode_set, bc_dist, chemistry, 2
        )

        # Assert
        assert bc == expected_seq
        assert msg == expected_msg

    @with_temporary_folder
    def test_bcext_correct_barcode_md5(self, temp_path=None):
        """Test calculation of raw barcode match distribution."""

        expected_hash = "81208771bc217c14628cc59bb886c6fd"

        # Init
        skip_op = 100
        test_file = os.path.join(temp_path, "corrected.txt")
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Iterate cell barcode reads and correct barcodes
        count = 0
        fq_file = FastqFile(barcode_ext.cell_barcode)
        stream = fq_file.open_read_iterator(as_string=True)
        with open(test_file, "w") as out_file:
            for _, seq, qs in stream:
                dqs = np.frombuffer(qs.encode("UTF-8"), dtype=np.byte) - 33
                barcode_chunk = chemistry.subset_barcode_chunks(seq, dqs)

                if count % skip_op == 0:
                    bc, msg = barcode_ext.correct_barcode(
                        seq, barcode_chunk, barcode_wl, barcode_set, bc_dist, chemistry, 2
                    )

                    if bc is None:
                        bc = ""

                    line = bc + "," + msg
                    out_file.write(line + "\n")

                count = count + 1

        utils.validate_file_md5(test_file, expected_hash)

    # ------------------------------------------------------------------------------ #
    # get_corrected_barcode
    # ------------------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "expected_read_name, expected_corr_bc, expected_msg, line",
        [
            (
                "NB501505:171:H3KMGAFX3:4:21612:13641:20150 2:N:0:CTATAGTCTT",
                "CCGTTCGTCCAATAGCGTGGGGTTAATCAC",
                "OK|WL_MATCH",
                0,
            ),  # Full whitelist match
            (
                "NB501505:171:H3KMGAFX3:3:21601:7618:10703 2:N:0:CTATAGTCTT",
                None,
                "FAIL|NIM|SUBSET:SPC2_NOTFND",
                5,
            ),  # Fail because spacer not found
            (
                "NB501505:171:H3KMGAFX3:3:11402:13723:5588 2:N:0:CTATAGTCTT",
                None,
                "FAIL|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:INDL_9:CORRFAIL|BC3:CORROK",
                6,
            ),  # Correction fail on chunk 3
            (
                "NB501505:171:H3KMGAFX3:2:21203:12986:2165 2:N:0:CTATAGTCTT",
                "TTGCAGTTCTACACGTTGTGAGTTGGAAGA",
                "OK|NIM|SUBSET:OK|BC1:CORROK|BC2:CORROK|BC3:CORROK",
                13,
            ),  # No immediate match, but no indels
        ],
    )
    def test_bcext_get_corrected_barcode_messages(
        self, expected_read_name, expected_corr_bc, expected_msg, line
    ):
        """Test barcode messaging"""
        # Init
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        count = 0
        # Yield barcode name, corrected barcode and message
        for name, corr_bc, msg in barcode_ext.get_corrected_barcode(
            barcode_wl, barcode_set, bc_dist, 2
        ):
            # print(name)
            # print(corr_bc)
            # print(msgs)
            # print(count)

            if count == line:
                # Assert
                assert name == expected_read_name
                assert corr_bc == expected_corr_bc
                assert msg == expected_msg
                break

            count = count + 1

    @with_temporary_folder
    def test_bcext_get_corrected_barcode_md5(self, temp_path=None):
        """Test barcode correction"""

        expected_hash = "b5a7aa8b036fee4e2dddeb2eee9a45ea"

        # Init
        test_file = os.path.join(temp_path, "barcodes_corrected.txt")
        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Iterate over all cell barcodes, correct them and write the output to a file
        with open(test_file, "w") as out_file:
            for name, corr_bc, msg in barcode_ext.get_corrected_barcode(
                barcode_wl, barcode_set, bc_dist, 2
            ):
                if corr_bc is None:
                    corr_bc = ""

                line = name + "," + corr_bc + "," + msg
                out_file.write(line + "\n")

                # print(line)

        utils.validate_file_md5(test_file, expected_hash)

    @pytest.mark.parametrize(
        "expected_full_match_fraction, expected_corr_match_fraction, expected_fail_match_fraction, expected_fail_spc_notfnd_fraction, expected_fail_corr_indl_fraction, expected_fail_corr_base_sub_fraction, expected_top_10_fractions",
        [
            (
                0.8651,
                0.0692,
                0.0657,
                0.6605783866057838,
                0.091324200913242,
                0.2480974124809741,
                [0.0088, 0.007, 0.0065, 0.0058, 0.0055, 0.0034, 0.0029, 0.0028, 0.0026, 0.0025],
            )
        ],
    )
    def test_bcext_stats_calc(
        self,
        expected_full_match_fraction,
        expected_corr_match_fraction,
        expected_fail_match_fraction,
        expected_fail_spc_notfnd_fraction,
        expected_fail_corr_indl_fraction,
        expected_fail_corr_base_sub_fraction,
        expected_top_10_fractions,
    ):
        """Test stats calculation"""
        # Init
        bc_dict = {}
        msg_dict = {}
        stats_dict = {}

        chemistry = ChemistryFactory.get_chemistry("hydrop")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)
        barcode_ext = BarcodeExtractor(R1_PATH, CB_PATH, "hydrop")
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Iterate over all cell barcodes, correct them and write the output to a file
        for name, corr_bc, msg in barcode_ext.get_corrected_barcode(
            barcode_wl, barcode_set, bc_dist, 2
        ):
            if corr_bc is None:
                corr_bc = "NO-MATCH"

            # Add to a dictionary of unique barcodes for counting
            if corr_bc in bc_dict:
                bc_dict[corr_bc] += 1
            else:
                bc_dict[corr_bc] = 1

            # Add to a dictionary of unique barcode messages for counting
            if msg in msg_dict:
                msg_dict[msg] += 1
            else:
                msg_dict[msg] = 1

        # Stats logging
        stats_dict = BarcodeExtractor.stats_calc(msg_dict, bc_dict)
        # print(stats_dict)

        # assert output in stats_dict with expected values
        assert stats_dict["full_match_fraction"] == expected_full_match_fraction
        assert stats_dict["corr_match_fraction"] == expected_corr_match_fraction
        assert stats_dict["fail_match_fraction"] == expected_fail_match_fraction
        assert stats_dict["fail_spc_notfnd_fraction"] == expected_fail_spc_notfnd_fraction
        assert stats_dict["fail_corr_indl_fraction"] == expected_fail_corr_indl_fraction
        assert stats_dict["fail_corr_base_sub_fraction"] == expected_fail_corr_base_sub_fraction
        assert all(
            [a == b for a, b in zip(stats_dict["top_10_fractions"], expected_top_10_fractions)]
        )

    def test_bcext_stats_calc_granular_carmack_custom_seq(self):
        """Test granular stats calculation for carmack custom seq chemistry failure patterns"""
        # Setup test message dictionary with carmack custom seq specific patterns
        msg_dict = {
            "OK|WL_MATCH": 100,
            "OK|NIM|SUBSET:OK|BC1:CORROK|BC2:CORROK|BC3:CORROK": 50,
            "FAIL|NIM|SUBSET:SEQLEN<96": 17,
            "FAIL|NIM|SUBSET:SEQSHORT1": 19,
            "FAIL|NIM|SUBSET:SEQSHORT2": 23,
            "FAIL|NIM|SUBSET:NOTFNDBC1": 29,
            "FAIL|NIM|SUBSET:NOTFNDBC2": 31,
            "FAIL|NIM|SUBSET:NOTFNDBC3": 37,
            "FAIL|NIM|SUBSET:AMBIGBC1": 41,
            "FAIL|NIM|SUBSET:AMBIGBC2": 43,
            "FAIL|NIM|SUBSET:AMBIGBC3": 47,
            "FAIL|NIM|SUBSET:INDL|BC1:CORRFAIL|BC2:CORROK|BC3:CORROK": 15,
            "FAIL|NIM|SUBSET:INDL|BC1:CORROK|BC2:CORRFAIL|BC3:CORROK": 12,
            "FAIL|NIM|SUBSET:INDL|BC1:CORROK|BC2:CORROK|BC3:CORRFAIL": 8,
            "FAIL|NIM|SUBSET:OK|BC1:CORRFAIL|BC2:CORROK|BC3:CORROK": 25,
            "FAIL|NIM|SUBSET:OK|BC1:CORROK|BC2:CORRFAIL|BC3:CORROK": 18,
            "FAIL|NIM|SUBSET:OK|BC1:CORROK|BC2:CORROK|BC3:CORRFAIL": 13,
            "FAIL|NIM|SUBSET:INDL|BC1:CORRFAIL|BC2:CORRFAIL|BC3:CORROK": 5,
            "FAIL|NIM|SUBSET:OK|BC1:CORRFAIL|BC2:CORRFAIL|BC3:CORRFAIL": 4,
        }

        bc_dict = {
            "VALID_BC_1": 80,
            "VALID_BC_2": 70,
            "NO-MATCH": sum([v for k, v in msg_dict.items() if "FAIL" in k]),
        }

        # Test
        stats_dict = BarcodeExtractor.stats_calc(msg_dict, bc_dict)

        # Assert granular statistics (keys in barcode_extractor.py)
        # Sequence too short buckets
        assert stats_dict["fail_seqlen_short_count"] == 17
        assert stats_dict["fail_seqlen_short_count_bc1"] == 19
        assert stats_dict["fail_seqlen_short_count_bc2"] == 23

        # No-alignments buckets
        assert stats_dict["fail_noalignments_bc1"] == 29
        assert stats_dict["fail_noalignments_bc2"] == 31
        assert stats_dict["fail_noalignments_bc3"] == 37

        # Ambiguous-alignment buckets
        assert stats_dict["fail_ambigalignment_bc1"] == 41
        assert stats_dict["fail_ambigalignment_bc2"] == 43
        assert stats_dict["fail_ambigalignment_bc3"] == 47

        # Barcode Chunk-Specific Correction Failures
        # BC1 failures: 15 + 25 + 5 + 4 = 49
        assert stats_dict["fail_bc1_corrfail_count"] == 49
        # BC2 failures: 12 + 18 + 5 + 4 = 39
        assert stats_dict["fail_bc2_corrfail_count"] == 39
        # BC3 failures: 8 + 13 + 4 = 25
        assert stats_dict["fail_bc3_corrfail_count"] == 25

        # Indel vs Substitution Context
        # Indel with correction failure: 15 + 12 + 8 + 5 = 40
        assert stats_dict["fail_indel_with_corrfail_count"] == 40
        # Substitution only (no indel) with correction failure: 25 + 18 + 13 + 4 = 60
        assert stats_dict["fail_substitution_only_corrfail_count"] == 60

        # Multi-chunk Failures
        # Multiple chunks failed: 5 + 4 = 9
        assert stats_dict["fail_multiple_chunks_count"] == 9
