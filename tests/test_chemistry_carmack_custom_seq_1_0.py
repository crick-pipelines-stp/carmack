# pylint: disable=missing-function-docstring,missing-class-docstring

import os

import numpy as np
import pytest
from assertpy import assert_that

import carmack.utils as utils
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.chemistry_carmack_custom_seq_1_0 import ChemistryCarmackCustomSeq10
from carmack.io.fastq_file import FastqFile


TEST_BC_1 = "TGTAGCAAGT"
TEST_BC_2 = "TTAGTTGGAC"
TEST_BC_3 = "TGACCGTACT"

BC_READS_PATH = "tests/data/carmack/SK462_EKDL250005055-1A_22VGJ7LT4_L6_1.fastq.gz"


class TestChemistryCarmackCustomSeq10():
    def test_chem_carmack_cs10_load_barcode_set(self):
        # Setup
        chemistry = ChemistryCarmackCustomSeq10()

        # Test
        barcode_set = chemistry.load_barcode_set()

        # Assert
        assert_that(barcode_set).is_length(3)
        assert_that(list(barcode_set[0])).contains(TEST_BC_1)
        assert_that(list(barcode_set[1])).contains(TEST_BC_2)
        assert_that(list(barcode_set[2])).contains(TEST_BC_3)
        assert_that(len(barcode_set[0])).is_equal_to(96)
        assert_that(len(barcode_set[1])).is_equal_to(96)
        assert_that(len(barcode_set[2])).is_equal_to(96)

    def test_chem_carmack_cs10_load_barcode_set_with_factory(self):
        # Setup
        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")

        # Test
        barcode_set = chemistry.load_barcode_set()

        # Assert
        assert_that(list(barcode_set[0])).contains(TEST_BC_1)
        assert_that(list(barcode_set[1])).contains(TEST_BC_2)
        assert_that(list(barcode_set[2])).contains(TEST_BC_3)
        assert_that(len(barcode_set[0])).is_equal_to(96)
        assert_that(len(barcode_set[1])).is_equal_to(96)
        assert_that(len(barcode_set[2])).is_equal_to(96)

    def test_chem_carmack_cs10_construct_whitelist_model_case(self):
        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = chemistry.construct_whitelist(barcode_set)

        # KEEP UPDATED WITH MODEL SEQUENCE CONSTRUCTION
        test_wl = list(barcode_set[0])[0] + list(barcode_set[1])[0] + list(barcode_set[2])[0]

        assert test_wl in barcode_wl


    def test_chem_carmack_cs10_construct_whitelist_md5(self, tmp_path):
        expected_hash = "fa1805f3bd180de020eb7e6c6512ffeb"
        test_file = os.path.join(tmp_path, "barcodes.txt")

        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")
        barcode_set = chemistry.load_barcode_set()
        barcode_wl = sorted(chemistry.construct_whitelist(barcode_set))

        with open(test_file, "w") as out_file:
            for bc_wl in barcode_wl:
                out_file.write(bc_wl + "\n")
        utils.validate_file_md5(test_file, expected_hash)


    def test_chem_carmack_cs10_subset_barcodes_md5(self, tmp_path):
        expected_hash = "7a71e9a2e2de7ac3a371d5aa0a077c36"

        test_file = os.path.join(tmp_path, "barcodes.txt")
        fq_file = FastqFile(BC_READS_PATH)
        stream = fq_file.open_read_iterator(as_string=True)
        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")

        count = 0
        with open(test_file, "w") as out_file:
            for _, seq, _ in stream:
                qs = np.full(len(seq), 30)
                chemistry.subset_barcode_chunks(seq, qs)
                barcode_chunks, _, _ = chemistry.subset_barcode_chunks(seq, qs)

                if barcode_chunks is not None:
                    line = ",".join(barcode_chunks)
                    out_file.write(line + "\n")
                count += 1
                if count > 10:
                    break

        utils.validate_file_md5(test_file, expected_hash)

    @pytest.mark.parametrize(
        "seq, expected_seq",
        [
            ("TGTAGCAAGTATGGAAGCCGACGAATTAGACCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACT", None),  # < 96 bases
            ("TGTAGCAAGTATGGAAGCCGACGAATTAGACCAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACT", None),  # < 96 bases
            ("TGTAGCAAGTATGGAAGCCGACGAATTAGACCAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACTTGTGTATAAGGACCTCGTTGCC", "CTTGTGTATAACATGGAAGCGAATTAGACC"),  # >= 96 bases, expect first 96 bases
        ],
    )
    def test_chem_carmack_cs10_subset_whitelist_guess(self, seq, expected_seq):
        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")
        result = chemistry.subset_whitelist_guess(seq)
        assert_that(result).is_equal_to(expected_seq)

    @pytest.mark.parametrize(
        "seq, qs, expected_barcodes, expected_qs, expected_msg",
        [
            ("TGTAGCAAGTATGGAAGCCGACGAATTAGACCAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACTTGTGTATAAGGACCTCGTTGCC", np.array([30]*96), None, None, "SUBSET:PRIMC_NOTFND"),
            ("TGTAGCAAGTATGGAAGCCGACGAATTAGACCAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACTTGTGTATAAGGACCTCGTTGCCATGGAAGCCGACGAATTAGACC", np.array([30]*120), None, None, "SUBSET:PRIMC_NOTFND"),
            ("TGTGTATAAGGACCTCGTTGCCTTAGTTGGACATGGAAGCCGACGAATTAGACCTGACCGTACTATGGAAGCCGACGAATTAGACC", np.array([30]*96), None, None, "SUBSET:SEQLEN<96"),
        ],
    )
    def test_chem_carmack_cs10_subset_barcode_chunks(self, seq, qs, expected_barcodes, expected_qs, expected_msg):
        chemistry = ChemistryFactory.get_chemistry("carmack_custom_seq_1_0")
        barcodes, qs_out, msg = chemistry.subset_barcode_chunks(seq, qs)
        assert_that(msg).is_equal_to(expected_msg)
        if expected_barcodes is not None:
            assert_that(barcodes).is_equal_to(expected_barcodes)
        if expected_qs is not None:
            for arr1, arr2 in zip(qs_out, expected_qs):
                assert_that(np.array_equal(arr1, arr2)).is_true()
