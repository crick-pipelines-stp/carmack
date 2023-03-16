import os
import pytest
import numpy as np

import carmack.utils as utils
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.barcode.barcode_extractor import BarcodeExtractor

from ..utils import with_temporary_folder

R1_PATH = 'tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz'
R2_PATH = 'tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz'
CB_PATH = 'tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz'

# ------------------------------------------------------------------------------ #
# calc_raw_barcode_match_counts
# ------------------------------------------------------------------------------ #

@with_temporary_folder
def test_calc_raw_barcode_match_counts_hydrop(self, temp_path):
    """Test calculation of raw barcode match counts."""

    expected_hash = '5a150d2473fbf0b0bb7993dfa4e4063a'

    # Init
    test_file = os.path.join(temp_path, 'barcode_counts.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_counts = barcode_ext.calc_raw_barcode_match_counts()

    # print(bc_counts)

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_counts:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)


@with_temporary_folder
def test_calc_raw_barcode_match_distribution_hydrop(self, temp_path):
    """Test calculation of raw barcode match distribution."""

    expected_hash = '4f28f158699d07bf0a790a130a61d2a2'

    # Init
    test_file = os.path.join(temp_path, 'barcode_dist.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_dist:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)

# ------------------------------------------------------------------------------ #
# gen_nearby_seqs
# ------------------------------------------------------------------------------ #

# TODO: GEN BARCODE SET ONLY ONCE

@pytest.mark.parametrize("maxdist,seq", [(1, 'TGTAGCAAGN'), (2, 'TGTAGCAAGN'), (3, 'TGTAGCAANN'), (4, 'TGTANCANNN'), (4, 'NGTAGCANNN')])
def test_gen_nearby_seqs_withn(self, maxdist, seq):
    """Test generation of nearby sequences."""

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Generate nearby sequences
    seqs, error = zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

@pytest.mark.parametrize("maxdist", [1, 2])
@pytest.mark.parametrize("seq", ['AGTTNNN', 'AGCTNNNNNNN', 'AGCTNNNN', 'AGCTNNN', 'AGCTNNNN'])
def test_gen_nearby_seqs_withn_none(self, maxdist, seq):
    """Test generation of nearby sequences."""

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Generate nearby sequences
    nearby_seqs = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

    assert len(nearby_seqs) == 0

@pytest.mark.parametrize("maxdist,seq", [(1, 'TGTAGCAAGC'), (3, 'TGTAGCAAGG'), (5, 'TGTAGCAAGC'), (4, 'AGCTGGCCGG')])
def test_gen_nearby_seqs_no_n(self, maxdist, seq):
    """Test generation of nearby sequences."""

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Generate nearby sequences
    seqs, error = zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

@pytest.mark.parametrize("maxdist,seq,expected", [(1, 'TGTAGCAAGN', 1), (3, 'TGTAGCAAGN', 1), (5, 'TGTAGCAAGN', 7), (5, 'TGTAGCAANN', 5), (5, 'TGTANCAANN', 3)])
def test_gen_nearby_seqs_expected(self, maxdist, seq, expected):
    """Test generation of nearby sequences."""

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Generate nearby sequences
    seqs, error = zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_sets[0], maxdist))

    for seq in list(seqs):
        assert seq in barcode_sets[0]
    
    assert len(seqs) == expected

# ------------------------------------------------------------------------------ #
# gen_indel_set
# ------------------------------------------------------------------------------ #

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAAGN', 10, 0), ('NNTAGCAAGC', 10, 0), ('NNTAGCAAGC', 8, 0)])
def test_gen_indel_set_n_in_seq(self, seq, target_len, expected):
    """Test generation of indel sets with N in input sequence raises a ValueError."""
    with pytest.raises(ValueError):

        # Init
        qs = np.full(len(seq), 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)        

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAAGC', 10, 1), ('TGTAGCAAGCG', 11, 1), ('TGTAGCAAGCGG', 12, 1)])
def test_gen_indel_set_correct_length(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has correct length."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)
        
    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAGC', 10, 10), ('TGTAGCAAGC', 11, 11), ('TGTAGCAGC', 11, 55)])
def test_gen_indel_set_deletions(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has one or more deletions."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)
        
    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAGCGC', 10, 11), ('TGTAGCAGCCC', 10, 9), ('TGTAGCAGCGCA', 10, 63)])
def test_gen_indel_set_insertions(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has one or more insertions."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)
        
    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len", [('TGTAGCAGC', 10), ('TGTAGCAAGC', 11), ('TGTAGCAGC', 11)])
def test_gen_indel_set_qs_deletions(self, seq, target_len):
    """Test generation of indel sets with high quality score for position N."""

    # Init
    qs = np.full(len(seq), 30)
    expected_qs = np.full(target_len, 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)

    for qs_seq in qs_set:
        assert not all([a == b for a, b in zip(qs_seq, expected_qs)])

# ------------------------------------------------------------------------------ #
# correct_barcode_chunk
# ------------------------------------------------------------------------------ #

@pytest.mark.parametrize("seq, qs, max_corrections, target_len, dist_updates, expected_seq", [
('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, {'TGTAGCAAGT': 1000000}, 'TGTAGCAAGT'), # Sequence matches barcode and has a high quality score - should return the original sequence
('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1000000}, 'TGTAGCAAGT'), # Sequence matches barcode and has a low quality score - should return the original sequence as the max dist is not high enough to generate other possible barcodes
('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1000000}, 'TGTAGCAAGT'), # Sequence matches barcode and has a low quality score - should return the original sequence as the prior distribution is weighted to the matched barcode 
('TGTAGCAAGT', [10,10,10,10,10,10,10,10,10,10], 5, 10, {'TGAATCCACC': 1000000}, None), # Sequence matches barcode and has a low quality score - should nothing as the qual score is so poor
('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 4, 10, {'CATTGCGAGT': 10000000000, 'TGTAGCAAGT': 0}, 'CATTGCGAGT'), # Sequence matches barcode and has a low quality score - should return the a diff sequence as the prior distribution is weighted to another sequence
('TGTAGCAAGTT', [30,30,30,30,30,30,30,30,30,30], 2, 10, {}, 'TGTAGCAAGT'), # Insertion - ends up with 3 copies of the same barcode generated in different ways
('TGTAGCAAGTTT', [30,30,30,30,30,30,30,30,30,30], 3, 10, {}, 'TGTAGCAAGT'), # Insertion
('TGTAGCAGT', [30,30,30,30,30,30,30,30,30], 3, 10, {}, None), # Deletion - no real dominating prior means we cant decide with confidence
('TGTAGCGT', [30,30,30,30,30,30,30,30], 3, 10, {}, 'TGTAGCAAGT') # Deletion 
])
def test_correct_barcode_chunk(self, seq, qs, max_corrections, target_len, dist_updates, expected_seq):
    """Test generation of barcode correction."""

    # Init
    np_qs = np.asarray(qs)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Get bc counts
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_counts = barcode_ext.calc_raw_barcode_match_counts()

    # Set either very high or very low counts for target barcodes
    for key in dist_updates.keys():
        bc_counts[0][key] = dist_updates[key]

    # Inject the altered numbers and calc the distribution
    barcode_ext.bc_counts = bc_counts
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    # Correct barcode
    corr_seq, match_candidates, unnorm_posterior, posterior = BarcodeExtractor.correct_barcode_chunk(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist[0])
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
@pytest.mark.parametrize("seq, expected_seq", [
# ('CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT', 'GAACAGTAGTACGGTGGACTCAGTGTGGAA'),
('TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA', ''),
# ('GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC', ''),
# ('GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC', ''),
]) 
def test_correct_barcode_perm(self, seq, expected_seq):
    """Test correct of whole barcode read"""
    print("")

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)

    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    # # Test
    # bc, qs, msg = barcode_ext.correct_barcode(seq, qs, barcode_wl, barcode_set, bc_dist, chemistry, 2)

    # # Log

    # print(bc)
    # print(qs)
    # print(msg)

    # Assert

# def test_correct_barcode_md5(self, seq, qs, max_corrections, target_len, dist_updates, expected_seq):


# GAACAGTAGT ACGGTGGACT CAGTGTGGAA

# AATCTGCACACTCGATCAACTGGTCTACTC