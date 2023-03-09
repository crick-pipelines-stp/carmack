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

@with_temporary_folder
def test_calc_raw_barcode_match_counts_hydrop(self, temp_path):
    """Test calculation of raw barcode match counts."""

    expected_hash = '777cf483afbc2db0408f151baa693037'

    # Init
    test_file = os.path.join(temp_path, 'barcode_counts.txt')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Calc distribution
    bc_counts = barcode_ext.calc_raw_barcode_match_counts()

    with open(test_file, 'w') as out_file:
       for bc_count_set in bc_counts:
            for bc in bc_count_set:
                line = bc + '-' + str(bc_count_set[bc])
                out_file.write(line + '\n')
    
    utils.validate_file_md5(test_file, expected_hash)


@with_temporary_folder
def test_calc_raw_barcode_match_distribution_hydrop(self, temp_path):
    """Test calculation of raw barcode match distribution."""

    expected_hash = '18d6549067432e6ca3d7d7843f02059c'

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


# First test
# # Testing correct_barcode_set
# @pytest.mark.parametrize("seq, expected", [('TGTAGCAAGT', 'TGTAGCAAGT'), ('NNNNNNNNNN', None)]) 
# def test_correct_barcode(self, seq, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     # qs = np.full(len(seq), 30)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     #print(barcode_sets[0])

#     # Correct barcode
#     val = BarcodeExtractor.correct_barcode(seq, barcode_sets[0])
#     print(val)
#     assert val == expected

# Second test
# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, expected", [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 'TGTAGCAAGT'), 
#                                                ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], None), 
#                                                ('TGTAGCAAGT', [22,30,22,30,22,30,22,30,22,30], None), 
#                                                ('NNNNNNNNNN', [30,30,30,30,30,30,30,30,30,30], None)]) 
# def test_correct_barcode(self, seq, qs, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     val = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0])
#     print(val)
#     assert val == expected

# # Third test
# # Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGNAAGT', [30,30,30,30,30,30,30,30,30,30], 1, [('TGTAGCAAGT', 30)]), 
#                                                                 ('TGTAGCACGT', [22,22,22,22,22,22,22,22,22,22], 1, [('TGTAGCAAGT', 22)]), 
#                                                                 ('TGTAGCCCGT', [22,30,22,30,22,30,22,30,22,30], 2, [('TGTAGCAAGT', 52)]), 
#                                                                 ('NNNNNNNNNN', [30,30,30,30,30,30,30,30,30,30], 1, [])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     val = list(BarcodeExtractor.gen_nearby_seqs(seq, np_qs, barcode_sets[0], max_corrections))
#     print(val)
#     assert val == expected


# Fourth test
# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 'TGTAGCAAGT'), 
#                                                                 ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 4, [('CATTGCGAGT', 88)]), 
#                                                                 ('TGTAGCAAGT', [22,30,22,30,22,30,22,30,22,30], 4, [('CATTGCGAGT', 104)]), 
#                                                                 ('NNNNNNNNNN', [30,30,30,30,30,30,30,30,30,30], 1, None)]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     output = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
#     #print(output)
#     assert output == expected

# Sixth test
# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCANGT', [30,30,30,30,30,30,30,50,30,30], 1, None), 
#                                                                 ('TGTAGAGT'  , [22,22,22,22,22,22,22,22,22,22], 4, None), 
#                                                                 ('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 4, None), 
#                                                                 ('NNNNNNNNNN', [30,30,30,30,30,30,30,30,30,30], 1, None)]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     output = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
#     #print(output)
#     assert output == expected

# Seventh test
# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCANGT', [30,30,30,30,30,30,30,50,30,30], 1, ('TGTAGCAAGT', 50)), 
#                                                                 ('TGTAGAGT'  , [22,22,22,22,22,22,22,22,22,22], 4, None), 
#                                                                 ('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 1, ('TGTAGCAAGT', 22)),
#                                                                 ('NNNNNNNNNN', [30,30,30,30,30,30,30,30,30,30], 1, None)]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     output = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
#     assert output == expected

# # Eighth test
# # Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 4, [('TGTAGCAAGT', 22), ('CATTGCGAGT', 104), ('AATAGGCAGG', 112), ('TGAATCCACC', 96)])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     output = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
#     # print(output)
#     assert output == expected


#WIP
# Ninth test
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCANGT', [30,30,30,30,30,30,30,50,30,30], 1, ('TGTAGCAAGT', 50)), 
#                                                                 ('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 1, ('TGTAGCAAGT', 22)), 
#                                                                 ('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 4, [('TGTAGCAAGT', 22), ('CATTGCGAGT', 104), ('AATAGGCAGG', 112), ('TGAATCCACC', 96)])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

    # Correct barcode
    # a,b = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
    # print(a)
    # print(b)


# Tenth test: Shorter sequence
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, expected", [('TGTAGCAAG', [30,30,30,30,30,30,30,30,30], 1, 10, [])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     # seq_set, qs_set = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len)
#     seq_set, qs_set = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len)
#     print(seq_set)
#     print(qs_set)

# Testing gen_indel_set output
#         seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)        
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, expected", [('TGTAGCAAG', [30,30,30,30,30,30,30,30,30], 1, 10, [])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     seq_set, qs_set = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len)
#     print(seq_set)
#     print(qs_set)


#  New tests
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, expected_seq", [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, 'TGTAGCAAGT'), 
#                                                                                 ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 1, 10, 'TGTAGCAAGT')]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, expected_seq):
#     """Test generation of barcode correction."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len)
#     print(seq)

#     assert seq == expected_seq  
        
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, expected_seq", [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, 'TGTAGCAAGT'), 
#                                                                                 ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 1, 10, 'TGTAGCAAGT')]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, expected_seq):
#     """Test generation of barcode correction."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     bc_dist = { 'TGTAGCAAGT': 0.8,
#                 'TGTAGCCAGT': 0.2}
#     bc_threshold = 0.1

#     # Correct barcode
#     seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist, bc_threshold)
#     #print(seq)

#     assert seq == expected_seq  

# THIS TEST IS FOR WHEN THE BC_THRESHOLD WAS SET TO 0.1 POSTERIOR PROBABILITY
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, bc_dist, expected_seq", 
# [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'), 
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 4, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.9999999999}, 'CATTGCGAGT'),
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.999999999, 'GCAAGCGTGT': 1e-10, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'CATTGCGAGT'),
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.4999999995, 'GCAAGCGTGT': 0.4999999995, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'CATTGCGAGT'),
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 1e-10, 'GCAAGCGTGT': 1e-10, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 0.4999999995, 'TGAATCCACC': 0.4999999995}, 'TGTAGCAAGT'),
# ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 1e-10, 'GCAAGCGTGT': 0.4999999995, 'CATCTCAGGT': 0.4999999995, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'TGTAGCAAGT')]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, bc_dist, expected_seq):
#     """Test generation of barcode correction."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     bc_threshold = 0.1

#     # Correct barcode
#     seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist, bc_threshold)
#     assert seq == expected_seq  

# Test for when input seq is not in the barcode set, because it has indels or has been mutated 
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, bc_dist, expected_seq", 
# [('TGTAGCAAT', [30,30,30,30,30,30,30,30,30], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
# ('TGTAGCAAT', [22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
# ('TGTAGCAAGTC', [22,22,22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
# ('TGTCAGCAAGTC', [22,22,22,22,22,22,22,22,22,22,22,22], 2, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
# ('TGTAGCAAGTC', [22,22,22,22,22,22,22,22,22,22,22], 3, 10, {'TGTAGCAAGT': 0.9999999992, 'GTGGAAGGTC': 1e-10, 'AGAGAATGTC': 1e-10, 'TGTGCGATTA': 1e-10, 'CTTAGCACTC': 1e-10, 'TGTAGCAAGT': 1e-10, 'TGTAGCAAGT': 1e-10, 'CTTAGCACTC': 1e-10, 'TGTAGCAAGT': 1e-10}, 'TGTAGCAAGT')]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, bc_dist, expected_seq):
#     """Test generation of barcode correction."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist)
#     # print(seq)
#     assert seq == expected_seq  

# Refactoring testing
@pytest.mark.parametrize("seq, qs, max_corrections, target_len, bc_dist, expected_seq", 
[('TGTAGCAT', [30,30,30,30,30,30,30,30], 2, 10, {'TGTAGCAAGT': 1, 'TGTAGCAAGT': 0}, 'TGTAGCAAGT'),
('TGTAGCAAT', [22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT')]) 
def test_correct_barcode(self, seq, qs, max_corrections, target_len, bc_dist, expected_seq):
    """Test generation of barcode correction."""

    # Init
    np_qs = np.asarray(qs)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_sets = chemistry.load_barcode_set()

    # Correct barcode
    seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist)
    print(seq)
    #assert seq == expected_seq  

# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, bc_dist, expected_seq", [('TGTAGCAAGT', [30,30,30,30,30,30,30,30,30,30], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'), 
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 1, 10, {'TGTAGCAAGT': 1, 'CATTGCGAGT': 0}, 'TGTAGCAAGT'),
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 4, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.9999999999}, 'CATTGCGAGT'),
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.999999999, 'GCAAGCGTGT': 1e-10, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'CATTGCGAGT'),
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 0.4999999995, 'GCAAGCGTGT': 0.4999999995, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'CATTGCGAGT'),
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 1e-10, 'GCAAGCGTGT': 1e-10, 'CATCTCAGGT': 1e-10, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 0.4999999995, 'TGAATCCACC': 0.4999999995}, 'TGTAGCAAGT'),
#                                                                                          ('TGTAGCAAGT', [22,22,22,22,22,22,22,22,22,22], 5, 10, {'TGTAGCAAGT': 1e-10, 'CATTGCGAGT': 1e-10, 'GCAAGCGTGT': 0.4999999995, 'CATCTCAGGT': 0.4999999995, 'AATAGGCAGG': 1e-10, 'CTTAGCACTC': 1e-10, 'CGTTATACGT': 1e-10, 'CGTAGTTACA': 1e-10, 'TTCATGAGGT': 1e-10, 'TGCGACAATG': 1e-10, 'TGAATCCACC': 1e-10}, 'TGTAGCAAGT')]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, bc_dist, expected_seq):
#     """Test generation of barcode correction."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     bc_threshold = 0.1

#     # for seq, error_sum in BarcodeExtractor.gen_nearby_seqs(seq, np_qs, barcode_sets[0], max_corrections):
#     #     print(seq)

#     # Correct barcode
#     seq = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len, bc_dist, bc_threshold)
#     #print(seq)
#     assert seq == expected_seq  
        

# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, expected", [('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 4, [('TGTAGCAAGT', 22), ('CATTGCGAGT', 104), ('AATAGGCAGG', 112), ('TGAATCCACC', 96)])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     seqs, error = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections)
#     print(seq)
    # for seq in list(seqs):
    #     print(seq) 

    # for seq in list(seqs):
    #     assert seq in barcode_sets[0]
    #output = )
    # print(output)
    #assert output == expected

# ('TGTAGCCAGT', [22,30,22,30,22,30,22,30,22,30], 4, [('TGTAGCAAGT', 22), ('CATTGCGAGT', 104), ('AATAGGCAGG', 112), ('TGAATCCACC', 96)]), 
# Eight test
# Testing correct_barcode_set
# @pytest.mark.parametrize("seq, qs, max_corrections, target_len, expected", [('TGTAGAGT', [22,22,22,22,22,22,22,22,22,22], 4, 10, [])]) 
# def test_correct_barcode(self, seq, qs, max_corrections, target_len, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     np_qs = np.asarray(qs)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()

#     # Correct barcode
#     seq_set, qs_set = BarcodeExtractor.correct_barcode(seq, np_qs, barcode_sets[0], max_corrections, target_len)
#     print(seq_set)
    # assert output == expected


# # Testing correct_barcode_set
# @pytest.mark.parametrize("maxdist,seq,expected", [(1, 'TGTAGCAAGN', 1)])
# def test_correct_barcode(self, maxdist, seq, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     qs = np.full(len(seq), 30)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     # print(barcode_sets[0])

#     # Correct barcode 
#     seqs, error = BarcodeExtractor.correct_barcode(seq, qs, barcode_sets[0], maxdist)
#     print(seqs)
#     print(error)
    
#     assert len([seqs[0]]) == expected

# # Testing correct_barcode_set
# @pytest.mark.parametrize("seq,target_len,max_corrections,expected", [('TGTAGCAGC', 10, 10)])
# def test_correct_barcode(self,seq, target_len, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     qs = np.full(len(seq), 30)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     # print(barcode_sets[0])

#     # Correct barcode 
#     seqs, qs_set = BarcodeExtractor.correct_barcode(seq, qs, barcode_sets[0], target_len)
#     print(seqs)
#     print(qs_set)
    
#     assert len(seqs) == expected
#     assert len(qs_set) == expected

# Testing correct_barcode_set
# @pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAGC', 10, 10)])
# def test_correct_barcode(self,seq, target_len, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     qs = np.full(len(seq), 30)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     # print(barcode_sets[0])

#     # Correct barcode 
#     seqs, qs_set = BarcodeExtractor.correct_barcode(seq, qs, barcode_sets[0], target_len)
#     print(seqs)
#     print(qs_set)
    
#     assert len(seqs) == expected
#     assert len(qs_set) == expected

# # Testing correct_barcode_set
# @pytest.mark.parametrize("seq,target_len, max_corrections,expected", [('TGTAGCAGC', 10, 1, 10)])
# def test_correct_barcode(self,seq, target_len, max_corrections, expected):
#     """Test generation of barcode correction when barcode matches perfectly with a barcode in the barcode set."""

#     # Init
#     qs = np.full(len(seq), 30)
#     chemistry = ChemistryFactory.get_chemistry('hydrop')
#     barcode_sets = chemistry.load_barcode_set()
#     # print(barcode_sets[0])

#     # Correct barcode 
#     new_seq, error_probs_sum = zip(*BarcodeExtractor.correct_barcode(seq, qs, barcode_sets[0], target_len, max_corrections))
#     # print(seqs)
#     # print(qs_set)
    
#     # assert len(seqs) == expected
#     # assert len(qs_set) == expected