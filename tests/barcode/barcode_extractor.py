import os
import pytest
import numpy as np
import carmack.utils as utils

from carmack.io.fastq_file import FastqFile
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

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAAGN', 10, 0), 
                                                     ('NNTAGCAAGC', 10, 0), 
                                                     ('NNTAGCAAGC', 8, 0)])
def test_gen_indel_set_n_in_seq(self, seq, target_len, expected):
    """Test generation of indel sets with N in input sequence raises a ValueError."""
    with pytest.raises(ValueError):

        # Init
        qs = np.full(len(seq), 30)

        # Generate indels
        seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)        

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAAGC', 10, 1), 
                                                     ('TGTAGCAAGCG', 11, 1), 
                                                     ('TGTAGCAAGCGG', 12, 1)])
def test_gen_indel_set_correct_length(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has correct length."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)
        
    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAGC', 10, 10), 
                                                     ('TGTAGCAAGC', 11, 11), 
                                                     ('TGTAGCAGC', 11, 55),
                                                     ('TGTGC', 10, 0)])
def test_gen_indel_set_deletions(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has one or more deletions."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)

    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len,expected", [('TGTAGCAGCGC', 10, 11), 
                                                     ('TGTAGCAGCCC', 10, 9), 
                                                     ('TGTAGCAGCGCA', 10, 63),
                                                     ('TGTAGCACGCCCGCGCA', 10, 0)])
def test_gen_indel_set_insertions(self, seq, target_len, expected):
    """Test generation of indel sets when input sequence has one or more insertions."""

    # Init
    qs = np.full(len(seq), 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)
        
    assert len(seq_set) == expected
    assert len(qs_set) == expected

@pytest.mark.parametrize("seq,target_len", [('TGTAGCAGC', 10), 
                                            ('TGTAGCAAGC', 11), 
                                            ('TGTAGCAGC', 11)])
def test_gen_indel_set_qs_deletions(self, seq, target_len):
    """Test generation of indel sets with high quality score for position N."""

    # Init
    qs = np.full(len(seq), 30)
    expected_qs = np.full(target_len, 30)

    # Generate indels
    seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len, 2)

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
@pytest.mark.parametrize("seq, expected_seq, expected_msg", [
('CAGTGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT', 'GAACAGTAGTACGGTGGACTCAGTGTGGAA', 'OK|WL_MATCH'), # Full whitelist match
('TCCTGATAAGAGGGTACTCGACCAAGAGAGCAGTAGCTGCTCCTCATCCGTA', None, 'FAIL|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:INDL_9:CORRFAIL|BC3:CORROK'), # Correction fail on chunk 3
('GAACTTGTAGAGGGTACTCGGGAGCTTGTCGCAGTAGCTGTTGAGATCGTAC', None, 'FAIL|NIM|SUBSET:OK|BC1:CORRFAIL|BC2:CORROK|BC3:CORROK'), # Correction fail on chunk 1 but no indels
('CATGTGGAAAGGGTACTCGACGGTGGACTGCAGTAGCTGGAACAGTAGTGT', 'GAACAGTAGTACGGTGGACTCAGTGTGGAA', 'OK|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:CORROK|BC3:INDL_9:CORROK'), # Indel but correction ok
('TGTCACAACAAGGGTACTCGGTCCAGGCTTGCAGGAGCGGGACTTGTGGCGT', None, 'FAIL|NIM|SUBSET:SPC2_NOTFND'), # Fail because spacer not found
('TTGTCCGCCAAGGGTACTCGTATGCAGTAGCTGCGTCAGACAAGTACTCTGC', None, 'FAIL|NIM|SUBSET:INDL|BC1:INDL_17:CORRFAIL|BC2:INDL_3:CORRFAIL|BC3:CORROK') # Fail because too many indels
]) 
def test_correct_barcode_perm(self, seq, expected_seq, expected_msg):
    """Test correction of whole barcode read"""
    print("")

    # Init
    qs = np.full(len(seq), 30)
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)

    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    # Test
    bc, msg = barcode_ext.correct_barcode(seq, qs, barcode_wl, barcode_set, bc_dist, chemistry, 2)

    # Log
    # print(bc)
    # print(msg)

    # Assert
    assert bc == expected_seq
    assert msg == expected_msg

@with_temporary_folder
def test_correct_barcode_md5(self, temp_path):
    """Test calculation of raw barcode match distribution."""
    print("")

    expected_hash = '81208771bc217c14628cc59bb886c6fd'

    # Init
    skip_op = 100
    test_file = os.path.join(temp_path, 'corrected.txt')
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    # Iterate cell barcode reads and correct barcodes
    count = 0
    fq_file = FastqFile(barcode_ext.cell_barcode)
    stream = fq_file.open_read_iterator(as_string=True)
    with open(test_file, 'w') as out_file:
        for (name, seq, qs) in stream:

            if count % skip_op == 0:
                dqs = np.frombuffer(qs.encode('UTF-8'), dtype=np.byte) - 33
                bc, msg = barcode_ext.correct_barcode(seq, dqs, barcode_wl, barcode_set, bc_dist, chemistry, 2)

                if bc is None:
                    bc = ""

                line = bc + ',' + msg
                out_file.write(line + '\n')

                # print(line)

            count = count + 1

    utils.validate_file_md5(test_file, expected_hash)

# ------------------------------------------------------------------------------ #
# get_corrected_barcode
# ------------------------------------------------------------------------------ #

@pytest.mark.parametrize("expected_read_name, expected_corr_bc, expected_msg, line", [
('NB501505:171:H3KMGAFX3:4:21612:13641:20150 2:N:0:CTATAGTCTT', 'CCGTTCGTCCAATAGCGTGGGGTTAATCAC', 'OK|WL_MATCH', 0), # Full whitelist match
('NB501505:171:H3KMGAFX3:3:21601:7618:10703 2:N:0:CTATAGTCTT', None, 'FAIL|NIM|SUBSET:SPC2_NOTFND', 5), # Fail because spacer not found
('NB501505:171:H3KMGAFX3:3:11402:13723:5588 2:N:0:CTATAGTCTT', None, 'FAIL|NIM|SUBSET:INDL|BC1:INDL_11:CORROK|BC2:INDL_9:CORRFAIL|BC3:CORROK', 6), # Correction fail on chunk 3
('NB501505:171:H3KMGAFX3:2:21203:12986:2165 2:N:0:CTATAGTCTT', 'TTGCAGTTCTACACGTTGTGAGTTGGAAGA', 'OK|NIM|SUBSET:OK|BC1:CORROK|BC2:CORROK|BC3:CORROK', 13) # No immediate match, but no indels
]) 
def test_get_corrected_barcode_messages(self, expected_read_name, expected_corr_bc, expected_msg, line):
    """Test barcode messaging"""
    # Init
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    count = 0
    # Yield barcode name, corrected barcode and message
    for (name, corr_bc, msg) in barcode_ext.get_corrected_barcode(barcode_wl, barcode_set, bc_dist, 2):
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
def test_get_corrected_barcode_md5(self, temp_path):
    """Test barcode correction"""

    expected_hash = 'b5a7aa8b036fee4e2dddeb2eee9a45ea'

    # Init
    test_file = os.path.join(temp_path, 'barcodes_corrected.txt')
    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()


    # Iterate over all cell barcodes, correct them and write the output to a file
    with open(test_file, 'w') as out_file:
        for (name, corr_bc, msg) in barcode_ext.get_corrected_barcode(barcode_wl, barcode_set, bc_dist, 2):
            if corr_bc is None:
                corr_bc = ""

            line = name + ',' + corr_bc + ',' + msg
            out_file.write(line + '\n')

            # print(line)
        
    utils.validate_file_md5(test_file, expected_hash)

@pytest.mark.parametrize("expected_full_match_fraction, expected_corr_match_fraction, expected_fail_match_fraction, expected_fail_spc_notfnd_fraction, expected_fail_corr_indl_fraction, expected_fail_corr_base_sub_fraction, expected_top_10_fractions", [
(0.8651, 0.0692, 0.0657, 0.6605783866057838, 0.091324200913242, 0.2480974124809741, [0.0088, 0.007 , 0.0065, 0.0058, 0.0055, 0.0034, 0.0029, 0.0028, 0.0026, 0.0025])
]) 
def test_stats_calc(self, expected_full_match_fraction, expected_corr_match_fraction, expected_fail_match_fraction, expected_fail_spc_notfnd_fraction, expected_fail_corr_indl_fraction, expected_fail_corr_base_sub_fraction, expected_top_10_fractions):
    """Test stats calculation"""
    # Init
    bc_dict = {}
    msg_dict = {}
    stats_dict = {}

    chemistry = ChemistryFactory.get_chemistry('hydrop')
    barcode_set = chemistry.load_barcode_set()
    barcode_wl = chemistry.construct_whitelist(barcode_set)
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    bc_dist = barcode_ext.calc_raw_barcode_match_dist()

    # Iterate over all cell barcodes, correct them and write the output to a file
    for (name, corr_bc, msg) in barcode_ext.get_corrected_barcode(barcode_wl, barcode_set, bc_dist, 2):
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
    assert stats_dict['full_match_fraction'] == expected_full_match_fraction
    assert stats_dict['corr_match_fraction'] == expected_corr_match_fraction
    assert stats_dict['fail_match_fraction'] == expected_fail_match_fraction
    assert stats_dict['fail_spc_notfnd_fraction'] == expected_fail_spc_notfnd_fraction
    assert stats_dict['fail_corr_indl_fraction'] == expected_fail_corr_indl_fraction
    assert stats_dict['fail_corr_base_sub_fraction'] == expected_fail_corr_base_sub_fraction
    assert all([a == b for a, b in zip(stats_dict['top_10_fractions'], expected_top_10_fractions)])

@with_temporary_folder
def test_extract_cell_barcodes_md5(self, temp_path):
    """Test cell barcode extraction"""

    expected_hash_file_all = '880f4312a753e63d9d8a81f1e7010f6a'
    expected_hash_file_valid = '55a3edbc0adf3a4552f9cdf5cfaeeb4d'
    expected_hash_file_bc_stats = '78bbdffef6690227bd8bc28b95646480'
    expected_hash_file_bc_counts = 'b49cdb0ad6b3fc4178a10f4e75e8b4be'

    # Init
    max_corrections = 2
    count = 100
    prefix = ''
    test_file_all = os.path.join(temp_path, prefix + '.bc_all.csv')
    test_file_valid = os.path.join(temp_path, prefix +'.bc_valid.csv')
    test_file_bc_stats = os.path.join(temp_path, prefix + '.bc_counts_stats.csv')
    test_file_bc_counts = os.path.join(temp_path, prefix + '.bc_counts.csv')
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')

    # Run extract_cell_barcodes
    barcode_ext.extract_cell_barcodes(max_corrections, count, temp_path, prefix)

    # Check md5
    utils.validate_file_md5(test_file_all, expected_hash_file_all)
    utils.validate_file_md5(test_file_valid, expected_hash_file_valid)
    utils.validate_file_md5(test_file_bc_stats, expected_hash_file_bc_stats)
    utils.validate_file_md5(test_file_bc_counts, expected_hash_file_bc_counts)

@with_temporary_folder
def test_extract_cell_barcodes_prefix(self, temp_path):
    """Test barcode extraction using either no or a user-specified prefix"""
    # Init
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    max_corrections = 2
    count = 100

    # Run extract_cell_barcodes
    barcode_ext.extract_cell_barcodes(max_corrections, count, temp_path, prefix='hydrop_scatac_1_S2_R2_001')

    # Get files
    files = os.listdir(temp_path)

    # assert prefix in all filenames
    assert all('hydrop_scatac_1_S2_R2_001' in filename for filename in files)

@with_temporary_folder
def test_extract_cell_barcodes_no_prefix(self, temp_path):
    """Test barcode extraction using either no or a user-specified prefix"""
    # Init
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    max_corrections = 2
    count = 100

    # Run extract_cell_barcodes
    barcode_ext.extract_cell_barcodes(max_corrections, count, temp_path)

    # Get files
    files = os.listdir(temp_path)
    barcode_ext = BarcodeExtractor(R1_PATH, R2_PATH, CB_PATH, 'hydrop')
    prefix = barcode_ext.read1.rsplit("/", 1)[-1].split(".", 1)[0]
    
    # assert prefix in all filenames
    assert all(prefix in filename for filename in files)

