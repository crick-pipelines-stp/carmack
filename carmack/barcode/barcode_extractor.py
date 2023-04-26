import os
import re
import operator
import logging
import itertools
import numpy as np

from..chemistry.chemistry_factory import ChemistryFactory
from..chemistry.chemistry_base import ChemistryBase
from ..io.fastq_file import FastqFile

DNA_ALPHABET = 'AGCT'
ALPHABET_MINUS = {char: {c for c in DNA_ALPHABET if c != char} for char in DNA_ALPHABET} # This is a set of alternative bases given a base
ALPHABET_MINUS['N'] = set(DNA_ALPHABET)
QS_SCORE_THRESHOLD = 24
BC_CONFIDENCE_THRESHOLD = 0.9
ILLUMINA_QUAL_MIN_SCORE = 2
ILLUMINA_QUAL_MAX_SCORE = 30
ILLUMINA_QUAL_OFFSET = 33

log = logging.getLogger(__name__)
class BarcodeExtractor:
    """
    Class that handles barcode extraction/correction from fastq files
    """

    bc_counts = None
    bc_dist = None

    def __init__(self, read1: str, read2 : str, cell_barcode: str, chemistry: str) -> None:
        self.read1 = read1
        self.read2 = read2
        self.cell_barcode = cell_barcode
        self.chemistry = ChemistryFactory.get_chemistry(chemistry)

    def calc_raw_barcode_match_counts(self) -> list:
        """
        Computes the counts of raw barcode matches across the barcode set for the given chemistry.
        """

        # Load the barcode set to match against
        barcode_set = self.chemistry.load_barcode_set()

        # Init counts
        bc_counts = []
        for bc_set in barcode_set:
            bc_counts.append({bc:0 for bc in bc_set})

        # Iterate over the fastq file and count the number of matches for each barcode
        fq_file = FastqFile(self.cell_barcode)
        stream = fq_file.open_read_iterator(as_string=True)
        for (name, seq, qual) in stream:
            barcodes, qs, msg = self.chemistry.subset_barcode_chunks(seq, qual)

            if barcodes is not None and msg == "SUBSET:OK":
                for idx, bc_set in enumerate(barcode_set):
                    ext_bc = barcodes[idx]
                    if barcodes[idx] in bc_set:
                        bc_counts[idx][ext_bc] = bc_counts[idx][ext_bc] + 1

        self.bc_counts = bc_counts
        return bc_counts

    def calc_raw_barcode_match_dist(self) -> list:
        """
        Computes the distribution of raw barcode matches across the barcode set for the given chemistry.
        Prior distribution over barcodes, with pseudo-count based on matching barcodes only
        """
        
        # Get counts
        if(self.bc_counts is None):
            self.bc_counts = self.calc_raw_barcode_match_counts()

        # Calculate distribution
        for count_set in self.bc_counts:
            counts = np.array(list(count_set.values()), dtype=float) + 1.0
            total_dist = counts.sum()
            bc_dist = counts / total_dist
            count_set.update(zip(list(count_set.keys()), bc_dist))

        self.bc_dist = bc_dist
        return self.bc_counts

    @staticmethod
    def gen_nearby_seqs(seq, qs, barcode_set, maxdist):
        """Generate all sequences with at most maxdist changes from seq that are in a provided seq, along with the
        quality values of the bases at the changed positions. Automatically will target N's in a sequence as letters
        which must be changed. If there are more N's than allowed changes - we return nothing
        """

        # Find all index positions which are not N in seq as a list
        non_n_indices = [i for i in range(len(seq)) if seq[i] != 'N']

        # Find all positions which are N in seq as a tuple
        n_indices = tuple([i for i in range(len(seq)) if seq[i] == 'N'])

        # The number of unknown N's dicates the minmimum hamming distance that combinations must be from the original sequence
        mindist = len(n_indices)

        # If this is too far away then we just return nothing
        if mindist > maxdist:
            return [], 0

        # If the input sequence is in the barcode set, include the seq and qs in the output
        if seq in barcode_set:
            yield seq, 0

        # Combinations are generated in batches by changing n number of indices in the sequence, then n+1 and so on
        # The min number of positions to change is dictated by the number of N's in the sequence
        # The max number of positions to change is dictated by the max hamming distance
        for dist in range(mindist, maxdist + 1):

            # Generate possible combinations of non-required indices to change for this hamming distance level
            # This list will be empty if the number of N's is equal to the hamming distance
            for modified_indices in itertools.combinations(non_n_indices, dist - mindist):

                # Combine the indices we have to change because of N's and the other potential cominations into a final 
                # List of indices to change
                indices = set(modified_indices + n_indices)

                # Convert the set to a list of indices for subsetting the qs scores (ignore the empty list at the beggining)                
                indices_list = np.array(list(indices))
                if len(indices_list) == 0: continue

                # Subset the quality scores for the indices we are changing and sum them
                error_probs = qs[indices_list]
                error_probs_sum = error_probs.sum()

                # Generate possible base substitutions from the indice positions using the minus alphabet
                for substitutions in itertools.product(*[ALPHABET_MINUS[base] if i in indices else base for i, base in enumerate(seq)]):
                    new_seq = ''.join(substitutions)

                    # If the new sequence is in the whitelist, sum the QS scores for the changed sequences and return 
                    if new_seq in barcode_set:
                        yield new_seq, error_probs_sum

    @staticmethod
    def gen_indel_set(seq, qs, target_len, maxdist):
        """Given an input sequence and a desired length, generate an exhaustive combinatorial 
        set of potential sequences with N in place of insertions. For qs, the indels are given
        a high qs as we are 100% sure about their letter given that we have inserted the N ourselves. 
        This seems counter-intuitive as it's an N, but this will be subsituted for a real letter 
        downstream that we are 100% sure on.
        """
        seq_set = []
        output_set = []
        output_qs = []
        seq_len = len(seq)
        seq_set.append(seq)

        # Check for N's in seq
        if "N" in seq:
            log.error(f"N detected when generating indel set - {seq}")
            return []

        # Return if seq is correct length
        if seq_len == target_len:
            return [seq],[qs]
        
        # Return if absolute difference between seq and   
        # target length is larger than max_corrections 
        diff =  seq_len - target_len
        if abs(diff) > maxdist:
            return [], []

        # DELETION
        if seq_len < target_len:
            while len(seq_set) > 0:
                curr_seq = seq_set.pop()
                curr_gen_seq = []
                for i in range(0, len(curr_seq) + 1):
                    new_seq_ar = list(curr_seq)
                    new_seq_ar.insert(i, 'N')
                    new_seq = ''.join(new_seq_ar)
                    if new_seq not in curr_gen_seq:
                        curr_gen_seq.append(new_seq)
                        if len(new_seq) < target_len:
                            seq_set.append(new_seq)
                        else:
                            if new_seq not in output_set:
                                output_set.append(new_seq)

        # INSERTION
        if seq_len > target_len:
            while len(seq_set) > 0:
                curr_seq = seq_set.pop()
                curr_gen_seq = []
                for i in range(0, len(curr_seq)):
                    new_seq_ar = list(curr_seq)
                    new_seq_ar.pop(i)
                    new_seq = ''.join(new_seq_ar)
                    if new_seq not in curr_gen_seq:
                        curr_gen_seq.append(new_seq)
                        if len(new_seq) > target_len:
                            seq_set.append(new_seq)
                        else:
                            if new_seq not in output_set:
                                output_set.append(new_seq)

        # Construct qs scores
        for seq in output_set:
            curr_qs = qs.copy()
            for idx, base in enumerate(seq):
                if base == 'N':
                    curr_qs=np.insert(curr_qs, idx, 50)
            output_qs.append(curr_qs)


        return output_set, output_qs

    @staticmethod
    def correct_barcode_chunk(seq, qs, barcode_set, max_corrections, target_len, bc_dist):
        """Given an input barcode chunk, generate a collection of nearby sequences for that chunk 
        that are at most max_corrections away from the input chunk. Use the summed error probabilities 
        for all added or modified indices to calculate the unnormalised posterior probability for 
        each nearby sequence. Return the nearby sequence with the highest posterior probability if 
        it exceeds the minimum barcode confidence threshold, or otherwise return None.             
        """
        # Init
        match_candidates = []
        unnorm_posterior = []
        posterior = []
        corr_seq = None

        # If the sequence matches
        if seq in barcode_set and (qs > QS_SCORE_THRESHOLD).all():
            corr_seq = seq
            match_candidates.append(seq)
            posterior.append(1.0)
            return corr_seq, match_candidates, unnorm_posterior, posterior

        # If the input sequence doesn't perfectly match an existing barcode, this can be either due to indels or mutations.
        # If there are indels, gen_indel_set will generate a set of sequences with the correct length containing N in each possible position and their associated qs
        # If the original sequence already had the correct length or or is already in the barcode_set but has low qs, it will just return the input sequence and qs 
        for indel_seq, indel_qs in zip(*BarcodeExtractor.gen_indel_set(seq, qs, target_len, max_corrections)):
            # For each indel sequence, generate all possible nearby sequences that are at most max_corrections (Hamming distance) away from the input sequence
            for curr_seq, error_sum in BarcodeExtractor.gen_nearby_seqs(indel_seq, indel_qs, barcode_set, max_corrections):
                # Find the posterior for each seq
                p_bc = bc_dist[curr_seq]

                # Divide by 10 to get the log10 probability for errors summed across all modified bases.
                log10p_edit = error_sum / 10.0

                # The likelihood is (10 ** -log10p_edit). This is the probability that the modified index or indices were originally incorrect. 
                # Multiply the prior with the likelihood to get the unnormalised posterior: p(A)*p(B|A)
                # Store the unnormalised posterior and the match candidates.
                unnorm_posterior.append(p_bc * (10 ** -log10p_edit))
                match_candidates.append(curr_seq)

        # Normalise the posterior values so that they all sum to 1
        posterior = np.array(unnorm_posterior)
        posterior /= posterior.sum()

        # Find the barcode that is above the confidence threshold and has maximal posterior probability
        if len(posterior) > 0:
            pmax = posterior.max()
            if pmax > BC_CONFIDENCE_THRESHOLD:
                corr_seq =  match_candidates[np.argmax(posterior)]
        
        return corr_seq, match_candidates, unnorm_posterior, posterior

    @staticmethod
    def correct_barcode(seq: str, qs: np.ndarray, barcode_wl: list, barcode_set: list, bc_dists: dict, chemistry: ChemistryBase, max_corrections: int):
        """Given a barcode sequence, correct each barcode chunk individually if it is not 
        immediately matched in the barcode whitelist. Join the corrected chunks and return 
        the entire corrected barcode sequence along with a message describing whether the 
        barcode the correction was succesfull or not, which corrections were applied to 
        each barcode chunk and potentially the cause of failed correction.    
        """
        # Init
        corr_bc = None
        msg = "{UNPROCESSED}"
        target_len = len(barcode_set[0][0])

        # First scale qs scores into a range so that the statistics dont get ruined by outliers
        qs[qs < ILLUMINA_QUAL_MIN_SCORE] = ILLUMINA_QUAL_MIN_SCORE
        qs[qs > ILLUMINA_QUAL_MAX_SCORE] = ILLUMINA_QUAL_MAX_SCORE

        # Generate a white list guess match
        wl_guess = chemistry.subset_whitelist_guess(seq)

        # Match to whitelist and exit if we have an immediate match accross all barcode chunks
        if wl_guess in barcode_wl:
            msg = "OK|WL_MATCH"
            return wl_guess, msg
        else:
            msg = "NIM"

        # Subset the barcode chunks
        bc_chunks, qs_chunks, sub_msg = chemistry.subset_barcode_chunks(seq, qs)
        msg = msg + "|" + sub_msg

        corr_chunks = []
        if bc_chunks is not None:
            # For each barcode chunk, correct
            for idx, bc_chunk in enumerate(bc_chunks):
                # Logging
                if len(bc_chunk) != target_len:
                    msg = msg + "|BC" + str(idx + 1) + ":INDL_" + str(len(bc_chunk))
                else:
                    msg = msg + "|BC" + str(idx + 1)

                # Correct  barcode chunk
                curr_corr_bc, match_candidates, unnorm_posterior, posterior = BarcodeExtractor.correct_barcode_chunk(bc_chunk, qs_chunks[idx], barcode_set[idx], max_corrections, target_len, bc_dists[idx])
                corr_chunks.append(curr_corr_bc)

                if curr_corr_bc is None:
                    msg = msg + ":CORRFAIL"
                else:
                    msg = msg + ":CORROK"

        # Assign corrected BC if the correction has not failed
        if "CORRFAIL" not in msg and ("SUBSET:OK" in msg or "SUBSET:INDL" in msg):
            corr_bc = ''.join(corr_chunks)
            msg = "OK|" + msg
        else:
            msg = "FAIL|" + msg

        return corr_bc, msg

    def get_corrected_barcode(self, barcode_wl: list, barcode_set: list, bc_dists: dict, max_corrections: int):
        """Given a FASTQ file containing cell barcodes, evaluate each barcode and correct it 
        if required. Return the read name, the original or corrected barcode and a message 
        describing the correction process. 
        """
        # Get the FASTQ file containing cell barcodes and create an iterator
        fq_file = FastqFile(self.cell_barcode)
        fastq_iter = fq_file.open_read_iterator(as_string=True)

        # Iterate over each line in the FASTQ file and correct the barcode in necessary
        for (name, seq, qs) in fastq_iter:
            # Conver the quality score to a numpy array 
            dqs = np.frombuffer(qs.encode('UTF-8'), dtype=np.byte) - ILLUMINA_QUAL_OFFSET

            # Match and correct the barcode
            corr_bc, msg = BarcodeExtractor.correct_barcode(seq, dqs, barcode_wl, barcode_set, bc_dists, self.chemistry, max_corrections)
            yield (name, corr_bc, msg)
    
    def report_extraction(line_index, full_match_fraction, corr_match_fraction, fail_match_fraction, fail_spc_notfnd_fraction, fail_corr_indl_fraction, fail_corr_mut_fraction, top_10_fractions):
        logging.info("SUMMARY")
        logging.info(f'{line_index:,}')
        logging.info("FULL-MATCH: " + str(full_match_fraction))
        logging.info("CORRECTED MATCH: " + str(corr_match_fraction))
        logging.info("FAILED MATCH: " + str(fail_match_fraction))
        logging.info("FAIL SPACER NOT FOUND: " + str(fail_spc_notfnd_fraction))
        logging.info("FAIL INDEL: " + str(fail_corr_indl_fraction))
        logging.info("FAIL MUTATION: " + str(fail_corr_mut_fraction))
        logging.info("PERCENTAGES TOP 10 BARCODES: " + str(top_10_fractions))

    def extract_cell_barcodes(self, max_corrections: int, k: int):
        """Given a FASTQ file containing cell barcodes, extract the corrected cell barcodes.
        Write output to separate files containing all barcodes, all matched barcodes, and stats
        for downstream visualisation.
        """
        # Init
        barcode_set = self.chemistry.load_barcode_set()
        barcode_wl = self.chemistry.construct_whitelist(barcode_set)
        barcode_ext = BarcodeExtractor(self.read1, self.read2, self.cell_barcode, self.chemistry)
        bc_dist = barcode_ext.calc_raw_barcode_match_dist()

        # Init
        line_index = 0
        bc_dict = {}
        msg_dict = {}

        # Open files and write
        with open(os.path.join(parsed_args.output, parsed_args.prefix +  '.bc_all.csv'), "w") as file_all: # parsed_args.output, parsed_args.prefix +  '.bc_all.csv'
            with open(os.path.join(parsed_args.output, parsed_args.prefix +'.bc_valid.csv'), "w") as file_valid: # parsed_args.output, parsed_args.prefix +'.bc_valid.csv'
                for (name, corr_bc, msg) in barcode_ext.get_corrected_barcode(barcode_wl, barcode_set, bc_dist, max_corrections):
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

                    # Write all barcodes to file_all
                    file_all.write(name + ',' + corr_bc + ',' + msg + '\n')

                    # Write all matched barcodes to file_valid
                    if corr_bc != "NO-MATCH":
                        file_valid.write(name + ',' + corr_bc + ',' + msg + '\n')
                    
                    # Stats logging
                    line_index += 1
                    if line_index % k == 0:
                        # Calculate percentage of full match, corrected match and failed match
                        total_msg_count = sum(msg_dict.values())
                        full_match_fraction = sum(dict(filter(lambda x: x[0] == 'OK|WL_MATCH', msg_dict.items())).values())/total_msg_count
                        corr_match_fraction = sum(dict(filter(lambda x: 'OK|NIM' in x[0], msg_dict.items())).values())/total_msg_count
                        fail_match_fraction = sum(dict(filter(lambda x: 'FAIL|NIM' in x[0], msg_dict.items())).values())/total_msg_count

                        # Calculate percentage of different categories of failed matches
                        total_fail_count = sum(dict(filter(lambda x: 'FAIL|NIM' in x[0], msg_dict.items())).values())
                        fail_spc_notfnd_fraction = sum(dict(filter(lambda x: 'NOTFND' in x[0], msg_dict.items())).values())/total_fail_count
                        fail_corr_indl_fraction = sum(dict(filter(lambda x: re.search('INDL_.*:CORRFAIL', x[0]), msg_dict.items())).values())/total_fail_count
                        fail_corr_mut_fraction = sum(dict(filter(lambda x: re.search('BC.:CORRFAIL', x[0]), msg_dict.items())).values())/total_fail_count

                        # Calculate percentage total barcodes that belong to the 10 most frequent barcodes
                        total_bc_count = sum(bc_dict.values())
                        bc_dict_matched = dict(filter(lambda x: x[0] != "NO-MATCH", bc_dict.items()))
                        bc_dict_sorted = dict(sorted(bc_dict_matched.items(), key=operator.itemgetter(1),reverse=True))
                        top_10_bcs = dict(itertools.islice(bc_dict_sorted.items(), 10))
                        top_10_counts = np.array(list(top_10_bcs.values()), dtype=float)
                        top_10_fractions = top_10_counts / total_bc_count

                        # Report logging information
                        barcode_ext.report_extraction(line_index, full_match_fraction, corr_match_fraction, fail_match_fraction, fail_spc_notfnd_fraction, fail_corr_indl_fraction, fail_corr_mut_fraction, top_10_fractions)

        # Update stats and write to stats files
