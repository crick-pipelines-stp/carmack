
import logging
import itertools
import numpy as np

from..chemistry.chemistry_factory import ChemistryFactory
from ..io.fastq_file import FastqFile

DNA_ALPHABET = 'AGCT'
ALPHABET_MINUS = {char: {c for c in DNA_ALPHABET if c != char} for char in DNA_ALPHABET} # This is a set of alternative bases given a base
ALPHABET_MINUS['N'] = set(DNA_ALPHABET)

log = logging.getLogger(__name__)
class BarcodeExtractor:
    """
    Class that handles barcode extraction/correction from fastq files
    """

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
            barcodes = self.chemistry.subset_barcodes(seq)
 
            for idx, bc_set in enumerate(barcode_set):
                ext_bc = barcodes[idx]
                if barcodes[idx] in bc_set:
                    bc_counts[idx][ext_bc] = bc_counts[idx][ext_bc] + 1

        return bc_counts

    def calc_raw_barcode_match_dist(self) -> list:
        """
        Computes the distribution of raw barcode matches across the barcode set for the given chemistry.
        Prior distribution over barcodes, with pseudo-count based on matching barcodes only
        """
        
        # Get counts
        bc_counts = self.calc_raw_barcode_match_counts()

        # Calculate distribution
        for count_set in bc_counts:
            counts = np.array(list(count_set.values()), dtype=float) + 1.0
            total_dist = counts.sum()
            bc_dist = counts / total_dist
            count_set.update(zip(list(count_set.keys()), bc_dist))

        return bc_counts

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

        # If this is too far away then we just return None
        if mindist > maxdist:
            return None

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
    def gen_indel_set(seq, qs, target_len):
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

    # First function
    # @staticmethod
    # def correct_barcode(seq, barcode_set):
    #     if seq in barcode_set:
    #         return seq
    #     else:
    #         return None

    # Second function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #     else:
    #         return None

    # Third function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             # for new_seq, error_probs_sum in BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections):
    #             #      return new_seq, error_probs_sum
    #             val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #             return val
                
    #     else:
    #         return None
        
    #     return None


    # Fourth function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             # for new_seq, error_probs_sum in BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections):
    #             #      return new_seq, error_probs_sum
    #             val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #             return val
                
    #     else:
    #         return None

    # Fifth function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #             return val
                
    #     else:
    #         val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #         return val

    # Seventh function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             for new_seq, error_probs_sum in BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections):
    #                 return new_seq, error_probs_sum
                
    #     else:
    #         for new_seq, error_probs_sum in BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections):
    #             return new_seq, error_probs_sum
    
    # Eighth function
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections, target_len):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             for new_seq, error_probs_sum in BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections):
    #                  return new_seq, error_probs_sum
    #             # val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #             # return val
                
    #     else:
    #         for seq_set, qs_set  in BarcodeExtractor.gen_indel_set(seq, qs, target_len):
    #                 return seq_set, qs_set
            # val = list(BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
            # return val

    # Seventh function (for tenth test)
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, max_corrections, target_len):
    #     if seq in barcode_set:
    #         if (qs > 24).all():
    #             return seq
    #         else:
    #             for seq_set, qs_set in zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections)):
    #                 print(seq_set)
    #                 print(qs_set)
    #                 return seq_set, qs_set
                
    #     else:
    #         for seq_set, qs_set in zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections)):
    #                 print(seq_set)
    #                 print(qs_set)
    #                 return seq_set, qs_set
            # seq_set, qs_set = BarcodeExtractor.gen_indel_set(seq, qs, target_len)
            # return seq_set, qs_set
            # for each pair, execute gen_nearby_seqs
            # for seq_set, qs_set in BarcodeExtractor.gen_indel_set(seq, qs, target_len):
            #     print(seq_set)
            #     return seq_set, qs_set
            
            
    # 
    # @staticmethod
    # def correct_barcode(seq, qs, barcode_set, target_len): 
    #     """Estimate the correct barcode given an input sequence, base quality scores, a barcode whitelist, and a prior
    #     distribution of barcodes.  Returns the corrected barcode if the posterior likelihood is above the confidence
    #     threshold, otherwise None.  Only considers corrected sequences out to a maximum Hamming distance of 2
    #     """

    #     # Check for indels
    #     if seq in barcode_set:
    #         return seq
    #     else:
    #         # generate indel set
    #         # seq_set, qs_set  = BarcodeExtractor.gen_indel_set(seq, qs, target_len)
    #         # return seq_set, qs_set

    #         # # For each possible indel seq, get the possible sequences and the summed error prob
    #         # for seq, qs in seq_set, qs_set:
    #             new_seq, error_probs_sum = zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
    #             yield new_seq, error_probs_sum

            # generate nearby seqs
            # new_seq, error_probs_sum = zip(*BarcodeExtractor.gen_nearby_seqs(seq, qs, barcode_set, max_corrections))
            # yield new_seq, error_probs_sum
        

        # Check if the barcode is already in the barcode set
        # If it is, then put the sequence in the match candidates and likelihood to the current value and return this

        # If it is not a perfect match, check for indels
        # Generate the match set, which will be more than the input sequence if there are indels 
        

        # If there are indels, generate the nearby sequences for each sequence in that set
        #  gen_nearby_seqs yields new_seq, error_probs_sum
        # return only unique values for sequences

        # Rotate through each possible barcode in the match set of nearby sequences
            # evt. filter on qs   



    # @staticmethod
    # def correct_barcode(seq, qs, target_len):
    #     # og values: self, seq, qs, target_len, barcode_set, bc_distribution, max_corrections
    #     """Estimate the correct barcode given an input sequence, base quality scores, a barcode whitelist, and a prior
    #     distribution of barcodes.  Returns the corrected barcode if the posterior likelihood is above the confidence
    #     threshold, otherwise None.  Only considers corrected sequences out to a maximum Hamming distance of 2
    #     """
    #     match_candidates = []
    #     likelihoods = []

    #     # Generate the match set, which will be more than the input sequence if there are indels
    #     match_set, out_qs = BarcodeExtractor.gen_indel_set(seq, qs, target_len)        
    #     #return match_set, out_qs

        # # Rotate through each possible barcode in the match_set
        # for seq in match_set:
        #     # If we get a match and the seq quality is good across whole read then return the first one
        #     # this is because we only have multiple barcodes here if we have indel and then we have N's anyway
        #     if seq in barcode_set:
        #         if (out_qs > 24).all():
        #             return seq 

        #         # If the quality score is no good, then we add it as a candidate and do hamming correction anyway
        #         match_candidates.append(seq)
        #         likelihoods.append(bc_dist[seq])

    #     return match_candidates, likelihoods

            # # Cycle through sequence candidates and calculate the prob
            # for test_seq, error_probs in gen_nearby_seqs(seq, out_qs, barcode_set, max_corrections):
            #     # Get the prior prob of the barcode
            #     p_bc = bc_dist[test_seq]
            #     log10p_edit = error_probs / 10.0
            #     likelihoods.append(p_bc * (10 ** -log10p_edit))
            #     match_candidates.append(test_seq)

        # posterior = np.array(likelihoods)
        # posterior /= posterior.sum()

        # if len(posterior) > 0:
        #     pmax = posterior.max()
        #     if pmax > BC_CONFIDENCE_THRESHOLD:
        #         return match_candidates[np.argmax(posterior)]
        # return None

    # def analyse_barcodes(self) -> list:
    #     pass
        # with open(os.path.join(parsed_args.output, parsed_args.prefix + '.bc_all.csv'), "w") as file_all:
        # with open(os.path.join(parsed_args.output, parsed_args.prefix + '.bc_valid.csv'), "w") as file_valid:
        #     for (name, bc, qs, bc_code) in bc_iter:
        #         # Assign no match
        #         if bc is None:
        #             bc = "NO-MATCH"

        #         # Add to list of valid barcodes for counting
        #         if bc != "NO-MATCH" and bc in bc_list:
        #             bc_list[bc] = bc_list[bc] + 1
        #         elif bc != "NO-MATCH" and bc not in bc_list:
        #             bc_list[bc] = 1
        #             bc_count = bc_count + 1

        #         # Write barcodes to file
        #         file_all.write(name + "," + bc + "," + qs + "," + bc_code + '\n')
        #         if bc != "NO-MATCH":
        #             file_valid.write(name + "," + bc + "," + qs + "," + bc_code + '\n')
