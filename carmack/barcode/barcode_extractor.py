
from..chemistry.chemistry_factory import ChemistryFactory
from ..io.fastq_file import FastqFile

class BarcodeExtractor:
    """
    Class that handles barcode extraction/correction from fastq files
    """

    def __init__(self, read1: str, read2 : str, cell_barcode: str, chemistry: str) -> None:
        self.read1 = read1
        self.read2 = read2
        self.cell_barcode = cell_barcode
        self.chemistry = ChemistryFactory.get_chemistry(chemistry)

    def calc_raw_barcode_match_distribution(self) -> list:
        """
        Computes the distribution of raw barcode matches across the barcode set for the given chemistry.
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
