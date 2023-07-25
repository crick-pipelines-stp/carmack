import pysam

BAM_PATH = 'data/h3k27me3_R1.bam'

def test_bam_file_contents(self):
    # Open BAM file
    with pysam.AlignmentFile(BAM_PATH, "rb") as bam_file:
        # Iterate over the first 10 lines
        for i, read in enumerate(bam_file):
            print(read)  # Or do any other processing with the read
            if i == 100:
                break  

