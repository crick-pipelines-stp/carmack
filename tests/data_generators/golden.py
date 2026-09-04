# Provenance of the golden regression inputs in tests/data/golden/
#
# These FASTQ files are the fixed inputs to the golden output baseline. They are
# committed so that every later change to the barcode matchers can be measured
# against a known-good set of outputs.
#
#   custom_seq_1_0_R1.fastq.gz  - 2000 reads of a real carmack_custom_seq_1_0
#                                 library (SK661). Read layout:
#                                 BC3(10) PRIMER_C(22) BC2(10) PRIMER_A(22)
#                                 BC1(10) UMI(8) poly-G TGIDX(8) Tn5 insert
#
#   hydrop_R1.fastq.gz          - 2000 reads taken from the head of the
#                                 committed hydrop library, subset for speed:
#
#                                 zcat tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz \
#                                     | head -8000 | gzip -n > hydrop_R1.fastq.gz
#
# The two "small" inputs are the always-run tier. They are strict head-subsets of
# the full inputs above, chosen to keep the default test suite fast while still
# driving every matcher tier (EXACTMATCH, KMERMATCH and ALIGNMATCH all appear for
# all three barcode components in the resulting bc_stats.txt report):
#
#   zcat custom_seq_1_0_R1.fastq.gz | head -800 | gzip -n > custom_seq_1_0_small_R1.fastq.gz
#   zcat hydrop_R1.fastq.gz         | head -200 | gzip -n > hydrop_small_R1.fastq.gz
#
# gzip -n omits the timestamp so re-deriving the inputs reproduces identical bytes.
