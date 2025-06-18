# Prepare peak file to test call-cells module
# This peak file is compatible to work with hydrop_scatac reads as they share the same cell line
#
# We process the file to remove most of the reads which do not overlap with the reads in the
# hydrop_scatac file. Aim is to reduce the file size for bundling with test data.
#
#   1. Download
#      Description: ATAC-seq on human cell line K562 (DNA), paired-end
#      Source: ENCODE
#      File: ENCFF349AKO.bam
#      Link: https://www.encodeproject.org/files/ENCFF349AKO/
#
#   2. Find overlapping peaks
#      ```bash
#      bedtools intersect -u -a ENCFF349AKO.bed -b hydrop_scatac_1_S1_R1.dedup.tagged.bam > atac_k562_peaks.bed
#      ````
#
#   3. Find non-overlapping peaks
#      ```bash
#      bedtools intersect -v -a ENCFF349AKO.bed -b hydrop_scatac_1_S1_R1.dedup.tagged.bam > non_overlapping_peaks.bed
#      ```
#
#   4. Manually copy some non-overlapping peaks to the atac_k562_peaks.bed file
#      (For the sake of testing, we are going to keep some non-overlapping peaks)
#
#   5. Sort the bed file
#      ```bash
#      bedtools sort -i atac_k562_peaks.bed > atac_k562_peaks.sorted.bed
#      ```

