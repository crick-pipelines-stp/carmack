import logging
from typing import Dict, List, Optional
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from carmack.io.bed_file import BedFile


log = logging.getLogger(__name__)


class CellCaller:
    """
    Class that calculates peak-barcode matrix from tagged BAM alignment file and peak file. Cells
    can then be filtered out based on the number of peaks they have dynamically.
    """

    def __init__(self, bed: str, bam: str, bai: Optional[str] = None) -> None:
        self.bam_path = bam
        self.bai_path = bai
        self.bed_path = bed

        self.matrix = None  # Peak-barcode matrix

        log.debug(f"CellCaller object created with BAM: {bam}, BAI: {bai}, and BED: {bed}")

    def init_matrix(self, peak_names: List[str], barcodes: List[str]) -> pd.DataFrame:
        """
        Initialise the peak-barcode matrix.

        Each value indicates instances of overalap of a barcode with a peak.

                    Unique Barcodes
                ---------------------
                |  1   |   0   |  0
        Peak    |  3   |   1   |  0
        Names   |  0   |   0   | ...
                | ...
        """

        #### check each fragment for overlap with a peak and assign to a count per cell barcode
        self.matrix = pd.DataFrame(
            np.zeros((len(peak_names), len(barcodes)), dtype=int),
            index=peak_names,
            columns=barcodes,
        )
        log.debug(f"Peak-barcode matrix initialised with shape: {self.matrix.shape}")

    def check_overlap(self, peak: Dict, read: pysam.AlignedSegment):
        """
        Find if a read overlaps with a peak.
        """
        if peak["chrom"] == str(read.reference_name):
            if (
                read.reference_start is not None
                and peak["start"] <= int(read.reference_start) <= peak["end"]
            ):
                return True
            if (
                read.reference_end is not None
                and peak["start"] <= int(read.reference_end) <= peak["end"]
            ):
                return True

    def compute_matrix(self) -> pd.DataFrame:
        """
        Find all overlapping peaks for each parcode and fill the peak-barcode matrix.
        """
        log.debug("Computing peak-barcode matrix...")

        # Initialise readers and matrix
        bed = BedFile(self.bed_path)
        peaks = [entry for entry in bed.open_read_iterator()]
        barcodes = list(
            set(
                [
                    read.get_tag("BC")
                    for read in pysam.AlignmentFile(
                        self.bam_path, "rb", index_filename=self.bai_path
                    )
                ]
            )
        )
        self.init_matrix(peak_names=[entry["name"] for entry in peaks], barcodes=barcodes)

        # Compute matrix
        with pysam.AlignmentFile(self.bam_path, "rb", index_filename=self.bai_path) as bam:
            # Go through the reads
            for read in bam:
                barcode = read.get_tag("BC")
                # Go through the peaks
                for peak in peaks:
                    if self.check_overlap(peak, read):
                        self.matrix.at[peak["name"], barcode] += 1

        log.debug("Peak-barcode matrix computed.")

    def make_plot(self):
        """
        Create a barcode rank plot to visualise the number of overlapping peaks per barcode.
        """
        if self.matrix is None:
            raise ValueError("Matrix not computed yet. Run compute_matrix() method first.")

        barcode_sums = sorted(self.matrix.sum(), reverse=True)
        ranks = np.arange(1, len(barcode_sums) + 1)

        # TODO: Add logic to visualise knee plot, when implemented

        plt.figure(figsize=(10, 6))
        plt.plot(ranks, barcode_sums, color="blue")

        plt.xscale("log")
        plt.yscale("log")

        # Customise more
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel("Barcodes")
        plt.ylabel("Fragments overlapping peaks")
        plt.title("Barcode Rank Plot")
        plt.xlim(left=1)
        plt.tight_layout()

        return plt.gcf()
