import logging
from typing import Iterable, List, Tuple

from scipy.sparse import csr_matrix, lil_matrix

log = logging.getLogger(__name__)


class PeakBarcodeMatrix:
    """
    A class to store and manipulate the peak-barcode matrix for a given single-cell sequencing
    dataset.

    Each value indicates instances of overalap of a barcode with a peak.

                    Unique Barcodes
                ---------------------
                |  1   |   0   |  0
        Peak    |  3   |   1   |  0
        Names   |  0   |   0   | ...
                | ...

    Attributes
    ----------
    peaks : Iterable[str]
        Iterable of peak names.
    barcodes : Iterable[str]
        Iterable of unique cell barcodes.
    matrix: lil_matrix
        Sparse matrix of peak-barcode

    Methods
    -------
    increment_index(peak_name, barcode)
        Increment the matrix value at the given peak and barcode index by 1.
    get_value(peak_name, barcode)
        Get the matrix value at the given peak and barcode index.
    filter_matrix(row_indices, column_indices)
        Filter the matrix by the given row and column indices. Returns a csr_matrix.
    sum_barcodes()
        Sum the matrix along the barcode axis to get the number of overlapping peaks per barcode.
    """

    def __init__(self, peak_names: Iterable[str], barcodes: Iterable[str], force=False) -> None:
        """
        Initialise the peak-barcode matrix.

        Parameters
        ----------
        peak_names : Iterable[str]
            Tuple of peak names.
        barcodes : Iterable[str]
            Set of unique cell barcodes.
        force : bool, optional
            Overwrite matrix if already initialised, by default False
        """
        self.peaks = self.check_elements(peak_names)
        self.barcodes = self.check_elements(barcodes)
        self.matrix = lil_matrix((len(peak_names), len(barcodes)), dtype=int)
        log.debug(f"Peak-barcode matrix initialised with shape: {self.matrix.shape} (lil_matrix)")

    def check_elements(self, iter: Iterable) -> Tuple[str]:
        """
        Check if all elements of the iterable are strings. Raise error, if not.

        Returns
        -------
        Tuple[str]
            Sorted tuple of elements as strings.
        """
        if not all(isinstance(element, str) for element in iter):
            raise TypeError("All elements should be of type str.")

        return tuple(sorted(iter))

    def get_index(self, peak_name, barcode) -> Tuple[int, int]:
        """
        Get the matrix index for the given peak and barcode.
        """
        if peak_name not in self.peaks:
            raise ValueError(f"Peak name '{peak_name}' not found in peak list.")
        if barcode not in self.barcodes:
            raise ValueError(f"Barcode '{barcode}' not found in barcode list.")

        # Get index of peak and barcode
        peak_idx = self.peaks.index(peak_name)
        barcode_idx = self.barcodes.index(barcode)

        return peak_idx, barcode_idx

    def get_value(self, peak_name, barcode) -> int:
        """
        Get the matrix value at the given peak and barcode index.
        """
        idx = self.get_index(peak_name, barcode)
        return self.matrix[idx]

    def increment_index(self, peak_name, barcode) -> None:
        """
        Increment the matrix value at the given peak and barcode index by 1.
        """
        idx = self.get_index(peak_name, barcode)

        # Increment value
        self.matrix[idx] += 1

    def filter_matrix(self, row_indices, column_indices) -> csr_matrix:
        """
        Filter the matrix by the given row and column indices. Converts and returns a csr_matrix
        for efficient slicing operations.
        """

        def check_indices(axis, indices):
            if not all(0 <= idx < axis for idx in indices):
                raise ValueError(f"Invalid indices for axis {axis}.")

        check_indices(self.matrix.shape[0], row_indices)
        check_indices(self.matrix.shape[1], column_indices)

        return self.matrix.tocsr()[row_indices, :][:, column_indices]

    def sum_barcodes(self) -> List[int]:
        """
        Sum the matrix along the barcode axis to get the number of overlapping peaks per barcode.
        Order of barcodes is preserved (`self.barcodes`).
        """
        return self.matrix.sum(axis=0).tolist()[0]
