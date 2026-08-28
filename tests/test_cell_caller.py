import matplotlib.pyplot as plt
import pytest
from scipy.io import mmread
from scipy.sparse import isspmatrix_csr, isspmatrix_lil

from carmack.cell_caller.cell_caller import CellCaller
from carmack.cell_caller.peak_barcode_matrix import PeakBarcodeMatrix

BED_PATH = "tests/data/atac_k562_peaks.sorted.bed"
BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1.dedup.tagged.bam"


class TestPeakBarcodeMatrix:
    @pytest.fixture(scope="class")
    def peak_barcode_matrix(self):
        peak_names = ("peak1", "peak2", "peak3")
        barcodes = {"barcode1", "barcode2"}

        obj = PeakBarcodeMatrix(peak_names, barcodes)
        return obj

    def test_initialisation(self, peak_barcode_matrix):
        assert peak_barcode_matrix.peaks == ("peak1", "peak2", "peak3")
        assert peak_barcode_matrix.barcodes == ("barcode1", "barcode2")
        assert isspmatrix_lil(peak_barcode_matrix.matrix)

    def test_invalid_operations(self, peak_barcode_matrix):
        # Initialisation with invalid elements
        peak_names = ("peak1", "peak2", "peak3", 1)
        barcodes = 1
        with pytest.raises(TypeError):
            # Invalid peak names
            PeakBarcodeMatrix(peak_names, {"barcode1", "barcode2"})
        with pytest.raises(TypeError):
            # Invalid barcodes
            PeakBarcodeMatrix(("peak1", "peak2", "peak3"), barcodes)

        # Incremenet value at invalid peak and barcode
        with pytest.raises(ValueError):
            # Invalid peak name
            peak_barcode_matrix.increment_index("peak4", "barcode1")
        with pytest.raises(ValueError):
            # Invalid barcode
            peak_barcode_matrix.increment_index("peak1", "barcode3")

        # Get value of non-existent peak and barcode
        with pytest.raises(ValueError):
            # Invalid peak name
            peak_barcode_matrix.get_value("peak4", "barcode1")
        with pytest.raises(ValueError):
            # Invalid barcode
            peak_barcode_matrix.get_value("peak1", "barcode3")

    def test_index_operations(self, peak_barcode_matrix):
        assert peak_barcode_matrix.get_value("peak1", "barcode1") == 0
        for _ in range(2):  # Increment same index value twice
            peak_barcode_matrix.increment_index("peak1", "barcode1")
        assert peak_barcode_matrix.get_value("peak1", "barcode1") == 2

    def test_filter(self, peak_barcode_matrix):
        filtered_matrix = peak_barcode_matrix.filter_matrix([0, 2], [0])
        assert isspmatrix_csr(filtered_matrix)
        assert filtered_matrix.shape == (2, 1)

        # Check invalid indices
        with pytest.raises(ValueError):
            peak_barcode_matrix.filter_matrix([0, 3], [0, 2])


class TestCellCaller:
    @pytest.fixture(scope="class")
    def cell_caller(self):
        # Compute matrix only once and reuse it for tests
        obj = CellCaller(bam=BAM_PATH, bed=BED_PATH)
        obj.compute_matrix()
        return obj

    def test_compute_matrix(self, cell_caller):
        # print(cell_caller.matrix.sum_barcodes())
        assert sum(cell_caller.matrix.sum_barcodes()) > 0  # Total overlaps > 0

    def test_plot(self, cell_caller):
        # Test different values of force_n (knee point, 200 cells and all cells)
        for n in (None, 200, 0):
            plot = cell_caller.make_plot(force_n=n)
            # plot.savefig(f"barcode_matrix_{str(n)}.png")
            assert isinstance(plot, plt.Figure)

    def test_export(self, cell_caller, tmp_path):
        cell_caller.export(output_dir=tmp_path, prefix="test")

        # Barcode file
        barcode_file = tmp_path / "test_barcodes.tsv"
        assert barcode_file.exists()

        with open(barcode_file, "r") as f:
            barcode_count = 0
            for line in f:
                if len(line.strip()) > 0:
                    barcode_count += 1

        # Peak file
        peak_file = tmp_path / "test_peaks.bed"
        assert peak_file.exists()

        with open(peak_file, "r") as f:
            peak_count = 0
            for line in f:
                if len(line.strip()) > 0:
                    peak_count += 1

        # Matrix file
        matrix_file = tmp_path / "test_matrix.mtx"
        assert matrix_file.exists()

        with open(matrix_file, "r") as f:
            matrix = mmread(f)
            assert matrix.shape == (peak_count, barcode_count)
