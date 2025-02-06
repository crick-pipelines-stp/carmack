import pytest
import pandas as pd
import matplotlib.pyplot as plt

from carmack.call_cells.call_cells import CellCaller

BED_PATH = "tests/data/atac_k562_peaks.sorted.bed"
BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1_dup.dedup.tagged.bam"  # Small data, no usable plot
# BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1.dedup.tagged.bam"    # Full data

class TestCellCaller:
    @pytest.fixture(scope="class")
    def cell_caller(self):
        obj = CellCaller(bam=BAM_PATH, bed=BED_PATH)
        obj.compute_matrix()
        return obj

    def test_compute_matrix(self, cell_caller):
        assert isinstance(cell_caller.matrix, pd.DataFrame)
        # print(cell_caller.matrix.sum())
        assert sum(cell_caller.matrix.sum()) > 0    # At least one overlap

    def test_plot(self, cell_caller):
        plot = cell_caller.make_plot()
        # plot.savefig("barcode_matrix.png")
        assert isinstance(plot, plt.Figure)
