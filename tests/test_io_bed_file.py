import pytest

from carmack.io.bed_file import BedFile

BED_PATH = "tests/data/atac_k562_peaks.sorted.bed"

class TestBedFile:
    def test_read_file(self):
        bed = BedFile(BED_PATH)
        bed_reader = bed.open_read_iterator()
        entry = next(bed_reader)

        assert entry["chrom"].startswith("chr")
        assert isinstance(entry["start"], int)
        assert isinstance(entry["end"], int)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["score"], int)
        assert (0 <= entry["score"] <= 1000)
        assert entry["strand"] in ["+", "-", "."]

    def test_read_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bed = BedFile(tmp_path / "nonexistent.bed")
            bed_reader = bed.open_read_iterator()
            next(bed_reader)

    def test_write_file(self, tmp_path):
        # Write file
        bed_source = BedFile(BED_PATH).open_read_iterator()
        bed_writer = BedFile(tmp_path / "test.bed").open_write_stream()

        for _ in range(5):  # Only try with 5 entries
            entry = next(bed_source)
            BedFile.write_entry(bed_writer, *entry.values())

        bed_source.close()
        bed_writer.close()

        # Read new file and compare
        bed_source = BedFile(BED_PATH).open_read_iterator()
        bed_source_new = BedFile(tmp_path / "test.bed").open_read_iterator()
        for entry_new in bed_source_new:
            entry = next(bed_source)
            assert entry == entry_new

        bed_source.close()
        bed_source_new.close()


