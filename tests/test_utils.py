from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.utils import (
    file_md5,
    format_duration,
    get_bai,
    get_cpu_count,
    get_prefix,
    homopolymer_run_length,
    validate_file_md5,
)


class TestUtils:
    # ===== Tests for get_prefix =====
    def test_get_prefix_str(self):
        assert get_prefix("path/to/file.fastq.gz") == "file"
        assert get_prefix("file.fastq") == "file"
        assert get_prefix("file") == "file"

    def test_get_prefix_path(self):
        assert get_prefix(Path("path/to/file.fastq.gz")) == "file"
        assert get_prefix(Path("file.fastq")) == "file"
        assert get_prefix(Path("file")) == "file"

    # ===== Tests for get_cpu_count =====
    def test_get_cpu_count(self):
        count = get_cpu_count()
        assert isinstance(count, int)
        assert count > 0

    def test_get_cpu_count_reserve(self):
        count = get_cpu_count(reserve=2)
        assert isinstance(count, int)
        assert count > 0

    # ===== Tests for get_bai =====
    def test_get_bai_exists(self, tmp_path):
        bam_file = tmp_path / "file.bam"
        bai_file = tmp_path / "file.bam.bai"
        bam_file.touch()
        bai_file.touch()
        assert get_bai(str(bam_file)) == str(bai_file)

    # ===== Tests for checksum utils =====
    @pytest.fixture
    def sample_file(self, tmp_path):
        file = tmp_path / "test.txt"
        content = "Hello, world!"
        file.write_text(content)
        return file

    def test_file_md5(self, sample_file):
        expected_md5 = "6cd3556deb0da54bca060b4c39479839"
        assert file_md5(str(sample_file)) == expected_md5

    def test_validate_file_md5_valid(self, sample_file):
        expected_md5 = "6cd3556deb0da54bca060b4c39479839"
        assert validate_file_md5(str(sample_file), expected_md5) is True

    # ===== Tests for format_duration =====
    def test_format_duration_seconds_only(self):
        assert format_duration(45) == "45s"
        assert format_duration(0) == "0s"
        assert format_duration(59) == "59s"

    def test_format_duration_minutes_and_seconds(self):
        assert format_duration(60) == "1m"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"

    def test_format_duration_hours_minutes_seconds(self):
        assert format_duration(3600) == "1h"
        assert format_duration(3661) == "1h 1m 1s"
        assert format_duration(7265) == "2h 1m 5s"

    def test_format_duration_float(self):
        assert format_duration(45.7) == "45s"
        assert format_duration(90.123) == "1m 30s"


class TestHomopolymerRunLength:
    """Tests for homopolymer_run_length.

    The scan counts consecutive copies of an anchor base from a start index onward.
    Observed anchor runs in real libraries span three to eighteen bases and peak
    around six, so the function must report the run it actually sees rather than
    clamping at any fixed ceiling.
    """

    @pytest.mark.parametrize("run_length", [3, 4, 5, 6, 7, 8])
    def test_homopolymer_run_length_run_before_other_base_returns_run_length(
        self, run_length: int
    ) -> None:
        """A run terminated by a non-anchor base reports exactly the run length."""
        prefix = "ACTCAC"
        seq = f"{prefix}{'G' * run_length}TTTT"

        result = homopolymer_run_length(seq, len(prefix), "G")

        assert_that(result).is_equal_to(run_length)

    def test_homopolymer_run_length_run_reaching_read_end_counts_to_end(self) -> None:
        """A run that continues to the final base counts to the end without overrunning."""
        prefix = "ACTCAC"
        seq = f"{prefix}{'G' * 5}"

        result = homopolymer_run_length(seq, len(prefix), "G")

        assert_that(result).is_equal_to(5)
        assert_that(len(prefix) + result).is_equal_to(len(seq))

    def test_homopolymer_run_length_start_on_other_base_returns_zero(self) -> None:
        """A start index sitting on a non-anchor base reports no run."""
        seq = "ACTCACTTTTGGGGGG"

        result = homopolymer_run_length(seq, 6, "G")

        assert_that(result).is_equal_to(0)

    def test_homopolymer_run_length_start_past_end_returns_zero(self) -> None:
        """A start index one past the final base reports no run instead of raising."""
        seq = "ACTCACGGGGGG"

        result = homopolymer_run_length(seq, len(seq), "G")

        assert_that(result).is_equal_to(0)

    def test_homopolymer_run_length_non_g_base_counts_only_that_base(self) -> None:
        """The anchor base is taken from the argument, so a downstream G ends an A run."""
        seq = "CTCTAAAAAGGGG"

        result = homopolymer_run_length(seq, 4, "A")

        assert_that(result).is_equal_to(5)

    def test_homopolymer_run_length_run_longer_than_eight_returns_full_length(self) -> None:
        """A long run from the observed tail reports its full length, uncapped."""
        prefix = "ACTCAC"
        seq = f"{prefix}{'G' * 18}TTTT"

        result = homopolymer_run_length(seq, len(prefix), "G")

        assert_that(result).is_equal_to(18)
