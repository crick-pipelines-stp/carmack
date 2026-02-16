from pathlib import Path

import pytest

from carmack.utils import file_md5, get_bai, get_cpu_count, get_prefix, validate_file_md5


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
