#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the shared test helpers in tests/utils.py.

These cover the golden-comparison helpers rather than any production module: report
normalisation, golden assertion and regeneration, and the PNG sanity check.
"""

import gzip

import pytest
from assertpy import assert_that

from tests.utils import (
    PNG_MAGIC_BYTES,
    REGEN_GOLDEN_ENV_VAR,
    assert_is_png,
    assert_matches_golden,
    read_gzip_text,
    strip_report_run_details,
)

BARCODE_REPORT = (
    "# Carmack version: 0.0.0\n"
    "# Report generated at: 2026-01-01 00:00:00 UTC\n"
    "\n"
    "# Overall Barcode Extraction Stats\n"
    "Total reads: 200\n"
)

# The UMI report has no blank line between the volatile pair and its section header, so a
# naive "strip while the line starts with #" loop would silently eat a content line.
UMI_REPORT = (
    "# Carmack version: 0.0.0\n"
    "# Report generated at: 2026-01-01 00:00:00 UTC\n"
    "# UMI Extraction Stats\n"
    "Total reads: 200\n"
)


@pytest.fixture
def regen_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure golden regeneration is off so the asserting path is exercised.

    Args:
        monkeypatch: Fixture used to remove the regeneration variable.
    """
    monkeypatch.delenv(REGEN_GOLDEN_ENV_VAR, raising=False)


class TestReadGzipText:
    """Tests for read_gzip_text."""

    def test_round_trips_gzip_content(self, tmp_path) -> None:
        """Test that gzipped text is returned decompressed, for str and Path inputs."""
        path = tmp_path / "sample.txt.gz"
        with gzip.open(path, "wt") as handle:
            handle.write("line one\nline two\n")

        assert_that(read_gzip_text(path)).is_equal_to("line one\nline two\n")
        assert_that(read_gzip_text(str(path))).is_equal_to("line one\nline two\n")


class TestStripReportRunDetails:
    """Tests for strip_report_run_details."""

    def test_drops_volatile_lines_and_keeps_section_header(self) -> None:
        """Test that only the version and timestamp lines are removed."""
        result = strip_report_run_details(BARCODE_REPORT)

        assert_that(result).is_equal_to("\n# Overall Barcode Extraction Stats\nTotal reads: 200\n")

    def test_preserves_umi_header_without_blank_line(self) -> None:
        """Test that a section header abutting the volatile pair survives."""
        result = strip_report_run_details(UMI_REPORT)

        assert_that(result).is_equal_to("# UMI Extraction Stats\nTotal reads: 200\n")

    def test_keeps_lines_that_merely_mention_the_phrases(self) -> None:
        """Test that only lines starting with a volatile prefix are dropped."""
        text = (
            "# Notes on the # Carmack version: field\n"
            "Carmack version: 1.2.3\n"
            "# Report generated at: 2026-01-01 00:00:00 UTC\n"
        )

        result = strip_report_run_details(text)

        assert_that(result).is_equal_to(
            "# Notes on the # Carmack version: field\nCarmack version: 1.2.3\n"
        )

    def test_is_noop_without_volatile_lines(self) -> None:
        """Test that a report carrying no run details is returned unchanged."""
        text = "# UMI Extraction Stats\nTotal reads: 5\n"

        assert_that(strip_report_run_details(text)).is_equal_to(text)


class TestAssertMatchesGolden:
    """Tests for assert_matches_golden."""

    def test_passes_on_exact_match(self, tmp_path, regen_disabled: None) -> None:
        """Test that matching text does not raise."""
        golden = tmp_path / "golden.txt"
        golden.write_text("expected\n")

        assert_matches_golden("expected\n", golden)

    def test_raises_on_mismatch_naming_the_golden(self, tmp_path, regen_disabled: None) -> None:
        """Test that differing text raises and names the golden file."""
        golden = tmp_path / "golden.txt"
        golden.write_text("expected\n")

        with pytest.raises(AssertionError) as excinfo:
            assert_matches_golden("produced\n", golden)

        assert_that(str(excinfo.value)).contains("golden.txt")

    def test_raises_when_golden_missing_and_mentions_regeneration(
        self, tmp_path, regen_disabled: None
    ) -> None:
        """Test that a missing golden explains how to generate it."""
        with pytest.raises(AssertionError) as excinfo:
            assert_matches_golden("produced\n", tmp_path / "absent.txt")

        assert_that(str(excinfo.value)).contains(REGEN_GOLDEN_ENV_VAR)

    def test_regeneration_writes_golden_and_creates_parents(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that regeneration writes the produced text into a missing directory tree."""
        monkeypatch.setenv(REGEN_GOLDEN_ENV_VAR, "1")
        golden = tmp_path / "nested" / "dir" / "golden.txt"

        assert_matches_golden("produced\n", golden)

        assert_that(golden.read_text()).is_equal_to("produced\n")

    @pytest.mark.parametrize("value", ["0", ""])
    def test_asserts_when_regeneration_not_exactly_one(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Test that only the literal "1" enables regeneration."""
        monkeypatch.setenv(REGEN_GOLDEN_ENV_VAR, value)
        golden = tmp_path / "golden.txt"
        golden.write_text("expected\n")

        with pytest.raises(AssertionError):
            assert_matches_golden("produced\n", golden)

        assert_that(golden.read_text()).is_equal_to("expected\n")


class TestAssertIsPng:
    """Tests for assert_is_png."""

    def test_passes_on_png_magic_bytes(self, tmp_path) -> None:
        """Test that a file starting with the PNG signature is accepted."""
        png = tmp_path / "plot.png"
        png.write_bytes(PNG_MAGIC_BYTES + b"payload")

        assert_is_png(png)

    @pytest.mark.parametrize(
        "content",
        [None, b"", b"not a png at all", PNG_MAGIC_BYTES[:-1]],
        ids=["missing", "empty", "plain-text", "truncated-magic"],
    )
    def test_raises_for_anything_but_a_png(self, tmp_path, content: bytes | None) -> None:
        """Test that missing, empty and non-PNG files are rejected."""
        png = tmp_path / "plot.png"
        if content is not None:
            png.write_bytes(content)

        with pytest.raises(AssertionError):
            assert_is_png(png)
