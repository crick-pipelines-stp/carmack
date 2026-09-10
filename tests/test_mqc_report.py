"""Tests for the shared MultiQC report module.

``carmack.mqc_report`` carries the two identifiers every MultiQC custom
content section needs to attach itself to the Carmack parent module
(``CARMACK_PARENT_ID`` and ``CARMACK_PARENT_NAME``), plus ``write_mqc_json``,
the one place a caller writes an MQC-readable JSON payload to disk. These
tests pin the two constants' exact values, since a MultiQC config keys off
them literally, and pin ``write_mqc_json`` on the property that matters to a
downstream MultiQC parse: a payload written out and read back with
``json.load`` reproduces exactly what was passed in, for both a flat mapping
and one nesting a mapping inside it, the shape the generalstats and bargraph
payloads this module serializes elsewhere will actually take.
"""

import json
from pathlib import Path

from assertpy import assert_that

from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME, write_mqc_json

FLAT_PAYLOAD = {"sample": "SK588", "total_reads": 12345, "matched_fraction": 0.875}

NESTED_PAYLOAD = {
    "id": "carmack_prepare_reads",
    "data": {
        "SK588": {"total_reads": 12345, "unmatched_written": 4321},
        "SK661": {"total_reads": 6789, "unmatched_written": 1234},
    },
    "config": {"title": "Prepare Reads", "sort_rows": False},
}


class TestCarmackParentConstants:
    """The two module-level identifiers a MultiQC config keys off literally."""

    def test_carmack_parent_id_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_ID is exactly the lowercase module id."""
        assert_that(CARMACK_PARENT_ID).is_equal_to("carmack")

    def test_carmack_parent_name_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_NAME is exactly the capitalised display name."""
        assert_that(CARMACK_PARENT_NAME).is_equal_to("Carmack")


class TestWriteMqcJsonRoundTrip:
    """``write_mqc_json``: round-tripping a payload through disk via json.load."""

    def test_write_mqc_json_round_trips_a_flat_payload(self, tmp_path: Path) -> None:
        """Test that a flat dict payload reads back exactly as it was written."""
        output_path = tmp_path / "flat_mqc.json"

        write_mqc_json(output_path, FLAT_PAYLOAD)

        with open(output_path) as handle:
            loaded = json.load(handle)
        assert_that(loaded).is_equal_to(FLAT_PAYLOAD)

    def test_write_mqc_json_round_trips_a_nested_payload(self, tmp_path: Path) -> None:
        """Test that a dict nesting further dicts reads back exactly as it was written.

        This mirrors the generalstats and bargraph payload shapes the module
        serializes elsewhere, where the top-level mapping's values are
        themselves mappings.
        """
        output_path = tmp_path / "nested_mqc.json"

        write_mqc_json(output_path, NESTED_PAYLOAD)

        with open(output_path) as handle:
            loaded = json.load(handle)
        assert_that(loaded).is_equal_to(NESTED_PAYLOAD)

    def test_write_mqc_json_creates_a_readable_file_at_the_given_path(
        self, tmp_path: Path
    ) -> None:
        """Test that the file is written at exactly the path given, not some other location."""
        output_path = tmp_path / "subdir_check" / "report_mqc.json"
        output_path.parent.mkdir()

        write_mqc_json(output_path, FLAT_PAYLOAD)

        assert_that(output_path.exists()).is_true()
