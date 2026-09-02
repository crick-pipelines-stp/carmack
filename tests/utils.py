#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for tests
"""

import functools
import gzip
import os
import tempfile
from pathlib import Path

REGEN_GOLDEN_ENV_VAR = "CARMACK_REGEN_GOLDEN"

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"

# Run-detail lines vary between runs, so golden comparisons drop them. Matching on the full
# comment prefix keeps section headers, which are comments too, and lines that merely mention
# one of these phrases.
VOLATILE_REPORT_PREFIXES = ("# Carmack version:", "# Report generated at:")


def with_temporary_folder(func):
    """
    Call the decorated funtion under the tempfile.TemporaryDirectory
    context manager. Pass the temporary directory name to the decorated
    function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            return func(*args, tmpdirname, **kwargs)

    return wrapper


def with_temporary_file(func):
    """
    Call the decorated funtion under the tempfile.NamedTemporaryFile
    context manager. Pass the opened file handle to the decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tempfile.NamedTemporaryFile() as tmpfile:
            return func(*args, tmpfile, **kwargs)

    return wrapper


def read_gzip_text(path: str | Path) -> str:
    """
    Decompress a gzip file and return its contents as text.

    Args:
        path: Path to the gzip file, as either a str or a pathlib.Path.

    Returns:
        The decompressed contents of the file, empty for an empty gzip member.
    """

    with gzip.open(path, "rt") as handle:
        return handle.read()


def strip_report_run_details(text: str) -> str:
    """
    Remove the volatile run-detail lines from a stats report.

    Only lines whose content begins with one of the volatile prefixes are dropped, so
    section headers are preserved even when they immediately follow the run details with
    no intervening blank line. A line that merely contains a volatile phrase is kept.

    Args:
        text: Full report text, including the run-detail header lines.

    Returns:
        The report text with the volatile run-detail lines removed. Every other line,
        blank lines included, is returned unchanged.
    """

    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith(VOLATILE_REPORT_PREFIXES)
    ]
    return "".join(kept)


def assert_matches_golden(produced_text: str, golden_path: str | Path) -> None:
    """
    Assert that produced text matches a golden file, or regenerate that file on request.

    Setting the ``CARMACK_REGEN_GOLDEN`` environment variable to exactly ``"1"`` writes the
    produced text to the golden path, creating parent directories as needed, and skips the
    comparison. Any other value, including an unset variable, compares as normal.

    Args:
        produced_text: Text produced by the code under test.
        golden_path: Path to the golden file holding the expected text.

    Raises:
        AssertionError: If the golden file is missing, or its contents differ from the
            produced text, while regeneration is not enabled.
    """

    golden = Path(golden_path)

    if os.environ.get(REGEN_GOLDEN_ENV_VAR) == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(produced_text)
        return

    assert golden.is_file(), (
        f"Golden file {golden} does not exist. "
        f"Re-run with {REGEN_GOLDEN_ENV_VAR}=1 to generate it."
    )

    expected = golden.read_text()
    assert produced_text == expected, (
        f"Produced text does not match golden file {golden}. "
        f"Re-run with {REGEN_GOLDEN_ENV_VAR}=1 to update it.\n"
        f"--- expected ---\n{expected}\n--- produced ---\n{produced_text}"
    )


def assert_is_png(path: str | Path) -> None:
    """
    Assert that a file exists, is non-empty and starts with the PNG signature.

    Args:
        path: Path to the file expected to be a PNG image.

    Raises:
        AssertionError: If the file is missing, empty, or does not begin with the PNG
            magic bytes.
    """

    png = Path(path)

    assert png.is_file(), f"Expected a PNG file at {png}, but no file exists there."

    with png.open("rb") as handle:
        header = handle.read(len(PNG_MAGIC_BYTES))

    assert header, f"Expected a PNG file at {png}, but the file is empty."
    assert header == PNG_MAGIC_BYTES, (
        f"Expected a PNG file at {png}, but it starts with {header!r} "
        f"rather than the PNG signature {PNG_MAGIC_BYTES!r}."
    )
