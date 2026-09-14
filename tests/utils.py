#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for tests
"""

import functools
import gzip
import json
import os
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from assertpy import assert_that

REGEN_GOLDEN_ENV_VAR = "CARMACK_REGEN_GOLDEN"

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"

# Owner-writable, world-executable, so a stub written into a folder on PATH is found and
# run the way any other command on PATH would be.
STUB_EXECUTABLE_MODE = 0o755

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


def gzip_bytes(text: str) -> bytes:
    """
    Compress text into the bytes of a single-member gzip file.

    The member header carries a fixed zero modification time, so the same text always
    compresses to the same bytes. Returning bytes rather than writing a file lets a test
    derive corrupted variants - truncated, bit-flipped, concatenated - by slicing the
    result, without a healthy file on disk that nothing reads.

    Args:
        text: Text to compress, encoded as UTF-8.

    Returns:
        The complete bytes of a valid single-member gzip file holding the text.
    """

    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode("UTF-8"))
    return buffer.getvalue()


def write_executable_stub(folder: str | Path, name: str, body: str) -> str:
    """
    Write an executable POSIX shell script into a folder and return its path.

    Used to stand a stub compressor in for gzip or pigz. Writing it under a chosen name
    inside a folder placed on PATH lets a test exercise command resolution, PATH lookup
    and process exit status end to end, rather than mocking the subprocess away.

    Args:
        folder: Directory to write the stub into. It must already exist.
        name: File name for the stub, which is the name PATH lookup will find.
        body: Shell commands making up the stub, without the shebang line.

    Returns:
        The path of the stub, as a str for direct use in a command list.
    """

    path = Path(folder) / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(STUB_EXECUTABLE_MODE)
    return str(path)


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


def assert_mqc_payload_file(path: str | Path, payload_id: str) -> dict[str, object]:
    """
    Assert that a file holds exactly one renderable MultiQC payload, and return it.

    MultiQC builds one section from one custom-content file, out of that file's own
    top-level ``id``, ``plot_type`` and ``data``. It never walks payloads nested under
    keys of carmack's own choosing, and loses them to a warning rather than an error,
    so those three top-level keys are what separate a file MultiQC renders from one it
    silently discards. Every stage asserts the same three keys, so the check lives here
    rather than once per stage's test module.

    Args:
        path: Path of the MultiQC payload file to check.
        payload_id: Section id the payload is expected to declare.

    Returns:
        The parsed payload.

    Raises:
        AssertionError: If the file is missing, or its payload lacks any of the three
            top-level keys, declares a different id, or carries empty top-level data.
    """

    payload_path = Path(path)

    assert_that(payload_path.exists()).described_as(
        f"missing MultiQC payload: {payload_path}"
    ).is_true()
    payload = json.loads(payload_path.read_text())
    assert_that(payload).contains_key("id", "plot_type", "data")
    assert_that(payload["id"]).is_equal_to(payload_id)
    assert_that(payload["data"]).is_not_empty()
    return payload


def assert_no_mqc_file_bundles_payloads(
    output_dir: str | Path, bundle_keys: Sequence[str]
) -> None:
    """
    Assert that every MultiQC file in a directory is one payload, never a bundle.

    Bundling costs every nested chart silently, so the emitted set is checked as a
    whole rather than file by file: a stage that regressed to writing one combined
    file would still satisfy that stage's per-file assertions. The bundle keys are
    the stage's own, since a bundle would nest its payloads under exactly the names
    that stage's builders are known by.

    Args:
        output_dir: Directory the run wrote its ``*_mqc.json`` files into.
        bundle_keys: Keys the stage would have nested its payloads under, none of
            which may appear at the top level of any emitted file.

    Raises:
        AssertionError: If the run emitted no MultiQC files at all, or any of them
            carries one of the bundle keys at its top level.
    """

    mqc_paths = sorted(Path(output_dir).glob("*_mqc.json"))

    assert_that(mqc_paths).is_not_empty()
    for path in mqc_paths:
        payload = json.loads(path.read_text())
        assert_that(payload).described_as(path.name).does_not_contain_key(*bundle_keys)
