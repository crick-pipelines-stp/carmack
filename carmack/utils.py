"""
Common utility functions for Carmack.
"""

import hashlib
import io
import logging
from os import cpu_count, path


log = logging.getLogger(__name__)


def file_md5(fname: str):
    """Calculates the md5sum for a file on the disk.

    Args:
        fname (str): Path to a local file.
    """

    # Calculate the md5 for the file on disk
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(io.DEFAULT_BUFFER_SIZE), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def validate_file_md5(file_name: str, expected_md5hex: str):
    """Validates the md5 checksum of a file on disk.

    Args:
        file_name (str): Path to a local file.
        expected (str): The expected md5sum.

    Raises:
        IOError, if the md5sum does not match the remote sum.
    """
    # Make sure the expected md5 sum is a hexdigest
    try:
        int(expected_md5hex, 16)
    except ValueError as ex:
        raise ValueError(
            f"The supplied md5 sum must be a hexdigest but it is {expected_md5hex}"
        ) from ex

    file_md5hex = file_md5(file_name)

    if file_md5hex.upper() != expected_md5hex.upper():
        raise IOError(f"{file_name} md5 does not match remote: {expected_md5hex} - {file_md5hex}")

    return True


def get_prefix(file_name: str) -> str:
    """
    Extracts the prefix from a file name.
    """
    return file_name.rsplit("/", 1)[-1].split(".", 1)[0]


def get_bai(bam_file: str) -> str:
    """
    Try to get the corresponding BAI file for a BAM file. Looks for a file with the same name as
    the BAM file but with a .bai extension.

    Returns None if the BAI file does not exist.
    """
    bai_file = f"{bam_file}.bai"
    if not path.exists(bai_file):
        raise FileNotFoundError(
            f"Could not find BAI file for {bam_file} - Try specifying it manually."
        )
    return bai_file


def get_cpu_count(reserve: int = 1) -> int:
    """
    Get the number of CPUs available on the system minus the reserved amount.
    """
    return max(1, cpu_count() - reserve)
