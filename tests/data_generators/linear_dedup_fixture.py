"""Generator for the linear-dedup fixture BAMs.

This is a one-off script: run it to (re)produce ``tests/data/linear_dedup_fixture.bam`` and
``tests/data/linear_dedup_fixture_missing_cb.bam`` (each alongside its ``.bai`` index) from the
record specs pinned in ``tests/test_linear_dedup.py`` -- ``FIXTURE_RECORDS`` and
``MISSING_CB_RECORDS``. Those lists are the single source of truth for every alignment's QNAME,
tags, flags, position and mate relationship; this script only turns each entry into a
``pysam.AlignedSegment``, writes an unsorted BAM, coordinate-sorts it and indexes it, so the two
never drift apart.

Run from the repository root:

    python tests/data_generators/linear_dedup_fixture.py
"""

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pysam

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_MODULE_PATH = Path(__file__).resolve().parent.parent / "test_linear_dedup.py"

DEFAULT_UNMAPPED_READ_LENGTH = 50
MAPPED_MAPPING_QUALITY = 60
UNMAPPED_MAPPING_QUALITY = 0


def load_fixture_spec() -> ModuleType:
    """Load tests/test_linear_dedup.py as a module without importing it as a package member."""
    spec = importlib.util.spec_from_file_location("linear_dedup_fixture_spec", TEST_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_header(references: dict) -> pysam.AlignmentHeader:
    """Build an unsorted-order BAM header carrying the fixture's reference sequences."""
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "unsorted"},
            "SQ": [{"SN": chrom, "LN": length} for chrom, length in references.items()],
        }
    )


def derive_placeholder_sequence(record: dict, length: int) -> str:
    """Derive a deterministic, arbitrary placeholder base sequence of the given length."""
    digest = hashlib.sha256(
        f"{record['qname']}:{record['mate']}:{record['category']}".encode()
    ).digest()
    bases = "ACGT"
    return "".join(bases[digest[position % len(digest)] % 4] for position in range(length))


def find_primary_mate(records: list, record: dict) -> dict:
    """Find the primary record sharing record's QNAME but the opposite mate number."""
    target_mate = 2 if record["mate"] == 1 else 1
    for candidate in records:
        if (
            candidate["qname"] == record["qname"]
            and candidate["mate"] == target_mate
            and candidate["category"] == "primary"
        ):
            return candidate
    raise ValueError(f"No primary mate found for {record['qname']} mate {record['mate']}")


def build_segment(header: pysam.AlignmentHeader, record: dict, mate: dict) -> pysam.AlignedSegment:
    """Build one pysam.AlignedSegment matching a FIXTURE_RECORDS/MISSING_CB_RECORDS entry."""
    segment = pysam.AlignedSegment(header)
    segment.query_name = record["qname"]
    segment.is_paired = True
    segment.is_read1 = record["mate"] == 1
    segment.is_read2 = record["mate"] == 2
    segment.is_secondary = record["category"] == "secondary"
    segment.is_supplementary = record["category"] == "supplementary"
    segment.is_reverse = record["is_reverse"]
    segment.is_unmapped = record["unmapped"]
    segment.mate_is_unmapped = record["mate_unmapped"]

    read_length = (
        record["ref_len"] if record["ref_len"] is not None else DEFAULT_UNMAPPED_READ_LENGTH
    )
    segment.query_sequence = derive_placeholder_sequence(record, read_length)
    segment.query_qualities = pysam.qualitystring_to_array("I" * read_length)

    segment.reference_id = header.get_tid(record["chrom"])
    segment.reference_start = record["ref_start"]
    if record["ref_len"] is not None:
        segment.cigartuples = [(0, record["ref_len"])]
    segment.mapping_quality = (
        UNMAPPED_MAPPING_QUALITY if record["unmapped"] else MAPPED_MAPPING_QUALITY
    )

    segment.next_reference_id = header.get_tid(mate["chrom"])
    segment.next_reference_start = mate["ref_start"]
    segment.mate_is_reverse = mate["is_reverse"]

    tags = []
    if record["cb"] is not None:
        tags.append(("CB", record["cb"], "Z"))
    if record["as_tag"] is not None:
        tags.append(("AS", record["as_tag"], "i"))
    segment.set_tags(tags)

    return segment


def write_unsorted_bam(records: list, header: pysam.AlignmentHeader, unsorted_path: Path) -> None:
    """Write every record as a pysam.AlignedSegment into an unsorted BAM."""
    with pysam.AlignmentFile(str(unsorted_path), "wb", header=header) as bam:
        for record in records:
            mate = find_primary_mate(records, record)
            bam.write(build_segment(header, record, mate))


def sort_and_index(unsorted_path: Path, sorted_path: Path) -> None:
    """Coordinate-sort an unsorted BAM, index it, and remove the unsorted intermediate."""
    pysam.sort("-o", str(sorted_path), str(unsorted_path))
    pysam.index(str(sorted_path))
    unsorted_path.unlink()


def build_fixture(records: list, references: dict, sorted_path: Path) -> None:
    """Build one indexed, coordinate-sorted fixture BAM from a list of record specs."""
    header = build_header(references)
    unsorted_path = sorted_path.with_suffix(".unsorted.bam")
    write_unsorted_bam(records, header, unsorted_path)
    sort_and_index(unsorted_path, sorted_path)


def main() -> None:
    """Generate both linear-dedup fixture BAMs from the specs pinned in tests/test_linear_dedup.py."""
    spec = load_fixture_spec()
    build_fixture(spec.FIXTURE_RECORDS, spec.REFERENCES, DATA_DIR / "linear_dedup_fixture.bam")
    build_fixture(
        spec.MISSING_CB_RECORDS,
        spec.REFERENCES,
        DATA_DIR / "linear_dedup_fixture_missing_cb.bam",
    )


if __name__ == "__main__":
    main()
