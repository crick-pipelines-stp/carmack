"""Tests for the on-disk UMI shard store.

Extraction spills each accepted UMI record to one of ``S`` temporary shard files
chosen by a stable digest of its cell barcode, so correction can run one shard at
a time instead of holding every record of the run in memory. These tests pin the
properties the design rests on: the shard digest is stable across processes and
hash seeds, a whole cell barcode always lands in exactly one shard, each shard is
written in ascending extraction order, a shard reads back whole without anything
outside the store having flushed it first, the temporary tree is always torn
down, and the k-way merge of the per-shard map files restores exact extraction
order byte for byte.
"""

import errno
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.umi.umi_corrector import CorrectedUmi, UmiRecord
from carmack.umi.umi_shards import DEFAULT_SHARD_COUNT, UmiShardStore, shard_index

REPO_ROOT = Path(__file__).resolve().parent.parent

# Twelve cell barcodes whose crc32 digests spread across all four shards used by
# the store tests, so no test can pass on a single accidentally populated shard.
BARCODES = [f"{'ACGT' * 2}{index:02d}{'C' * 10}" for index in range(12)]

# Run in a child interpreter to prove the shard digest does not depend on the
# per-process randomisation that PYTHONHASHSEED controls; the built-in hash of
# the same barcode is reported alongside to show that randomisation is real.
SHARD_INDEX_PROBE = """
import json
import sys

from carmack.umi.umi_shards import shard_index

barcodes = json.loads(sys.argv[1])
print(
    json.dumps(
        {
            "shards": [shard_index(barcode, 256) for barcode in barcodes],
            "builtin_hash": hash(barcodes[0]),
        }
    )
)
"""


def probe_shard_indexes(barcodes: list[str], hash_seed: str) -> dict:
    """Compute shard indexes in a fresh interpreter running under a given hash seed.

    Args:
        barcodes: The cell barcodes to shard in the child interpreter.
        hash_seed: The ``PYTHONHASHSEED`` value the child interpreter runs under.

    Returns:
        The decoded probe payload: the shard index per barcode under ``shards``
        and the child's built-in ``hash()`` of the first barcode under
        ``builtin_hash``.
    """
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", SHARD_INDEX_PROBE, json.dumps(barcodes)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
    )
    assert_that(result.returncode).described_as(result.stderr).is_zero()
    return json.loads(result.stdout)


def numbered(ordinal: int, read_id: str, barcode: str, raw_umi: str) -> tuple[int, UmiRecord]:
    """Build an ``(ordinal, UmiRecord)`` pair in the shape ``read_shard`` returns.

    Args:
        ordinal: The read's position in extraction order.
        read_id: Identifier of the read the UMI was extracted from.
        barcode: The full cell barcode.
        raw_umi: The faithful raw UMI sequence.

    Returns:
        The ordinal paired with the assembled record.
    """
    return ordinal, UmiRecord(read_id=read_id, barcode=barcode, raw_umi=raw_umi)


def write_all_shard_maps(
    store: UmiShardStore,
    rows_by_shard: dict[int, list[tuple[int, UmiRecord]]],
    mapping: dict[str, CorrectedUmi],
) -> None:
    """Write a map file for every shard, empty where that shard holds no rows.

    Args:
        store: The store to write the per-shard map files into.
        rows_by_shard: The ``(ordinal, record)`` rows to write, keyed by shard.
        mapping: The corrected-UMI mapping the map rows are rendered from.
    """
    for index in store.indexes:
        store.write_shard_map(index, rows_by_shard.get(index, []), mapping)


class TestShardIndex:
    """The stable barcode digest that decides which shard a record is spilled to."""

    @pytest.mark.parametrize("shard_count", [1, 4, DEFAULT_SHARD_COUNT])
    def test_shard_index_always_within_range(self, shard_count: int) -> None:
        indexes = [shard_index(barcode, shard_count) for barcode in BARCODES]
        assert_that(indexes).is_not_empty()
        for index in indexes:
            assert_that(index).is_in(*range(shard_count))

    def test_shard_index_is_stable_across_hash_seeds(self) -> None:
        in_process = [shard_index(barcode, 256) for barcode in BARCODES]
        zero_seed = probe_shard_indexes(BARCODES, "0")
        other_seed = probe_shard_indexes(BARCODES, "999")

        assert_that(zero_seed["shards"]).is_equal_to(in_process)
        assert_that(other_seed["shards"]).is_equal_to(in_process)
        # The built-in hash of the same string differs between the two seeds,
        # which is precisely why the digest cannot be built on hash().
        assert_that(other_seed["builtin_hash"]).is_not_equal_to(zero_seed["builtin_hash"])

    @pytest.mark.parametrize(
        "barcode,shard_count,expected",
        [
            ("ACGTACGTAC", 256, 172),
            ("AAAAAAAAAA", 256, 207),
            ("ACGT-TTTT-GGGG", 4, 2),
        ],
    )
    def test_shard_index_pins_expected_values(
        self, barcode: str, shard_count: int, expected: int
    ) -> None:
        assert_that(shard_index(barcode, shard_count)).is_equal_to(expected)

    def test_same_barcode_always_maps_to_the_same_shard(self) -> None:
        for barcode in BARCODES:
            repeats = {shard_index(barcode, 8) for _ in range(20)}
            assert_that(repeats).is_length(1)


class TestUmiShardStore:
    """Spilling records to temporary shard files and reading them back."""

    def test_records_round_trip_through_a_shard(self, tmp_path) -> None:
        records = [
            UmiRecord(read_id=f"r{index:02d}", barcode=barcode, raw_umi="ACGTACGT")
            for index, barcode in enumerate(BARCODES)
        ]
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            for ordinal, record in enumerate(records):
                store.write(ordinal, record)
            recovered = [pair for index in store.indexes for pair in store.read_shard(index)]

        assert_that(recovered).is_length(len(records))
        by_ordinal = dict(recovered)
        for ordinal, record in enumerate(records):
            assert_that(by_ordinal[ordinal].read_id).is_equal_to(record.read_id)
            assert_that(by_ordinal[ordinal].barcode).is_equal_to(record.barcode)
            assert_that(by_ordinal[ordinal].raw_umi).is_equal_to(record.raw_umi)

    def test_whole_barcode_group_lands_in_one_shard(self, tmp_path) -> None:
        ordinal = 0
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            for barcode in BARCODES:
                for replicate in range(5):
                    record = UmiRecord(
                        read_id=f"{barcode}-{replicate}", barcode=barcode, raw_umi="ACGTACGT"
                    )
                    store.write(ordinal, record)
                    ordinal += 1
            shards_per_barcode: dict[str, set[int]] = {barcode: set() for barcode in BARCODES}
            populated = 0
            for index in store.indexes:
                contents = store.read_shard(index)
                populated += 1 if contents else 0
                for _, record in contents:
                    shards_per_barcode[record.barcode].add(index)

        assert_that(populated).is_greater_than(1)
        for barcode, indexes in shards_per_barcode.items():
            assert_that(indexes).described_as(barcode).is_length(1)

    def test_read_shard_returns_ascending_ordinals(self, tmp_path) -> None:
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            for ordinal in range(10):
                store.write(
                    ordinal, UmiRecord(read_id=f"r{ordinal}", barcode=barcode, raw_umi="ACGTACGT")
                )
            ordinals = [ordinal for ordinal, _ in store.read_shard(target)]

        assert_that(ordinals).is_equal_to(sorted(ordinals))
        assert_that(ordinals).is_equal_to(list(range(10)))

    def test_files_are_created_under_the_supplied_temp_dir(self, tmp_path) -> None:
        temp_root = tmp_path / "spill"
        temp_root.mkdir()
        with UmiShardStore(shard_count=4, temp_dir=str(temp_root)) as store:
            store.write(0, UmiRecord(read_id="r0", barcode=BARCODES[0], raw_umi="ACGTACGT"))
            files = [path for path in temp_root.rglob("*") if path.is_file()]

        assert_that(files).is_not_empty()

    def test_temp_directory_is_removed_on_normal_exit(self, tmp_path) -> None:
        temp_root = tmp_path / "spill"
        temp_root.mkdir()
        with UmiShardStore(shard_count=4, temp_dir=str(temp_root)) as store:
            store.write(0, UmiRecord(read_id="r0", barcode=BARCODES[0], raw_umi="ACGTACGT"))
            assert_that(list(temp_root.iterdir())).is_not_empty()

        assert_that(list(temp_root.iterdir())).is_empty()

    def test_temp_directory_is_removed_when_an_exception_escapes(self, tmp_path) -> None:
        temp_root = tmp_path / "spill"
        temp_root.mkdir()
        with pytest.raises(RuntimeError, match="correction failed"):
            with UmiShardStore(shard_count=4, temp_dir=str(temp_root)) as store:
                store.write(0, UmiRecord(read_id="r0", barcode=BARCODES[0], raw_umi="ACGTACGT"))
                raise RuntimeError("correction failed")

        assert_that(list(temp_root.iterdir())).is_empty()

    def test_a_failed_construction_leaves_no_temporary_tree(self, tmp_path) -> None:
        # The store opens one write handle per shard, so a large shard count on a
        # small descriptor budget runs out of handles part-way through building
        # the tree. The caller never receives the object and so can never close
        # it: the constructor has to unwind what it built on its own.
        temp_root = tmp_path / "spill"
        temp_root.mkdir()
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        probe_path = temp_root / "descriptor-probe"
        # open() hands back the lowest free descriptor, so the probe's own number
        # is a count of what this process already holds. Budgeting a few on top
        # of it leaves room for the tree and a handful of shard handles before
        # the limit bites, which is what makes the construction a partial one.
        with probe_path.open("w") as probe:
            budget = probe.fileno() + 16
        probe_path.unlink()

        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (budget, hard))
            with pytest.raises(OSError) as failure:
                UmiShardStore(shard_count=budget * 4, temp_dir=str(temp_root))
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

        assert_that(failure.value.errno).is_equal_to(errno.EMFILE)
        assert_that(list(temp_root.glob("carmack-umi-*"))).is_empty()

    @pytest.mark.parametrize("shard_count", [0, -1])
    def test_non_positive_shard_count_raises(self, tmp_path, shard_count: int) -> None:
        with pytest.raises(ValueError):
            UmiShardStore(shard_count=shard_count, temp_dir=str(tmp_path))

    def test_discard_shard_drops_the_data_file_and_keeps_the_map(self, tmp_path) -> None:
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        record = UmiRecord(read_id="r0", barcode=barcode, raw_umi="ACGTACGT")
        mapping = {"r0": CorrectedUmi(barcode=barcode, ur="ACGTACGT", ub="ACGTACGT")}
        temp_root = tmp_path / "spill"
        temp_root.mkdir()
        destination = tmp_path / "umi_map.tsv"
        with UmiShardStore(shard_count=4, temp_dir=str(temp_root)) as store:
            store.write(0, record)
            write_all_shard_maps(store, {target: [(0, record)]}, mapping)
            before = {path for path in temp_root.rglob("*") if path.is_file()}

            store.discard_shard(target)

            after = {path for path in temp_root.rglob("*") if path.is_file()}
            # The map file survives, so the discarded shard still merges.
            store.merge_shard_maps(destination)

        assert_that(after).is_length(len(before) - 1)
        assert_that(destination.read_text()).is_equal_to(f"r0\t{barcode}\tACGTACGT\tACGTACGT\n")

    def test_empty_shard_reads_back_as_an_empty_list(self, tmp_path) -> None:
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            store.write(0, UmiRecord(read_id="r0", barcode=barcode, raw_umi="ACGTACGT"))
            empties = [index for index in store.indexes if index != target]
            contents = {index: store.read_shard(index) for index in empties}

        for index, shard in contents.items():
            assert_that(shard).described_as(f"shard {index}").is_empty()

    def test_read_shard_recovers_records_still_held_in_the_write_buffer(self, tmp_path) -> None:
        # Two rows are a fraction of the write handle's text buffer, so not one
        # byte of the shard has reached the file yet. Reading the shard has to be
        # what makes it whole; nothing else in the run will.
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        records = [
            UmiRecord(read_id=f"r{ordinal}", barcode=barcode, raw_umi="ACGTACGT")
            for ordinal in range(2)
        ]
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            for ordinal, record in enumerate(records):
                store.write(ordinal, record)
            recovered = store.read_shard(target)

        assert_that(recovered).is_equal_to([(0, records[0]), (1, records[1])])

    def test_read_shard_recovers_every_record_across_many_buffer_fills(self, tmp_path) -> None:
        # A row is around forty bytes, so a thousand of them overrun the write
        # handle's buffer several times over: earlier rows have reached the file
        # while the tail is still buffered. A shard read that misses the buffered
        # tail therefore returns a plausible prefix rather than nothing at all,
        # which is how the loss stays silent.
        record_count = 1000
        records = [
            UmiRecord(read_id=f"r{ordinal:04d}", barcode=BARCODES[0], raw_umi="ACGTACGT")
            for ordinal in range(record_count)
        ]
        with UmiShardStore(shard_count=1, temp_dir=str(tmp_path)) as store:
            for ordinal, record in enumerate(records):
                store.write(ordinal, record)
            recovered = store.read_shard(0)

        assert_that(recovered).is_length(record_count)
        assert_that([ordinal for ordinal, _ in recovered]).is_equal_to(list(range(record_count)))
        assert_that([record.read_id for _, record in recovered]).is_equal_to(
            [record.read_id for record in records]
        )

    def test_writing_to_a_shard_after_it_is_read_raises(self, tmp_path) -> None:
        # Reading a shard hands its descriptor back, so a record written to that
        # shard afterwards would land in a file nothing reads again. Giving the
        # handle up rather than merely flushing it is what turns that silent loss
        # into a raised error.
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        record = UmiRecord(read_id="r0", barcode=barcode, raw_umi="ACGTACGT")
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            store.write(0, record)
            store.read_shard(target)

            with pytest.raises(ValueError, match="closed file"):
                store.write(1, record)

    def test_a_shard_can_be_read_more_than_once(self, tmp_path) -> None:
        # Correction reads each shard once, but a second read of the same shard
        # must still return the same records rather than tripping over the handle
        # the first read gave up.
        barcode = BARCODES[0]
        target = shard_index(barcode, 4)
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            store.write(0, UmiRecord(read_id="r0", barcode=barcode, raw_umi="ACGTACGT"))
            first = store.read_shard(target)
            second = store.read_shard(target)

        assert_that(first).is_not_empty()
        assert_that(second).is_equal_to(first)


class TestMergeShardMaps:
    """The k-way merge that restores extraction order across the per-shard maps."""

    def test_merge_orders_rows_numerically_not_lexicographically(self, tmp_path) -> None:
        # 1, 2, 10, 20, 100 split across shards: merged as text these would come
        # back as 1, 10, 100, 2, 20, so the ordinal has to be compared as an int.
        rows_by_shard = {
            0: [numbered(1, "r1", "BC-A", "AAAAAAAA"), numbered(100, "r100", "BC-A", "AAAAAAAA")],
            1: [numbered(2, "r2", "BC-B", "CCCCCCCC"), numbered(20, "r20", "BC-B", "CCCCCCCC")],
            2: [numbered(10, "r10", "BC-C", "GGGGGGGG")],
        }
        mapping = {
            "r1": CorrectedUmi(barcode="BC-A", ur="AAAAAAAA", ub="AAAAAAAA"),
            "r2": CorrectedUmi(barcode="BC-B", ur="CCCCCCCC", ub="CCCCCCCC"),
            "r10": CorrectedUmi(barcode="BC-C", ur="GGGGGGGG", ub="GGGGGGGG"),
            "r20": CorrectedUmi(barcode="BC-B", ur="CCCCCCCC", ub="CCCCCCCC"),
            "r100": CorrectedUmi(barcode="BC-A", ur="AAAAAAAA", ub="AAAAAAAA"),
        }
        destination = tmp_path / "umi_map.tsv"
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            write_all_shard_maps(store, rows_by_shard, mapping)
            store.merge_shard_maps(destination)

        read_ids = [line.split("\t")[0] for line in destination.read_text().splitlines()]
        assert_that(read_ids).is_equal_to(["r1", "r2", "r10", "r20", "r100"])

    def test_merged_rows_drop_the_ordinal_and_are_byte_identical(self, tmp_path) -> None:
        rows_by_shard = {
            0: [numbered(0, "r0", "BC-A", "AAAAAAAT")],
            1: [numbered(1, "r1", "BC-B", "CCCCCCCG")],
        }
        mapping = {
            "r0": CorrectedUmi(barcode="BC-A", ur="AAAAAAAT", ub="AAAAAAAA"),
            "r1": CorrectedUmi(barcode="BC-B", ur="CCCCCCCG", ub="CCCCCCCC"),
        }
        destination = tmp_path / "umi_map.tsv"
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            write_all_shard_maps(store, rows_by_shard, mapping)
            store.merge_shard_maps(destination)

        assert_that(destination.read_bytes()).is_equal_to(
            b"r0\tBC-A\tAAAAAAAT\tAAAAAAAA\nr1\tBC-B\tCCCCCCCG\tCCCCCCCC\n"
        )

    def test_reads_absent_from_the_mapping_produce_no_row(self, tmp_path) -> None:
        rows_by_shard = {
            0: [
                numbered(0, "kept0", "BC-A", "AAAAAAAA"),
                numbered(1, "dropped", "BC-A", "AAAANAAA"),
                numbered(2, "kept2", "BC-A", "AAAAAAAA"),
            ]
        }
        mapping = {
            "kept0": CorrectedUmi(barcode="BC-A", ur="AAAAAAAA", ub="AAAAAAAA"),
            "kept2": CorrectedUmi(barcode="BC-A", ur="AAAAAAAA", ub="AAAAAAAA"),
        }
        destination = tmp_path / "umi_map.tsv"
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            write_all_shard_maps(store, rows_by_shard, mapping)
            store.merge_shard_maps(destination)

        read_ids = [line.split("\t")[0] for line in destination.read_text().splitlines()]
        assert_that(read_ids).is_equal_to(["kept0", "kept2"])

    def test_merging_empty_shards_writes_an_empty_destination(self, tmp_path) -> None:
        destination = tmp_path / "umi_map.tsv"
        with UmiShardStore(shard_count=4, temp_dir=str(tmp_path)) as store:
            write_all_shard_maps(store, {}, {})
            store.merge_shard_maps(destination)

        assert_that(destination.exists()).is_true()
        assert_that(destination.read_text()).is_equal_to("")
