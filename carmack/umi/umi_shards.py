"""On-disk shard store that keeps UMI correction off the heap.

Correction has to see every read of a cell barcode at once, but a production run
carries hundreds of millions of reads and the whole record set will not fit in
memory. Correcting one cell barcode is independent of every other, so the records
can instead be spilled to ``S`` temporary files chosen by a digest of the
barcode. A barcode group is then confined to a single shard, and correction runs
shard by shard against an unchanged
:class:`carmack.umi.umi_corrector.UmiCorrector` with only one shard resident at a
time.

The per-shard map rows come back out of extraction order, so the store also owns
the k-way merge that restores it, leaving ``umi_map.tsv`` byte for byte what a
single in-memory pass would have written.
"""

import heapq
import zlib
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from carmack.umi.umi_corrector import CorrectedUmi, UmiRecord

DEFAULT_SHARD_COUNT = 256


def shard_index(barcode: str, shard_count: int) -> int:
    """Return the shard that a cell barcode's records belong to.

    The digest is crc32 rather than the built-in ``hash``, which is salted per
    interpreter by ``PYTHONHASHSEED``: a salted digest would scatter the same
    barcode to a different shard on every run, so neither the sharding nor the
    stats derived from it would be reproducible.

    Args:
        barcode: The full cell barcode grouping the records.
        shard_count: Number of shards the store is spread over.

    Returns:
        The index of the shard owning the barcode, within
        ``range(shard_count)``.
    """
    return zlib.crc32(barcode.encode()) % shard_count


class UmiShardStore:
    """Spills UMI records to temporary per-barcode shards and merges their maps.

    The store is a context manager: leaving the block closes every write handle
    and removes the whole temporary tree, on the normal path and when an
    exception escapes alike, so a failed correction never strands spill files.

    A shard is written to first and read from second. Reading one closes that
    shard's own write handle, so a write to an already-read shard raises
    ``ValueError`` instead of landing in a file nothing opens again.
    """

    def __init__(
        self, shard_count: int = DEFAULT_SHARD_COUNT, temp_dir: str | None = None
    ) -> None:
        """Create the temporary tree and open a write handle per shard.

        Every data and map file is created up front, so a shard that never
        receives a record still reads back and merges as an empty file rather
        than needing to be special-cased.

        Args:
            shard_count: Number of shards to spread records over.
            temp_dir: Directory to create the temporary tree in; ``None``
                inherits the platform default (``TMPDIR``).

        Raises:
            ValueError: If ``shard_count`` is less than one.
        """
        if shard_count < 1:
            raise ValueError(f"shard_count must be at least 1, got {shard_count}")
        self.shard_count = shard_count
        # Build under the stack's own ``with`` and only detach it once every
        # handle is open, so construction is all or nothing. A store that fails
        # part-way -- running the process out of file descriptors on a large
        # shard count, say -- never reaches the caller, so the caller can never
        # close it; without the self-guard its half-filled tree would be left
        # behind on disk.
        self.stack = ExitStack()
        with self.stack:
            self.root = Path(
                self.stack.enter_context(TemporaryDirectory(prefix="carmack-umi-", dir=temp_dir))
            )
            self.handles = [
                self.stack.enter_context(self.data_path(index).open("w")) for index in self.indexes
            ]
            for index in self.indexes:
                self.map_path(index).touch()
            self.stack = self.stack.pop_all()

    def __enter__(self) -> "UmiShardStore":
        """Return the store for use inside a ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close every shard handle and remove the temporary tree."""
        self.stack.close()

    @property
    def indexes(self) -> range:
        """Return the range of valid shard indexes."""
        return range(self.shard_count)

    def data_path(self, index: int) -> Path:
        """Return the path of the shard's spilled record file.

        Args:
            index: The shard to locate.

        Returns:
            Path to the shard's data file.
        """
        return self.root / f"shard-{index:05d}.records.tsv"

    def map_path(self, index: int) -> Path:
        """Return the path of the shard's corrected-map file.

        Args:
            index: The shard to locate.

        Returns:
            Path to the shard's map file.
        """
        return self.root / f"shard-{index:05d}.map.tsv"

    def write(self, ordinal: int, record: UmiRecord) -> None:
        """Append one accepted record to the shard owning its cell barcode.

        Records are appended as they are extracted, so each shard file is already
        ascending by ordinal and needs no sort before it is read back.

        Args:
            ordinal: The read's position in extraction order.
            record: The extracted UMI observation to spill.
        """
        handle = self.handles[shard_index(record.barcode, self.shard_count)]
        handle.write(f"{ordinal}\t{record.read_id}\t{record.barcode}\t{record.raw_umi}\n")

    def read_shard(self, index: int) -> list[tuple[int, UmiRecord]]:
        """Read one shard's spilled records back in extraction order.

        The shard's own write handle is closed first, so the rows it is still
        holding in its text buffer reach the file: without that, the read
        returns a plausible prefix of the shard rather than nothing at all, and
        the loss goes unnoticed. Closing rather than merely flushing is
        deliberate. It hands a descriptor back as correction walks every shard
        in turn, and it turns a write to an already-read shard -- a record that
        would otherwise land in a file nothing opens again -- into a raised
        ``ValueError``. Reading the same shard twice still works, because
        closing an already-closed file does nothing.

        Args:
            index: The shard to read.

        Returns:
            The shard's ``(ordinal, record)`` pairs, ascending by ordinal.
        """
        self.handles[index].close()
        records: list[tuple[int, UmiRecord]] = []
        with self.data_path(index).open() as handle:
            for line in handle:
                ordinal, read_id, barcode, raw_umi = line.rstrip("\n").split("\t")
                records.append(
                    (int(ordinal), UmiRecord(read_id=read_id, barcode=barcode, raw_umi=raw_umi))
                )
        return records

    def write_shard_map(
        self, index: int, records: list[tuple[int, UmiRecord]], mapping: dict[str, CorrectedUmi]
    ) -> None:
        """Write one shard's corrected-map rows, carrying the ordinal as a merge key.

        Rows are emitted in the shard's own order, one per corrected read, as
        ``ordinal``, ``read_id``, ``barcode``, ``UR``, ``UB``. Reads dropped
        during correction (raw sentinel or off-length) are omitted.

        Args:
            index: The shard the rows belong to.
            records: The shard's ``(ordinal, record)`` pairs in extraction order.
            mapping: The correction result for that shard, keyed by read id.
        """
        with self.map_path(index).open("w") as handle:
            for ordinal, record in records:
                corrected = mapping.get(record.read_id)
                if corrected is None:
                    continue
                handle.write(
                    f"{ordinal}\t{record.read_id}\t{corrected.barcode}"
                    f"\t{corrected.ur}\t{corrected.ub}\n"
                )

    def discard_shard(self, index: int) -> None:
        """Delete a shard's spilled records once it has been corrected.

        Only the data file goes; the shard's map file is still needed by the
        merge, so a corrected shard's map replaces its data rather than adding
        to it: peak temporary usage stays around one copy of the spill instead
        of data plus maps.

        Args:
            index: The shard whose spilled records are no longer needed.
        """
        self.data_path(index).unlink()

    def merge_shard_maps(self, destination: Path) -> None:
        """Merge every shard's map file into one file in extraction order.

        The shard maps are each ascending by ordinal, so a streaming k-way merge
        restores extraction order while holding only one row per shard. The key
        parses the ordinal as an integer: compared as text, decimal ordinals sort
        ``1, 10, 100, 2, 20``, which would silently scramble the output past the
        tenth read. Each row is written with only its ordinal column sliced away,
        so the remainder (trailing newline included) is copied verbatim.

        Args:
            destination: Path of the merged ``umi_map.tsv`` file to write.
        """
        with ExitStack() as stack:
            sources = [stack.enter_context(self.map_path(index).open()) for index in self.indexes]
            output = stack.enter_context(destination.open("w"))
            for row in heapq.merge(*sources, key=lambda row: int(row.split("\t", 1)[0])):
                output.write(row.split("\t", 1)[1])
