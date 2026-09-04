"""UMI extraction and correction from annotated R1 FASTQ reads.

For each annotated read the extractor locates the raw UMI lying between its left
anchor (the BC1 barcode, read from the header ``BC1_POS`` tag) and the
downstream poly-G run, annotates the read with ``UMI`` and ``UMI_POS`` tags, and
records the observed UMI length and anchor run-length distributions.

Extraction is a single streaming pass. Unless ``raw`` is set, a correction pass
then groups the accepted reads by full cell barcode and collapses directional
UMI variants (see :mod:`carmack.umi.umi_corrector`), writing the corrected map
to ``{prefix}.umi_map.tsv``. With ``raw`` this is the terminal step: no map is
written and the faithful raw ``UMI`` tag stands alone.

A production run carries far more reads than the correction pass could hold at
once, so the accepted records are never accumulated in memory: they are spilled
during extraction to the per-barcode shards of a
:class:`carmack.umi.umi_shards.UmiShardStore`, corrected one shard at a time and
merged back into extraction order. The map is therefore byte for byte what a
single in-memory pass would have written, at the peak cost of one shard.
"""

import logging
from collections import Counter
from contextlib import ExitStack
from itertools import chain
from pathlib import Path

from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponentType
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.umi.umi_corrector import UmiCorrector, UmiRecord
from carmack.umi.umi_reporting import CorrectionStats, UmiExtractionStats
from carmack.umi.umi_shards import DEFAULT_SHARD_COUNT, UmiShardStore
from carmack.utils import get_prefix, homopolymer_run_length

log = logging.getLogger(__name__)


class UmiExtractor:
    """Extracts raw UMIs from an annotated R1 FASTQ using its chemistry layout.

    The extractor is a thin wrapper mirroring the barcode extractor pattern: it
    resolves the chemistry, validates that a UMI can be anchored on both sides,
    and streams the reads once to write the annotated UMI FASTQ and stats report.
    """

    def __init__(self, fastq_file: str, chemistry_name: str) -> None:
        """Resolve the chemistry and derive the UMI extraction parameters.

        Args:
            fastq_file: Path to the annotated R1 FASTQ (produced by barcode
                extraction, carrying ``BC1_POS`` header tags).
            chemistry_name: Name of the chemistry describing the read layout.

        Raises:
            ValueError: If the chemistry is unknown, cannot anchor a UMI on its
                5' side, or has no downstream poly-G anchor.
        """
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_umi_extraction():
            raise ValueError(
                f"chemistry '{chemistry_name}' has no usable left anchor for UMI extraction"
            )

        left = self.chemistry.umi_left_anchor()
        umi = self.chemistry.umi_component()
        right = self.chemistry.umi_right_anchor()

        if right is None:
            raise ValueError(
                f"chemistry '{chemistry_name}' has no poly-G anchor for UMI extraction"
            )

        self.umi_name = umi.name
        self.umi_length = umi.length
        self.umi_length_tolerance = umi.length_tolerance
        self.polyg_base = right.homopolymer_base
        self.polyg_min_run = right.min_run
        self.left_key = position_key(left.name)
        self.barcode_names = [
            component.name
            for component in self.chemistry.read_structure.get_components_by_type(
                ReadComponentType.BARCODE
            )
        ]

    def find_polyg_start(self, seq: str, umi_start: int) -> int | None:
        """Return the greedy earliest poly-G run start bounding the UMI.

        The UMI length must lie in ``[x - tol, x + tol]``, so the poly-G run may
        begin only at offsets ``[umi_start + (x - tol), umi_start + (x + tol)]``
        inclusive. Offsets are scanned from the smallest upward and the first
        position ``p`` where ``min_run`` copies of the poly-G base begin is
        returned. This naturally enforces the length window: runs beginning
        before the window make the UMI too short and are never scanned, while
        runs beginning after it make the UMI too long and are likewise skipped.

        Args:
            seq: The read sequence.
            umi_start: 0-based index at which the UMI begins (the BC1 span end).

        Returns:
            The poly-G run start index, or ``None`` when no qualifying run begins
            within the window.
        """
        run = self.polyg_base * self.polyg_min_run
        min_start = umi_start + (self.umi_length - self.umi_length_tolerance)
        max_start = umi_start + (self.umi_length + self.umi_length_tolerance)
        for polyg_start in range(min_start, max_start + 1):
            end = polyg_start + self.polyg_min_run
            if end <= len(seq) and seq[polyg_start:end] == run:
                return polyg_start
        return None

    def validate_header(self, ann: ReadAnnotation) -> None:
        """Validate that an annotated read carries the tags this chemistry needs.

        Checks the first annotated read against the supplied chemistry: it must
        carry the UMI left-anchor position tag and a value tag for every barcode
        component. A mismatch almost always means the FASTQ was produced with a
        different chemistry than the one supplied (e.g. a chemistry with three
        barcodes run against reads annotated with only two).

        Args:
            ann: The parsed header of the first annotated read.

        Raises:
            ValueError: When an expected tag is absent from the header.
        """
        required = [self.left_key, *self.barcode_names]
        missing = [key for key in required if ann.get(key) is None]
        if missing:
            raise ValueError(
                f"Annotated read '{ann.read_id}' is missing expected tag(s) {missing} "
                f"for chemistry '{self.chemistry_name}'. The annotated FASTQ may have "
                "been produced with a different chemistry."
            )

    def extract_umis(
        self,
        output_dir: str = ".",
        prefix: str | None = None,
        raw: bool = False,
        temp_dir: str | None = None,
        shard_count: int = DEFAULT_SHARD_COUNT,
    ) -> UmiExtractionStats:
        """Stream the reads, annotate extracted UMIs and write the output files.

        Accepted records are spilled to a :class:`UmiShardStore` as they are
        extracted rather than accumulated, and correction then runs shard by
        shard so only one shard is ever resident. Under ``raw`` no store is
        created at all: nothing is spilled, hashed or merged.

        Args:
            output_dir: Directory for the generated files.
            prefix: Prefix for the generated files (default: derived from the
                input filename).
            raw: When ``True`` this is the terminal step and no correction is
                applied; no ``UB`` tag or map is emitted.
            temp_dir: Directory to create the correction spill tree in; ``None``
                inherits the platform default (``TMPDIR``). Ignored under
                ``raw``. The tree is removed when the run ends, whether it
                succeeds or raises.
            shard_count: Number of shards to spread the spilled records over.
                Peak memory is roughly one shard, so more shards means less of
                it, down to the floor the largest single cell barcode sets: a
                barcode group is indivisible, so no shard count splits it. One
                shard reproduces the old single-pass behaviour.

        Returns:
            The reconciling :class:`UmiExtractionStats` for the run.

        Raises:
            ValueError: If ``shard_count`` is less than one, or the first read's
                header does not match the supplied chemistry.
        """
        log.info(f"Extracting UMIs from {self.fastq.filename} (raw={raw})...")

        prefix = prefix or get_prefix(self.fastq.filename)
        output_path = Path(output_dir)
        umi_fastq_path = output_path / f"{prefix}.r1_umi.fastq.gz"
        umi_stats_path = output_path / f"{prefix}.umi_stats.txt"
        umi_map_path = output_path / f"{prefix}.umi_map.tsv"

        total = 0
        accepted = 0
        missing_left_anchor = 0
        no_polyg_anchor = 0
        length_counts: Counter[int] = Counter()
        run_counts: Counter[int] = Counter()
        correction_stats: CorrectionStats | None = None

        # Validate the first read's header against the chemistry before writing
        # anything, so a chemistry / FASTQ mismatch fails fast with a clear error
        # rather than a cryptic failure once correction reconstructs the barcode.
        reads = self.fastq.open_read_iterator(as_string=True)
        first_read = next(reads, None)
        if first_read is not None:
            self.validate_header(ReadAnnotation.parse(first_read[0]))
            reads = chain([first_read], reads)

        with ExitStack() as stack:
            # The store's lifetime spans extraction and correction alike, so the
            # spill tree is removed even when correction raises part-way through.
            store = None if raw else stack.enter_context(UmiShardStore(shard_count, temp_dir))
            umi_stream = stack.enter_context(GzipFile(str(umi_fastq_path)).open_write_stream())
            for name, seq, qual, *_ in reads:
                total += 1
                ann = ReadAnnotation.parse(name)

                pos = ann.get(self.left_key)
                if pos is None:
                    missing_left_anchor += 1
                    continue

                umi_start = parse_span(pos)[1]
                polyg_start = self.find_polyg_start(seq, umi_start)
                if polyg_start is None:
                    no_polyg_anchor += 1
                    continue

                raw_umi = seq[umi_start:polyg_start]
                ann.set(self.umi_name, raw_umi)
                ann.set(position_key(self.umi_name), format_span(umi_start, polyg_start))
                FastqFile.write_read(umi_stream, ann.render(), seq, qual)

                accepted += 1
                length_counts[len(raw_umi)] += 1
                run_counts[homopolymer_run_length(seq, polyg_start, self.polyg_base)] += 1
                if store is not None:
                    # The ordinal is the read's 0-based position among accepted
                    # reads; the merge sorts on it to restore extraction order.
                    store.write(
                        accepted - 1,
                        UmiRecord(
                            read_id=ann.read_id,
                            barcode=self.chemistry.construct_full_barcode(
                                {name: ann.get(name) for name in self.barcode_names}
                            ),
                            raw_umi=raw_umi,
                        ),
                    )

            if store is not None:
                correction_stats = self.correct_shards(store, umi_map_path)

        stats = UmiExtractionStats(
            total_reads=total,
            accepted=accepted,
            missing_left_anchor=missing_left_anchor,
            no_polyg_anchor=no_polyg_anchor,
            length_counts=dict(length_counts),
            homopolymer_base=self.polyg_base,
            homopolymer_run_counts=dict(run_counts),
            correction=correction_stats,
        )

        with umi_stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        log.info(f"Extracted UMIs for {accepted}/{total} reads")
        return stats

    def correct_shards(self, store: UmiShardStore, umi_map_path: Path) -> CorrectionStats:
        """Correct every spilled shard in turn and merge their map rows into order.

        The store is closed first so the whole spill is readable, then each shard
        is read, corrected, written out as map rows and discarded, holding one
        shard at a time. Correcting a shard in isolation is exact because the
        shard is a function of the cell barcode alone, so a barcode's reads are
        never split across two shards. One corrector serves every shard: it is
        stateless between calls and building its clusterer is not free.

        Args:
            store: The shard store holding the run's spilled records.
            umi_map_path: Destination path for the merged ``umi_map.tsv``.

        Returns:
            The run-level :class:`CorrectionStats`, summed over the shards.
        """
        store.close()
        corrector = UmiCorrector(self.umi_length, self.umi_length_tolerance)
        parts: list[CorrectionStats] = []
        for index in store.indexes:
            shard = store.read_shard(index)
            mapping, shard_stats = corrector.correct(record for ordinal, record in shard)
            store.write_shard_map(index, shard, mapping)
            store.discard_shard(index)
            parts.append(shard_stats)
        store.merge_shard_maps(umi_map_path)
        return CorrectionStats.combine(parts)
