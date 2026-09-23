"""Position-based ("linear") deduplication of aligned, paired-end BAM reads.

For each cell (grouped by a configurable barcode tag), reads are grouped by
chromosome and strand-aware fragment position, and only the single
highest-scoring read pair per group is kept. This is an alternative to
UMI-based deduplication, for chemistries with no UMI to key on.

This module implements a two-pass design. Pass 1 finds, for every duplicate
group, the query name of the single best-scoring pair. The ``+`` strand key
tracks the BAM's own coordinate sort order, so those duplicate groups are
always contiguous -- but the ``-`` strand key does not: two reverse-strand R1
reads with the same fragment end but different starts can land at different,
non-adjacent positions in a coordinate-sorted file. A duplicate group is
therefore not guaranteed to be contiguous, which rules out a
single-pass/windowed design and requires seeing the whole file before any
winner is final. Pass 2 reopens the BAM and writes out both mates of every
winning pair, unchanged, to a coordinate-sorted, indexed output BAM.

A fragment key always includes its chromosome, so a duplicate group never
spans two contigs. Pass 1 uses this to bound its own memory: it resolves
winners one reference contig at a time (using the BAM's index), discarding
each contig's duplicate-group accumulator before moving to the next, rather
than holding one accumulator sized to the whole genome. A final scan over the
reads with no coordinate at all (both mates unmapped) accounts for them in
the run's stats without ever contributing a winner.
"""

import logging
from contextlib import ExitStack
from pathlib import Path

import pysam

from carmack.mqc_report import write_mqc_payloads
from carmack.utils import get_prefix, progress_bar

from .linear_dedup_reporting import LinearDedupStats

log = logging.getLogger(__name__)

# pysam's region syntax for "reads with no coordinate at all" (both mates unmapped) --
# the trailing block of a coordinate-sorted BAM that a plain fetch(contig) never reaches.
NO_COORDINATE_REGION = "*"


class LinearDedup:
    """Position- and score-based deduplication engine for aligned BAM reads."""

    def __init__(self, bam: str, bai: str, barcode_tag: str = "CB") -> None:
        """Store the input BAM/BAI paths and the cell-barcode tag to group by.

        Args:
            bam: Path to the coordinate-sorted, indexed input BAM.
            bai: Path to the BAM's index.
            barcode_tag: The tag carrying the cell barcode used to group
                reads into duplicate groups.
        """
        self.bam = bam
        self.bai = bai
        self.barcode_tag = barcode_tag

        log.debug(
            f"LinearDedup object created with BAM: {self.bam}, BAI: {self.bai}, "
            f"barcode_tag: {self.barcode_tag}"
        )

    def fragment_key(self, read: pysam.AlignedSegment) -> tuple[str, str, bool, int]:
        """Compute the strand-aware fragment key a read's duplicate group is keyed on.

        Args:
            read: A primary, paired, mapped R1 record carrying the configured
                barcode tag.

        Returns:
            ``(barcode, chrom, is_reverse, pos)``, where ``pos`` is
            ``reference_end`` for a reverse-strand read and
            ``reference_start`` otherwise -- the biological fragment/cut-site
            convention, not a literal always-leftmost-coordinate reading.
        """
        barcode = read.get_tag(self.barcode_tag)
        chrom = read.reference_name
        pos = read.reference_end if read.is_reverse else read.reference_start
        return (barcode, chrom, read.is_reverse, pos)

    def read_score(self, read: pysam.AlignedSegment) -> float:
        """Score a read for duplicate-group winner selection.

        Args:
            read: A primary, paired, mapped R1 record.

        Returns:
            The read's ``AS`` (alignment score) tag as a float, or negative
            infinity when the tag is absent -- logged, since ``AS`` is
            aligner-supplied and its absence is unexpected, but never fatal:
            such a read still wins a group it is alone in.
        """
        if not read.has_tag("AS"):
            log.warning(
                f"Read {read.query_name} has no AS tag; treated as lowest priority for scoring."
            )
            return float("-inf")
        return float(read.get_tag("AS"))

    def find_best_reads(self, input_bam: pysam.AlignmentFile) -> tuple[set[str], LinearDedupStats]:
        """Scan every R1 record once and resolve the single winner per duplicate group.

        Reads are scanned one reference contig at a time (via the BAM's own index), then
        once more over the trailing block of pairs with no coordinate at all. A fragment
        key always includes its chromosome, so no duplicate group spans two contigs --
        only one contig's ``best_by_key`` accumulator is ever held at once, bounding its
        peak size to the largest single contig's duplicate-group count rather than the
        whole genome's.

        Args:
            input_bam: An open, coordinate-sorted, indexed BAM.

        Returns:
            A tuple of the winning query names (one per duplicate group, across every
            contig) and the reconciling :class:`LinearDedupStats` for the whole scan.

        Raises:
            ValueError: If a primary, paired, mapped R1 record carries no
                barcode tag.
        """
        total_pairs = 0
        eligible_pairs = 0
        skipped_unmapped = 0
        skipped_non_primary = 0
        skipped_unpaired = 0
        reads_missing_as = 0
        eligible_pairs_by_chromosome: dict[str, int] = {}
        pairs_kept_by_chromosome: dict[str, int] = {}
        winners: set[str] = set()

        # AlignmentFile.mapped/.unmapped are read straight from the BAM index, so this
        # avoids the full extra linear scan input_bam.count(until_eof=True) would cost.
        total_reads = input_bam.mapped + input_bam.unmapped
        with progress_bar(unit="reads") as pbar:
            task = pbar.add_task("Deduplicating reads", total=total_reads)

            for contig in (*input_bam.references, NO_COORDINATE_REGION):
                contig_label = "no-coordinate reads" if contig == NO_COORDINATE_REGION else contig
                best_by_key: dict[tuple[str, str, bool, int], tuple[str, float]] = {}
                contig_eligible = 0

                for read in input_bam.fetch(contig=contig):
                    pbar.advance(task)

                    if not read.is_read1:
                        continue
                    total_pairs += 1

                    if read.is_secondary or read.is_supplementary:
                        skipped_non_primary += 1
                        continue
                    if not read.is_paired:
                        skipped_unpaired += 1
                        continue
                    if read.is_unmapped or read.mate_is_unmapped:
                        skipped_unmapped += 1
                        continue
                    if not read.has_tag(self.barcode_tag):
                        raise ValueError(f"Read does not have a barcode tag ({self.barcode_tag}).")

                    eligible_pairs += 1
                    contig_eligible += 1
                    chrom = read.reference_name
                    eligible_pairs_by_chromosome[chrom] = (
                        eligible_pairs_by_chromosome.get(chrom, 0) + 1
                    )

                    score = self.read_score(read)
                    if score == float("-inf"):
                        reads_missing_as += 1

                    key = self.fragment_key(read)
                    current = best_by_key.get(key)
                    if current is None or score > current[1]:
                        best_by_key[key] = (read.query_name, score)

                if contig_eligible or best_by_key:
                    log.info(
                        f"linear-dedup pass 1: {contig_label} done "
                        f"({contig_eligible} eligible pairs, {len(best_by_key)} duplicate-group "
                        "winners)."
                    )
                if best_by_key:
                    pairs_kept_by_chromosome[contig] = len(best_by_key)
                    winners.update(qname for qname, _ in best_by_key.values())
                # best_by_key goes out of scope on the next iteration -- this is the whole
                # point of scanning contig by contig rather than the whole genome at once.

        stats = LinearDedupStats(
            total_pairs=total_pairs,
            eligible_pairs=eligible_pairs,
            skipped_unmapped=skipped_unmapped,
            skipped_non_primary=skipped_non_primary,
            skipped_unpaired=skipped_unpaired,
            reads_missing_as=reads_missing_as,
            eligible_pairs_by_chromosome=eligible_pairs_by_chromosome,
            pairs_kept_by_chromosome=pairs_kept_by_chromosome,
        )
        return winners, stats

    def write_deduplicated_reads(
        self,
        input_bam: pysam.AlignmentFile,
        output_bam: pysam.AlignmentFile,
        winners: set[str],
    ) -> int:
        """Write both mates of every winning pair, unchanged, to the output BAM.

        Every record in ``input_bam`` is examined (both R1 and R2), so a
        winner's mate is written regardless of where it physically sits in
        the file. A primary record's query name is checked again
        independently of pass 1, so a secondary or supplementary alignment
        sharing a winner's query name is still excluded.

        Args:
            input_bam: An open, coordinate-sorted, indexed BAM.
            output_bam: An open BAM opened for writing, sharing the input's
                header.
            winners: The winning query names resolved by
                :meth:`find_best_reads`.

        Returns:
            The number of records written.
        """
        written = 0
        last_sort_key: tuple[int, int] | None = None
        total_reads = input_bam.mapped + input_bam.unmapped
        with progress_bar(unit="reads") as pbar:
            task = pbar.add_task("Writing deduplicated reads", total=total_reads)
            for read in input_bam.fetch(until_eof=True):
                pbar.advance(task)

                if read.is_secondary or read.is_supplementary:
                    continue
                if read.query_name not in winners:
                    continue

                # linear_dedup_reads skips re-sorting this output on the assumption that
                # filtering an already coordinate-sorted stream can never unsort it. Every
                # winner is, by construction, a mapped pair, so this checks every written
                # record without needing to special-case unmapped/no-coordinate reads.
                sort_key = (read.reference_id, read.reference_start)
                if last_sort_key is not None and sort_key < last_sort_key:
                    log.warning(
                        f"Deduplicated output is out of coordinate order at "
                        f"{read.query_name} ({sort_key} < {last_sort_key}); the "
                        "no-resort assumption in linear_dedup_reads has been violated."
                    )
                last_sort_key = sort_key

                output_bam.write(read)
                written += 1

        log.info(f"linear-dedup pass 2: wrote {written} records for {len(winners)} winning pairs.")
        return written

    def linear_dedup_reads(self, output_dir: str, prefix: str | None = None) -> LinearDedupStats:
        """Run both passes end to end and write the coordinate-sorted, indexed output BAM.

        The output BAM is written directly in its final form: pass 2 only ever filters
        the already coordinate-sorted input stream, it never reorders it, and the header
        (copied from the input via ``template=``) already carries the input's own
        ``SO:coordinate`` tag. There is nothing left for an external sort to fix, so this
        indexes the filtered output directly rather than re-sorting the whole thing first.

        Args:
            output_dir: Directory to write the output BAM (and its index)
                into.
            prefix: Prefix for the generated files (default: derived from
                the input BAM's filename).

        Returns:
            The :class:`LinearDedupStats` resolved by pass 1
            (:meth:`find_best_reads`).

        Raises:
            ValueError: If a primary, paired, mapped R1 record carries no
                barcode tag.
        """
        prefix = prefix or get_prefix(self.bam)
        output_path = Path(output_dir)
        output_bam_path = output_path / f"{prefix}.linear_dedup.bam"

        log.info(f"linear-dedup: starting pass 1 (winner resolution) for {self.bam}")
        with ExitStack() as stack:
            pass_one_bam = stack.enter_context(
                pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai)
            )
            winners, stats = self.find_best_reads(pass_one_bam)
        log.info(
            f"linear-dedup: pass 1 complete - {stats.eligible_pairs} eligible pairs, "
            f"{len(winners)} winning fragment-key groups."
        )

        log.info("linear-dedup: starting pass 2 (writing deduplicated output)")
        with ExitStack() as stack:
            input_bam = stack.enter_context(
                pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai)
            )
            output_bam = stack.enter_context(
                pysam.AlignmentFile(str(output_bam_path), "wb", template=input_bam)
            )
            self.write_deduplicated_reads(input_bam, output_bam, winners)

        pysam.index(str(output_bam_path))
        log.info(f"linear-dedup: wrote indexed output BAM to {output_bam_path}")

        stats_path = output_path / f"{prefix}.linear_dedup_stats.txt"
        with stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        # The chromosome-breakdown payload is None when eligible_pairs_by_chromosome is
        # empty, and the writer skips it rather than emitting an empty chart.
        write_mqc_payloads(
            output_path,
            prefix,
            [
                stats.to_mqc_general_stats(prefix),
                stats.to_mqc_breakdown(prefix),
                stats.to_mqc_chromosome_breakdown(prefix),
            ],
        )

        return stats
