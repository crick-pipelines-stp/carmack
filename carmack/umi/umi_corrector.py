"""UMI correction: normalise + directional clustering.

Step 2 of UMI handling, running after raw extraction and skipped by ``--raw``.
Reads are grouped by their full cell barcode; within each group the raw UMIs are
normalised to the canonical length and collapsed with umi_tools' directional
clusterer so that PCR / sequencing variants of one molecule share a single
corrected UMI (``UB``) while the faithful raw UMI (``UR``) is retained.

Raw UMIs already containing the padding sentinel are dropped before clustering:
a genuine sentinel base would otherwise collide with a padding sentinel and
silently inflate a cluster's count. This is the companion rule to
:data:`carmack.umi.umi_normalizer.UMI_PAD_CHAR`.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from umi_tools import UMIClusterer

from carmack.umi.umi_normalizer import UMI_PAD_CHAR, normalize_umi
from carmack.umi.umi_reporting import CorrectionStats


@dataclass(frozen=True)
class UmiRecord:
    """A single extracted UMI observation awaiting correction.

    Attributes:
        read_id: Identifier of the read the UMI was extracted from.
        barcode: The full cell barcode; the grouping key for correction.
        raw_umi: The faithful raw UMI sequence as extracted (``UR``).
    """

    read_id: str
    barcode: str
    raw_umi: str


@dataclass(frozen=True)
class CorrectedUmi:
    """The corrected UMI assignment for one read.

    Attributes:
        barcode: The full cell barcode the read belongs to.
        ur: The faithful raw UMI, unchanged from extraction.
        ub: The corrected, length-``x`` cluster representative (trailing padding
            sentinels retained).
    """

    barcode: str
    ur: str
    ub: str


class UmiCorrector:
    """Normalises and directionally clusters raw UMIs within each cell barcode."""

    def __init__(self, x: int, tol: int) -> None:
        """Store the UMI length parameters and build the directional clusterer.

        Args:
            x: Canonical UMI length; every normalised UMI is forced to this
                length so umi_tools' equal-length requirement is satisfied.
            tol: Length tolerance; raw UMIs outside ``[x - tol, x + tol]`` are
                rejected during normalisation.
        """
        self.x = x
        self.tol = tol
        self.clusterer = UMIClusterer(cluster_method="directional")

    def correct(
        self, records: Iterable[UmiRecord]
    ) -> tuple[dict[str, CorrectedUmi], CorrectionStats]:
        """Correct raw UMIs grouped by full cell barcode.

        Args:
            records: The per-read extracted UMI observations.

        Returns:
            A ``(mapping, stats)`` pair. ``mapping`` maps each corrected read id
            to its :class:`CorrectedUmi`; only correctable reads (those that
            survive the raw-sentinel and length filters) appear. ``stats`` holds
            the reconciling :class:`CorrectionStats` for the run.
        """
        groups: dict[str, list[UmiRecord]] = defaultdict(list)
        for record in records:
            groups[record.barcode].append(record)

        mapping: dict[str, CorrectedUmi] = {}
        corrections_applied = 0
        num_cell_barcodes = 0
        dropped_raw_n = 0
        dropped_off_length = 0
        distinct_corrected_umis = 0
        umi_collapses = 0

        for barcode, group in groups.items():
            survivors: list[tuple[UmiRecord, bytes]] = []
            counts: Counter[bytes] = Counter()
            for record in group:
                if UMI_PAD_CHAR in record.raw_umi:
                    dropped_raw_n += 1
                    continue
                normalized = normalize_umi(record.raw_umi, self.x, self.tol)
                if normalized is None:
                    dropped_off_length += 1
                    continue
                key = normalized.encode()
                survivors.append((record, key))
                counts[key] += 1

            if not counts:
                continue

            num_cell_barcodes += 1
            representative = self.build_representatives(counts)
            distinct_corrected_umis += len(set(representative.values()))
            umi_collapses += len(counts) - len(set(representative.values()))

            for record, key in survivors:
                ub = representative[key]
                if ub != key.decode():
                    corrections_applied += 1
                mapping[record.read_id] = CorrectedUmi(barcode=barcode, ur=record.raw_umi, ub=ub)

        stats = CorrectionStats(
            assigned_reads=len(mapping),
            corrections_applied=corrections_applied,
            num_cell_barcodes=num_cell_barcodes,
            dropped_raw_n=dropped_raw_n,
            dropped_off_length=dropped_off_length,
            distinct_corrected_umis=distinct_corrected_umis,
            umi_collapses=umi_collapses,
        )
        return mapping, stats

    def build_representatives(self, counts: Counter[bytes]) -> dict[bytes, str]:
        """Cluster one barcode's normalised UMIs and map each to its representative.

        A cluster's representative is its highest-count member, with equal counts
        resolved in favour of the lexicographically smaller UMI sequence. That
        tie-break has to be imposed here: umi_tools seeds its cluster search by
        stably sorting the mapping it is handed on count alone, so equal-count
        UMIs are seeded in that mapping's own iteration order and whichever seeds
        first permanently claims a lower-count neighbour they share. Left in the
        read-arrival order the counts were tallied in, the same reads presented
        in a different order would therefore correct differently, so the counts
        are handed over ordered by descending count then ascending sequence.

        Args:
            counts: Mapping of normalised-UMI bytes to observed read counts within
                a single cell barcode.

        Returns:
            A mapping from each normalised-UMI bytes key to its cluster
            representative string (the highest-count member, decoded).
        """
        ordered_counts = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        clusters = self.clusterer(ordered_counts, threshold=1)
        representative: dict[bytes, str] = {}
        for cluster in clusters:
            rep = cluster[0].decode()
            for member in cluster:
                representative[member] = rep
        return representative
