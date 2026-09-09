"""Assignment stats value object and report rendering for the assign-targets stage.

Holds the tallies that summarise a target-index assignment run and render the
plain-text ``tgidx_stats.txt`` report, mirroring the UMI module's
``umi_reporting``. The stage is an assignment rather than a filter: every input
read is re-emitted carrying either a whitelist entry or ``TGIDX=NONE``, so the
counts here account for reads written as well as reads read, and the report's
wording deliberately never calls an unmatched read rejected or dropped.

``AssignCounts`` is the mutable accumulator those tallies are gathered in while
a run streams, and ``AssignStats`` is the frozen value object it becomes once
the run is over.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version


@dataclass(frozen=True)
class AssignStats:
    """Reconciling tallies for a target-index assignment run.

    Attributes:
        total_reads: Number of input reads processed.
        matched: Reads whose window resolved to a single whitelist entry.
        unmatched_no_match: Reads whose window was searched and yielded no
            usable answer. This deliberately merges two outcomes - no candidate
            at all, and candidates that tied with none resolvable - because
            ``KmerMatcher.match`` returns the identical, untouched
            ``BarcodeMatchAttempt`` from both paths (its no-candidate return and
            its ambiguity fallthrough), with ``match``, ``candidate``,
            ``read_idx`` and ``edit_distance`` all ``None``, so the caller has no
            field to branch on. Separating them would mean returning the tied
            candidates as multiple attempts, which flows into the barcode path's
            ``extraction_reporting`` ambiguity counter and therefore into the
            committed ``.bc_stats.txt`` output. A tie is also not necessarily a
            tie between targets: it arises either from two whitelist entries or
            from one entry matching at two offsets, so a counter named for
            ambiguity would mostly not be measuring ambiguity between targets.
        unmatched_no_left_anchor_pos: Reads whose header carried no anchor position tag, so
            no anchor run end and no window could be derived. Counted and
            annotated ``NONE`` rather than raised on, to conserve read count.
        unmatched_short_window: Reads whose window was shorter than the floor the
            matcher needs, typically because the read ended inside the window.
        target_counts: Mapping of whitelist entry to the number of matched reads
            assigned to it. The cross-target hopping signal.
        edit_distance_counts: Mapping of edit distance to the number of matched
            reads called at that distance.
        homopolymer_base: The anchor homopolymer base of the chemistry (e.g.
            ``"G"``), or ``None`` when that anchor is not a homopolymer.
        homopolymer_run_counts: Mapping of the observed anchor run length to the
            number of reads with that run. The observed length recorded for an
            index which itself opens with anchor bases includes those bases,
            because at that boundary they are indistinguishable from the run.

    By construction ``matched + unmatched_no_match + unmatched_no_left_anchor_pos +
    unmatched_short_window`` always equals ``total_reads``. Uniquely to this
    stage, ``total_reads`` also equals the number of reads **written**: no read
    is ever dropped, so reads written equals reads read.
    """

    total_reads: int
    matched: int
    unmatched_no_match: int
    unmatched_no_left_anchor_pos: int
    unmatched_short_window: int
    target_counts: dict[str, int]
    edit_distance_counts: dict[int, int]
    homopolymer_base: str | None = None
    homopolymer_run_counts: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    @property
    def reads_with_measured_run(self) -> int:
        """Return the number of reads whose anchor run length was measured.

        Derived from the run-length counter rather than stored alongside it, so
        the two cannot drift apart and the outcome invariant gains no extra term.
        It is the denominator of the run-length distribution: a read only reaches
        the forward scan once its anchor position tag is in hand, so the reads
        counted here are a subset of ``total_reads``.
        """
        return sum(self.homopolymer_run_counts.values())

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the run.

        Returns:
            A plain-text report carrying the run details, the denominator note,
            the matched / unmatched counts (by reason), the per-target and edit
            distance distributions, and the anchor run-length distribution when
            any run was measured.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        report = f"# Carmack version: {carmack_version}\n"
        report += f"# Report generated at: {run_time}\n"
        report += "# Target Index Assignment Stats\n"
        report += self.denominator_note()
        report += f"Total reads: {self.total_reads}\n"
        report += (
            f"Matched: {self.matched} ({self.fraction(self.matched, self.total_reads):.2%})\n"
        )
        report += (
            f"Unmatched (no_match): {self.unmatched_no_match} "
            f"({self.fraction(self.unmatched_no_match, self.total_reads):.2%})\n"
        )
        report += (
            f"Unmatched (no_left_anchor_pos): {self.unmatched_no_left_anchor_pos} "
            f"({self.fraction(self.unmatched_no_left_anchor_pos, self.total_reads):.2%})\n"
        )
        report += (
            f"Unmatched (short_window): {self.unmatched_short_window} "
            f"({self.fraction(self.unmatched_short_window, self.total_reads):.2%})\n"
        )
        report += self.target_section()
        report += self.edit_distance_section()
        if self.homopolymer_run_counts:
            report += self.anchor_run_section()
        return report

    def denominator_note(self) -> str:
        """Render the caveat that qualifies every count below it.

        The stage reads ``{prefix}.r1_umi.fastq.gz``, which ``extract-umis`` has
        already subset: a read whose anchor was never recorded, or which ends
        before its UMI does, never arrives here at all. ``total_reads`` is
        therefore not the run's read count and the matched fraction is not the
        scTIP fraction of the library. The note is emitted before the counts so
        the caveat is read before the number it qualifies; without it someone
        will read the target index rate as a modality fraction of the library.
        """
        note = "# Input is {prefix}.r1_umi.fastq.gz, which extract-umis has already subset:\n"
        note += "# a read whose anchor was never recorded, or which ends inside its UMI, does\n"
        note += "# not arrive here, so the matched fraction below is not the scTIP fraction of\n"
        note += "# the library. See the matching .umi_stats.txt for the run's total read count.\n"
        return note

    def target_section(self) -> str:
        """Render the per-target distribution over matched reads.

        Entries are sorted by whitelist entry rather than by count, so two runs
        of the same chemistry line up row for row when diffed.
        """
        section = "\n# Target Distribution\n"
        for target in sorted(self.target_counts):
            count = self.target_counts[target]
            section += f"\t{target}\t{count} ({self.fraction(count, self.matched):.2%})\n"
        return section

    def edit_distance_section(self) -> str:
        """Render the edit-distance distribution over matched reads.

        Only matched reads have an edit distance at all, so ``matched`` is the
        denominator; a distribution skewed towards the error budget is the signal
        that the index is being called on the strength of the budget alone.
        """
        section = "\n# Edit Distance Distribution\n"
        for edit_distance in sorted(self.edit_distance_counts):
            count = self.edit_distance_counts[edit_distance]
            section += f"\t{edit_distance}\t{count} ({self.fraction(count, self.matched):.2%})\n"
        return section

    def anchor_run_section(self) -> str:
        """Render the observed anchor homopolymer run-length distribution.

        The anchor base is chemistry-defined, so the section is labelled with the
        actual base rather than hard-coding poly-G, falling back to a generic
        label when the chemistry's anchor is not a homopolymer. The denominator is
        ``reads_with_measured_run``, not ``total_reads``: reads that never reached
        the forward scan have no run length to place in any bin. This is the cheap
        diagnostic for anchor slippage.
        """
        base = self.homopolymer_base or "homopolymer"
        measured = self.reads_with_measured_run
        section = f"\n# Anchor {base}-run Length Distribution\n"
        for run_length in sorted(self.homopolymer_run_counts):
            count = self.homopolymer_run_counts[run_length]
            section += f"\t{run_length}\t{count} ({self.fraction(count, measured):.2%})\n"
        return section


@dataclass
class AssignCounts:
    """Mutable accumulator for the tallies of a target-index assignment run.

    Deliberately not frozen, unlike the :class:`AssignStats` it hands off to: it
    is an accumulator, bumped read by read as a batch is tallied and folded once
    per batch into the run's running totals, and it doubles as the payload a
    worker returns for the batch it tallied, so the parent and the worker share
    one type. ``AssignStats`` stays the frozen value object the finished totals
    become, once nothing more will be added to them.

    Attributes:
        total: Reads tallied.
        matched: Reads whose window resolved to a single whitelist entry.
        no_left_anchor_pos: Reads whose header carried no anchor position tag, so no window
            could be derived.
        short_window: Reads whose window was shorter than the floor the matcher
            needs.
        no_match: Reads whose window was searched and yielded no usable answer.
        target_counts: Matched reads per whitelist entry.
        edit_distance_counts: Matched reads per edit distance.
        run_counts: Reads per observed anchor homopolymer run length.
    """

    total: int = 0
    matched: int = 0
    no_left_anchor_pos: int = 0
    short_window: int = 0
    no_match: int = 0
    target_counts: Counter[str] = field(default_factory=Counter)
    edit_distance_counts: Counter[int] = field(default_factory=Counter)
    run_counts: Counter[int] = field(default_factory=Counter)

    def add(self, other: "AssignCounts") -> None:
        """Fold another batch's tallies into these, in place.

        Args:
            other: Tallies to add in. Left untouched, so the batch it came from
                stays readable after the fold.
        """
        self.total += other.total
        self.matched += other.matched
        self.no_left_anchor_pos += other.no_left_anchor_pos
        self.short_window += other.short_window
        self.no_match += other.no_match
        # Counter.update ADDS counts, where dict.update would overwrite them.
        # Overwriting would silently lose every count an earlier batch recorded
        # for a key a later batch also saw, leaving only the last batch's.
        self.target_counts.update(other.target_counts)
        self.edit_distance_counts.update(other.edit_distance_counts)
        self.run_counts.update(other.run_counts)

    def to_stats(self, homopolymer_base: str | None) -> AssignStats:
        """Hand the accumulated tallies off as the run's frozen value object.

        Each counter is converted to a plain ``dict``, so nothing downstream can
        keep counting into what is meant to be a finished tally.

        Args:
            homopolymer_base: The anchor homopolymer base of the chemistry, or
                ``None`` when that anchor is not a homopolymer.

        Returns:
            The reconciling :class:`AssignStats` for the tallies held here.
        """
        return AssignStats(
            total_reads=self.total,
            matched=self.matched,
            unmatched_no_match=self.no_match,
            unmatched_no_left_anchor_pos=self.no_left_anchor_pos,
            unmatched_short_window=self.short_window,
            target_counts=dict(self.target_counts),
            edit_distance_counts=dict(self.edit_distance_counts),
            homopolymer_base=homopolymer_base,
            homopolymer_run_counts=dict(self.run_counts),
        )
