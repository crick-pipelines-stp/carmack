"""Prepare-reads stats value object and report rendering for the prepare-reads stage.

Holds the tallies that summarise a prepare-reads run and render both of the run's
plain-text outputs, mirroring the assign-targets and UMI modules'
``assign_reporting``/``umi_reporting``: the human-readable ``prepare_stats.txt``
report, and the machine-readable ``detected_targets.txt`` list of the arms and
buckets that actually received reads. Unlike those stages, this stage never
filters: every input read is dispatched to exactly one output arm - the scRNA
(``TGIDX=NONE``) arm, or one scTIP target bucket - so ``total_reads`` is the
denominator throughout, including for the per-target distribution, rather than
just for the unmatched count.

``PrepareCounts`` is the mutable accumulator those tallies are gathered in
while a run streams, and ``PrepareStats`` is the frozen value object it
becomes once the run is over.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from carmack import __version__ as carmack_version
from carmack.assign_targets.target_assigner import NO_TARGET


@dataclass(frozen=True)
class PrepareStats:
    """Reconciling tallies for a prepare-reads run.

    Attributes:
        total_reads: Number of input reads processed.
        unmatched_written: Reads written to the scRNA arm (``TGIDX=NONE``).
        target_written: Mapping of scTIP target to the number of reads written
            to that target's bucket.

    By construction ``unmatched_written + sum(target_written.values())``
    always equals ``total_reads``: this stage never filters, so reads written
    equals reads read.
    """

    total_reads: int
    unmatched_written: int
    target_written: dict[str, int]

    @staticmethod
    def fraction(numerator: int, denominator: int) -> float:
        """Return ``numerator / denominator``, or ``0.0`` when the denominator is zero."""
        return numerator / denominator if denominator else 0.0

    @property
    def matched_written(self) -> int:
        """Return the number of reads written to any scTIP target bucket.

        Derived from ``target_written`` rather than stored alongside it, so the
        two cannot drift apart and the reconciling invariant gains no extra
        term.
        """
        return sum(self.target_written.values())

    def get_report(self) -> str:
        """Render a human-readable, reconciling report of the run.

        Returns:
            A plain-text report carrying the run details, the literal
            reconciling invariant, the unmatched / matched counts as fractions
            of ``total_reads``, and the per-target distribution.
        """
        run_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        report = f"# Carmack version: {carmack_version}\n"
        report += f"# Report generated at: {run_time}\n"
        report += "# Prepare Reads Stats\n"
        report += self.invariant_note()
        report += f"Total reads: {self.total_reads}\n"
        report += (
            f"Unmatched (scRNA arm): {self.unmatched_written} "
            f"({self.fraction(self.unmatched_written, self.total_reads):.2%})\n"
        )
        report += (
            f"Matched (scTIP arms): {self.matched_written} "
            f"({self.fraction(self.matched_written, self.total_reads):.2%})\n"
        )
        report += self.target_section()
        return report

    def invariant_note(self) -> str:
        """Render the literal statement of the reconciling invariant.

        Stated in the field names the module exposes, so a reader of the
        report can check the invariant by eye rather than infer it from the
        numbers alone. Emitted before the counts so the formula is read before
        the numbers it reconciles.
        """
        return "# unmatched_written + sum(target_written.values()) == total_reads\n"

    def target_section(self) -> str:
        """Render the per-target distribution over every read the run saw.

        Entries are sorted by target name rather than by count, so two runs of
        the same chemistry line up row for row when diffed. Unlike
        assign-targets' equivalent section, this stage never filters, so every
        row's denominator is ``total_reads``, not ``matched_written``. The
        heading is always rendered, even with no targets at all, so a
        scRNA-only chemistry's report still names the (empty) section.
        """
        section = "\n# Target Distribution\n"
        for target in sorted(self.target_written):
            count = self.target_written[target]
            section += f"\t{target}\t{count} ({self.fraction(count, self.total_reads):.2%})\n"
        return section

    def get_detected_targets(self) -> str:
        """Render the arms and buckets that actually received reads, one token per line.

        This stage's machine-readable companion to :meth:`get_report`: bare tokens
        with no header, no comment lines and no counts, each either ``NO_TARGET``
        for the scRNA arm or a target index whitelist entry, and each naming an
        output the run actually wrote a read into. A consumer fanning out over the
        buckets a dataset really has reads this file rather than globbing the
        output directory -- every whitelisted target's bucket is opened, and so
        exists, whether or not a read ever landed in it -- or parsing the
        human-readable distribution section for the same answer.

        Targets are sorted by name, the order :meth:`target_section` renders them
        in, so the two files line up row for row. A target carrying a zero count
        is treated as undetected, exactly like one absent from ``target_written``
        altogether: both describe a bucket no read reached.

        Returns:
            One newline-terminated token per detected arm or bucket, the scRNA
            arm's ``NO_TARGET`` first when it received any read. Empty when the
            run wrote no reads at all.
        """
        detected = [NO_TARGET] if self.unmatched_written else []
        detected += [
            target for target in sorted(self.target_written) if self.target_written[target]
        ]
        return "".join(f"{token}\n" for token in detected)


@dataclass
class PrepareCounts:
    """Mutable accumulator for the tallies of a prepare-reads run.

    Deliberately not frozen, unlike the :class:`PrepareStats` it hands off to:
    it is an accumulator, bumped read by read as a batch is tallied and folded
    once per batch into the run's running totals, and it doubles as the
    payload a worker returns for the batch it tallied, so the parent and the
    worker share one type. ``PrepareStats`` stays the frozen value object the
    finished totals become, once nothing more will be added to them.

    Attributes:
        total: Reads tallied.
        unmatched: Reads written to the scRNA arm.
        target_counts: Reads written per scTIP target bucket.
    """

    total: int = 0
    unmatched: int = 0
    target_counts: Counter[str] = field(default_factory=Counter)

    def add(self, other: "PrepareCounts") -> None:
        """Fold another batch's tallies into these, in place.

        Args:
            other: Tallies to add in. Left untouched, so the batch it came
                from stays readable after the fold.
        """
        self.total += other.total
        self.unmatched += other.unmatched
        # Counter.update ADDS counts, where dict.update would overwrite them.
        # Overwriting would silently lose every count an earlier batch
        # recorded for a key a later batch also saw, leaving only the last
        # batch's.
        self.target_counts.update(other.target_counts)

    def to_stats(self) -> PrepareStats:
        """Hand the accumulated tallies off as the run's frozen value object.

        The target counter is converted to a plain ``dict``, so nothing
        downstream can keep counting into what is meant to be a finished
        tally.

        Returns:
            The reconciling :class:`PrepareStats` for the tallies held here.
        """
        return PrepareStats(
            total_reads=self.total,
            unmatched_written=self.unmatched,
            target_written=dict(self.target_counts),
        )
