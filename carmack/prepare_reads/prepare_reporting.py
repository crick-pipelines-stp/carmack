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
from typing import Any

from carmack import __version__ as carmack_version
from carmack.assign_targets.target_assigner import NO_TARGET
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME


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
        """Render the arms and buckets that received reads, one token and count per line.

        This stage's machine-readable companion to :meth:`get_report`: tab-separated
        ``token<TAB>count`` lines with no header and no comment lines, the token
        either ``NO_TARGET`` for the scRNA arm or a target index whitelist entry,
        and each line naming an output the run actually wrote a read into. A
        consumer fanning out over the buckets a dataset really has reads this file
        rather than globbing the output directory -- every whitelisted target's
        bucket is opened, and so exists, whether or not a read ever landed in it --
        or parsing the human-readable distribution section for the same answer.

        The count rides alongside the token because a fan-out that has to size the
        work per bucket otherwise has to read the MultiQC report to get it, which
        turns a report-shaped artefact into pipeline control flow and makes any
        change to how the report is shaped a breaking change downstream. The
        tallies are already held here, so carrying them gives that consumer a
        contract that only moves when the buckets themselves do.

        Targets are sorted by name, the order :meth:`target_section` renders them
        in, so the two agree on the order of the targets they share. A target
        carrying a zero count is treated as undetected, exactly like one absent
        from ``target_written`` altogether: both describe a bucket no read
        reached, so neither is rendered at all, not even as a zero-count row.

        Returns:
            One newline-terminated ``token<TAB>count`` line per detected arm or
            bucket, the scRNA arm's ``NO_TARGET`` first when it received any read.
            Empty when the run wrote no reads at all.
        """
        detected = [(NO_TARGET, self.unmatched_written)] if self.unmatched_written else []
        detected += [
            (target, self.target_written[target])
            for target in sorted(self.target_written)
            if self.target_written[target]
        ]
        return "".join(f"{token}\t{count}\n" for token, count in detected)

    def to_mqc_general_stats(self, prefix: str) -> dict[str, Any]:
        """Render a MultiQC generalstats payload of the run's unmatched and matched percentages.

        Reduces the run to the two percentages a generalstats table needs,
        reusing :meth:`fraction` so these can never drift from the ones
        :meth:`get_report` already prints.

        Args:
            prefix: Key the per-run section of ``data`` is filed under, so a
                caller reporting several runs can tell them apart.

        Returns:
            A MultiQC custom content payload attached to the shared Carmack
            parent module, carrying ``pct_unmatched`` and ``pct_matched`` as
            percentages of ``total_reads``. ``id`` names the payload's own
            MultiQC module: without it the custom content parser falls back to
            the cleaned filename, so the module varies with the sample and an
            N-sample run renders N one-row tables rather than one table with N
            rows. ``pconfig`` declares a header per data column: without it
            MultiQC guesses the columns from the raw data keys and renders them
            untitled, unsuffixed and unscaled beside the sibling stages' fully
            configured percentage columns.
        """
        return {
            "id": "carmack_prepare_general_stats",
            "plot_type": "generalstats",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "pconfig": [
                # Scaled informationally rather than as a warning, unlike the
                # sibling stages' leftover columns: a read on the scRNA arm is
                # one of this stage's two legitimate destinations, not a read
                # lost, so a red tail would report a problem that is not one.
                {
                    "pct_unmatched": {
                        "title": "% scRNA Arm",
                        "description": "Percentage of reads written to the scRNA arm, carrying no scTIP target index.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlGnBu",
                    }
                },
                {
                    "pct_matched": {
                        "title": "% scTIP Arms",
                        "description": "Percentage of reads written to a scTIP target bucket.",
                        "min": 0,
                        "max": 100,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "RdYlGn",
                    }
                },
            ],
            "data": {
                prefix: {
                    "pct_unmatched": self.fraction(self.unmatched_written, self.total_reads) * 100,
                    "pct_matched": self.fraction(self.matched_written, self.total_reads) * 100,
                }
            },
        }

    def to_mqc_target_distribution(self, prefix: str) -> dict[str, Any]:
        """Render a MultiQC bargraph payload of the per-target distribution.

        Mirrors :meth:`target_section`'s own sort by target name, and adds the
        unmatched (scRNA arm) count as one more category, so every read the
        run saw is accounted for in one chart.

        Args:
            prefix: Key the per-run section of ``data`` is filed under, so a
                caller reporting several runs can tell them apart.

        Returns:
            A MultiQC custom content payload attached to the shared Carmack
            parent module, carrying one category per target in
            ``target_written`` plus the unmatched arm, keyed by ``NO_TARGET``.
            ``id`` names the payload's own MultiQC module: without it the
            custom content parser falls back to the cleaned filename, so an
            N-sample run renders N single-sample bargraphs rather than one
            chart carrying every sample's bars. ``section_name`` and
            ``description`` head the section that chart renders as: this chart
            and assign-targets' hang off the same parent and are both "the
            target distribution" for their stage, so the heading has to say
            whose numbers these are and the description has to say what the
            bars count and against what denominator. ``pconfig`` names the plot
            itself, keeping it addressable separately from the section it sits
            in, and labels the axis for the reads the bars count.
        """
        categories = {
            target: self.target_written[target] for target in sorted(self.target_written)
        }
        categories[NO_TARGET] = self.unmatched_written
        return {
            "id": "carmack_prepare_target_distribution",
            "plot_type": "bargraph",
            "parent_id": CARMACK_PARENT_ID,
            "parent_name": CARMACK_PARENT_NAME,
            "section_name": "Prepare Reads Output Arm Distribution",
            "description": (
                "Read counts per output arm: one category per scTIP target bucket, plus "
                "the scRNA arm. This stage never filters, so the categories partition "
                "every read the run saw and the denominator is the whole run, unlike "
                "assign-targets' target distribution, which is a fraction of matched "
                "reads alone. These are the arms reads were actually written to, not a "
                "record of what the target index matched."
            ),
            "pconfig": {
                "id": "carmack_prepare_target_distribution_plot",
                "title": "Prepare Reads: Output Arm Distribution",
                "ylab": "Reads",
            },
            "data": {prefix: categories},
        }


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
