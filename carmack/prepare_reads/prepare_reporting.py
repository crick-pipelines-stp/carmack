"""Prepare-reads stats value object and report rendering for the prepare-reads stage.

Holds the tallies that summarise a prepare-reads run and render both of the run's
plain-text outputs, mirroring the assign-targets and UMI modules'
``assign_reporting``/``umi_reporting``: the human-readable ``prepare_stats.txt``
report, and the machine-readable ``detected_targets.txt`` list of the arms and
buckets that actually received reads. This stage dispatches every input read to
one of two output arms - the scRNA (``TGIDX=NONE``) arm, or one scTIP target
bucket - and filters in exactly one case: a read whose computed insert start
reaches the end of the read has no insert left to write, so it is counted as
``insert_not_sequenced`` and written to no arm at all. ``total_reads`` is
therefore still the denominator throughout, including for the per-target
distribution, rather than just for the unmatched count, and the reconciling
invariant carries three terms rather than two.

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
        insert_not_sequenced: Reads written to no arm because the computed
            insert start reached the end of the read.

    By construction ``unmatched_written + sum(target_written.values()) +
    insert_not_sequenced`` always equals ``total_reads``: a read is written to
    exactly one arm, or dropped for having no sequenced insert and counted
    here, and nothing else can happen to it.
    """

    total_reads: int
    unmatched_written: int
    target_written: dict[str, int]
    insert_not_sequenced: int

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
            reconciling invariant, the unmatched / matched counts and the count
            of reads dropped for having no sequenced insert as fractions of
            ``total_reads``, and the per-target distribution.
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
        # Rendered even at zero, in the rejected vocabulary the sibling stages
        # already use: a reader of a healthy run's report can then tell "no read
        # was dropped" from "this build does not check", which is the ambiguity
        # that let zero-length records reach a consumer unnoticed.
        report += (
            f"Rejected (insert_not_sequenced): {self.insert_not_sequenced} "
            f"({self.fraction(self.insert_not_sequenced, self.total_reads):.2%})\n"
        )
        report += self.target_section()
        return report

    def invariant_note(self) -> str:
        """Render the literal statement of the reconciling invariant.

        Stated in the field names the module exposes, so a reader of the
        report can check the invariant by eye rather than infer it from the
        numbers alone. Emitted before the counts so the formula is read before
        the numbers it reconciles.

        Three terms, not two: a read is written to the scRNA arm, written to a
        target bucket, or dropped for having no sequenced insert, and nothing
        else can happen to it. The formula outgrew the line length once the
        third term was added, so it is held here as two concatenated literals;
        that is a detail of the source and never reaches the file, which carries
        the formula on one line for a reader checking it against the counts
        below it.
        """
        return (
            "# unmatched_written + sum(target_written.values()) "
            "+ insert_not_sequenced == total_reads\n"
        )

    def target_section(self) -> str:
        """Render the per-target distribution over every read the run saw.

        Entries are sorted by target name rather than by count, so two runs of
        the same chemistry line up row for row when diffed. Unlike
        assign-targets' equivalent section, every row's denominator is
        ``total_reads`` - every read the run saw, including the few dropped for
        having no sequenced insert - rather than ``matched_written``. The
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

        A read dropped for having no sequenced insert reached no arm, so it is
        named by no token here and moves no count: that omission is deliberate,
        because the consumer that sizes its arm fan-out from this file would
        otherwise size work for an output that does not exist.

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
        """Render a MultiQC generalstats payload of the run's per-outcome percentages.

        Reduces the run to the three percentages a generalstats table needs -
        the two arms a read can be written to, and the one outcome that reaches
        neither - reusing :meth:`fraction` so these can never drift from the
        ones :meth:`get_report` already prints.

        ``pct_insert_not_sequenced`` is bounded at ``max: 1`` where its two
        neighbours take the convention's ``100``, and that is a statement about
        colour rather than about units. The value stays ``fraction(...) * 100``,
        on the same 0-100 percentage scale every sibling column carries;
        ``max`` is the colour ramp's ceiling in those same units, so it
        rescales the colour and not the number. Spanning the ramp over 0-1% is
        what gives a metric that is pathological at one read in half a million
        its resolution over the range it actually varies in: against the
        conventional 0-100 every healthy library sits in the bottom thousandth
        of the scale and renders in one flat colour, saying nothing the
        column's absence would not also have said. ``format`` is deliberately
        left at the sibling convention's ``{:,.2f}``, so the cell still prints
        ``0.00%`` at that rate - what the bound buys is that the cell is no
        longer flat-coloured, and the exact integer is one line away in
        ``prepare_stats.txt``.

        Args:
            prefix: Key the per-run section of ``data`` is filed under, so a
                caller reporting several runs can tell them apart.

        Returns:
            A MultiQC custom content payload carrying ``pct_unmatched``,
            ``pct_matched`` and ``pct_insert_not_sequenced`` as percentages of
            ``total_reads``. ``namespace`` is
            what attributes those columns to Carmack, and the only key that can:
            the custom content parser branches on the generalstats plot type and
            returns before it reads ``parent_id``, so the parent keys that nest
            this stage's chart sections are inert here, and a namespace left
            unset falls back to the raw payload id -- one per stage, rather than
            one Carmack heading over all four. ``id`` names the payload's own
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
            "namespace": CARMACK_PARENT_NAME,
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
                # Scaled like the sibling stages' reject columns rather than
                # like either column above: those two count legitimate
                # destinations, so neither of their ramps can stand in for one
                # counting reads that were thrown away.
                {
                    "pct_insert_not_sequenced": {
                        "title": "% Unsequenced Insert",
                        "description": (
                            "Percentage of reads dropped because the anchor run reached the end "
                            "of the read, leaving no insert to write. On 2-colour chemistry an "
                            "unsequenced tail reads as the anchor base, so these are clusters "
                            "that died before the insert, not empty library fragments. The scan "
                            "behind this number bridges a single interrupting base, so this "
                            "count can exceed the extract-umis anchor-run distribution's count "
                            "at the saturating run length, which is measured unbridged."
                        ),
                        "min": 0,
                        "max": 1,
                        "suffix": "%",
                        "format": "{:,.2f}",
                        "scale": "YlOrRd",
                    }
                },
            ],
            "data": {
                prefix: {
                    "pct_unmatched": self.fraction(self.unmatched_written, self.total_reads) * 100,
                    "pct_matched": self.fraction(self.matched_written, self.total_reads) * 100,
                    "pct_insert_not_sequenced": self.fraction(
                        self.insert_not_sequenced, self.total_reads
                    )
                    * 100,
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
                "the scRNA arm. The categories are the arms reads were written to, so "
                "they sum to the reads the run wrote rather than to every read it saw: "
                "a read dropped for having no sequenced insert reached no arm, and is "
                "reported by the insert_not_sequenced column in the General Statistics "
                "table and by the stats report rather than as a bar here. Every arm the "
                "run wrote to is counted, unlike assign-targets' target distribution, "
                "which is a fraction of matched reads alone. These are the arms reads "
                "were actually written to, not a record of what the target index "
                "matched."
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
        insert_not_sequenced: Reads written to no arm because the computed
            insert start reached the end of the read.
        target_counts: Reads written per scTIP target bucket.
    """

    total: int = 0
    unmatched: int = 0
    insert_not_sequenced: int = 0
    target_counts: Counter[str] = field(default_factory=Counter)

    def add(self, other: "PrepareCounts") -> None:
        """Fold another batch's tallies into these, in place.

        Args:
            other: Tallies to add in. Left untouched, so the batch it came
                from stays readable after the fold.
        """
        self.total += other.total
        self.unmatched += other.unmatched
        self.insert_not_sequenced += other.insert_not_sequenced
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
            insert_not_sequenced=self.insert_not_sequenced,
        )
