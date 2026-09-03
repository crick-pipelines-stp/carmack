"""Target index assignment from the UMI-annotated R1 FASTQ.

For each annotated read the assigner derives a bounded window off the end of the
anchor homopolymer run recorded by UMI extraction, matches the target index
lying in that window against the chemistry whitelist, and re-emits the read
carrying either a whitelist entry with its ``TGIDX_POS`` span or ``TGIDX=NONE``.

This is an assignment, not a filter. ``NONE`` is the correct answer for every
scRNA read in a mixed library, so no read is ever dropped and reads written
reconciles with reads read. That shapes the failure model: the checks in this
module are all about the chemistry, made once at construction, because a
chemistry that could never assign a target is an operator error worth failing
on, whereas a read that carries no target is simply a read that carries no
target.
"""

import logging
from collections import Counter
from itertools import chain
from pathlib import Path

from carmack.assign_targets.assign_reporting import AssignStats
from carmack.assign_targets.tgidx_locator import homopolymer_run_end, locate_tgidx_window
from carmack.barcode.matchers.kmer_matcher import KmerMatcher
from carmack.chemistry.annotation import format_span, parse_span, position_key
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.io.fastq_file import FastqFile
from carmack.io.gzip_file import GzipFile
from carmack.io.read_annotation import ReadAnnotation
from carmack.utils import get_prefix

log = logging.getLogger(__name__)

# Written to the TGIDX tag of every read no whitelist entry was assigned to.
# It is a value, not an absence: a read carrying it has been through the stage
# and come out unassigned, which is a different thing from a read that never
# reached it, and every downstream consumer keys on this literal to tell them
# apart.
NO_TARGET = "NONE"


class TargetAssigner:
    """Assigns target indices to annotated R1 reads using its chemistry layout.

    The assigner mirrors the UMI extractor's shape: it resolves the chemistry,
    validates up front that the chemistry could ever locate a target index, and
    streams the reads once to write the annotated FASTQ and stats report.
    """

    def __init__(self, fastq_file: str, chemistry_name: str) -> None:
        """Resolve the chemistry and derive the target assignment parameters.

        Args:
            fastq_file: Path to the UMI-annotated R1 FASTQ (produced by UMI
                extraction, carrying ``UMI_POS`` header tags).
            chemistry_name: Name of the chemistry describing the read layout.

        Raises:
            ValueError: If the chemistry is unknown, declares no target index
                anchored by a homopolymer run, cannot locate that run's start
                from the UMI span, or declares a target index whitelist that
                loads no entries.
        """
        self.fastq = FastqFile(fastq_file)
        self.chemistry_name = chemistry_name
        self.chemistry = ChemistryFactory.get_chemistry(chemistry_name)

        if not self.chemistry.supports_target_assignment():
            raise ValueError(
                f"chemistry '{chemistry_name}' declares no target index anchored by a "
                "homopolymer run, so no target can be assigned"
            )

        tgidx = self.chemistry.tgidx_component()
        anchor = self.chemistry.tgidx_anchor()
        umi = self.chemistry.umi_component()
        right = self.chemistry.umi_right_anchor()

        # The linchpin of the stage. The anchor run's start is not recorded
        # anywhere in the read; it is read back from the END of the UMI span
        # that extraction wrote. That equality holds only when the component
        # following the UMI is the very homopolymer that anchors the target
        # index. Where the two diverge - a linker between them, or no UMI at all
        # - the window would hang off a coordinate belonging to some other
        # component and every read would be scored, silently and plausibly,
        # against arbitrary sequence. There is no downstream symptom to catch
        # that by, so it has to be fatal here.
        if umi is None:
            raise ValueError(
                f"chemistry '{chemistry_name}' declares no UMI component, so the anchor run "
                "start cannot be read from a UMI span and no target window can be derived"
            )
        if right is None:
            raise ValueError(
                f"chemistry '{chemistry_name}' has no anchor 3' of its UMI, so the anchor run "
                "start cannot be read from the UMI span and no target window can be derived"
            )
        if right.name != anchor.name:
            raise ValueError(
                f"chemistry '{chemistry_name}' bounds its UMI with '{right.name}' but anchors "
                f"its target index on '{anchor.name}'. The UMI span end is only the anchor run "
                "start when these are the same component, so the target window would be cut "
                "from the wrong coordinate"
            )

        # Closed here rather than in the chemistry because
        # ChemistryBase.supports_target_assignment() asks only whether the index
        # is preceded by a homopolymer anchor and never looks at the whitelist.
        # An empty whitelist would sail through it and build a KmerMatcher over
        # an empty seed index, which returns no candidates for any window and so
        # reports every read as NONE - a run that looks complete and is entirely
        # a miss.
        whitelist = self.chemistry.tgidx_whitelist()
        if not whitelist:
            raise ValueError(
                f"chemistry '{chemistry_name}' loads no target index whitelist entries, so "
                "every read would be reported as unassigned"
            )

        self.tgidx_name = tgidx.name
        self.tgidx_length = tgidx.length
        self.anchor_base = anchor.homopolymer_base
        self.max_errors = self.chemistry.max_errors.tgidx
        self.umi_pos_key = position_key(umi.name)

        # Built once, here, because the k-mer index is built inside the
        # matcher's constructor: a matcher created per read would rebuild that
        # index for every read in the FASTQ and dominate the stage's runtime.
        self.matcher = KmerMatcher(
            whitelist=whitelist,
            component=tgidx,
            chemistry=self.chemistry,
            max_errors=self.max_errors,
        )

    def validate_header(self, ann: ReadAnnotation) -> None:
        """Validate that an annotated read carries the tag this stage reads.

        Checks the first annotated read against the supplied chemistry: it must
        carry the UMI position tag, since its span end is where the anchor run
        is found. Its absence means the FASTQ was produced with a different
        chemistry than the one supplied, or was never put through UMI
        extraction at all.

        Args:
            ann: The parsed header of the first annotated read.

        Raises:
            ValueError: When the UMI position tag is absent from the header.
        """
        if ann.get(self.umi_pos_key) is None:
            raise ValueError(
                f"Annotated read '{ann.read_id}' is missing the expected "
                f"'{self.umi_pos_key}' tag for chemistry '{self.chemistry_name}'. The "
                "annotated FASTQ may have been produced with a different chemistry, or "
                "without running the 'extract-umis' stage."
            )

    def assign_targets(self, output_dir: str = ".", prefix: str | None = None) -> AssignStats:
        """Stream the reads, assign target indices and write the output files.

        Every input read is re-emitted exactly once, in input order, with its
        sequence and quality untouched: this stage annotates, it never filters.
        ``NONE`` is a valid and expected outcome rather than a failure - it is
        the correct answer for every scRNA read in a mixed library - which is
        what makes reads written reconcile with reads read. That is the sharp
        difference from ``extract-umis``, whose annotated FASTQ holds only the
        reads it succeeded on, so its output is a subset of its input.

        Args:
            output_dir: Directory for the generated files.
            prefix: Prefix for the generated files (default: derived from the
                input filename).

        Returns:
            The reconciling :class:`AssignStats` for the run.

        Raises:
            ValueError: If the first annotated read carries no UMI position tag,
                raised before any output file is opened.
        """
        log.info(f"Assigning target indices in {self.fastq.filename}...")

        prefix = prefix or get_prefix(self.fastq.filename)
        output_path = Path(output_dir)
        tgidx_fastq_path = output_path / f"{prefix}.r1_tgidx.fastq.gz"
        tgidx_stats_path = output_path / f"{prefix}.tgidx_stats.txt"

        total = 0
        matched = 0
        no_umi_pos = 0
        short_window = 0
        no_match = 0
        target_counts: Counter[str] = Counter()
        edit_distance_counts: Counter[int] = Counter()
        run_counts: Counter[int] = Counter()

        # Validate the first read's header before opening either output file, so
        # a FASTQ that never went through extract-umis fails fast and leaves no
        # truncated outputs behind to be mistaken for a completed run. The read
        # is pushed back onto the iterator rather than consumed, since validating
        # it is not the same as processing it.
        reads = self.fastq.open_read_iterator(as_string=True)
        first_read = next(reads, None)
        if first_read is not None:
            self.validate_header(ReadAnnotation.parse(first_read[0]))
            reads = chain([first_read], reads)

        with GzipFile(str(tgidx_fastq_path)).open_write_stream() as tgidx_stream:
            for name, seq, qual, *_ in reads:
                total += 1
                ann = ReadAnnotation.parse(name)
                attempt = None
                window = None

                pos = ann.get(self.umi_pos_key)
                if pos is None:
                    # Fatal on the first read, merely unassigned here: by this
                    # point the chemistry has been shown to match the FASTQ, so a
                    # read missing the tag is a read, not an operator error.
                    no_umi_pos += 1
                else:
                    run_start = parse_span(pos)[1]
                    run_end = homopolymer_run_end(seq, run_start, self.anchor_base)
                    run_counts[run_end - run_start] += 1
                    # The seed length is passed explicitly, never defaulted. Both
                    # defaults are 4 today so omitting it would work by
                    # coincidence, and TrimWindow.is_matchable only keeps a
                    # window the matcher would raise on away from the matcher
                    # while the two agree. Threading the matcher's own k through
                    # is what stops them drifting apart unnoticed.
                    window = locate_tgidx_window(
                        seq,
                        run_end,
                        self.tgidx_length,
                        self.max_errors,
                        k=self.matcher.k,
                    )
                    if not window.is_matchable:
                        short_window += 1
                    else:
                        attempts = self.matcher.match(window.sequence)
                        # A tie and a total miss both come back as one untouched
                        # attempt, so a null match is the only thing to test.
                        if len(attempts) == 1 and attempts[0].match is not None:
                            attempt = attempts[0]
                        else:
                            no_match += 1

                if attempt is None:
                    ann.set(self.tgidx_name, NO_TARGET)
                else:
                    matched += 1
                    ann.set(self.tgidx_name, attempt.match)
                    # The span bounds the sequence found in the read, not the
                    # entry it verified against; the two coincide only at edit
                    # distance zero. Window coordinates become read coordinates
                    # by adding the window's own start.
                    ann.set(
                        position_key(self.tgidx_name),
                        format_span(
                            window.start + attempt.read_idx[0],
                            window.start + attempt.read_idx[1],
                        ),
                    )
                    target_counts[attempt.match] += 1
                    edit_distance_counts[attempt.edit_distance] += 1

                # The single write every branch above falls through to. Keeping
                # it here, rather than one call per outcome, is what makes "every
                # read emitted exactly once, in input order" a property of the
                # loop's shape instead of four branches each remembering to.
                FastqFile.write_read(tgidx_stream, ann.render(), seq, qual)

        stats = AssignStats(
            total_reads=total,
            matched=matched,
            unmatched_no_match=no_match,
            unmatched_no_umi_pos=no_umi_pos,
            unmatched_short_window=short_window,
            target_counts=dict(target_counts),
            edit_distance_counts=dict(edit_distance_counts),
            homopolymer_base=self.anchor_base,
            homopolymer_run_counts=dict(run_counts),
        )

        with tgidx_stats_path.open("w") as report_file:
            report_file.write(stats.get_report())

        log.info(f"Assigned target indices to {matched}/{total} reads")
        return stats
