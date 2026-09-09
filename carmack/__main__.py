#!/usr/bin/env python
"""carmack: Helper tools for analysis of single-cell mutli-omic data"""

import atexit
import logging
import os
import time

import rich
import rich.console
import rich.logging
import rich.traceback
import rich_click as click

import carmack
from carmack.assign_targets.target_assigner import DEFAULT_MAX_WORKERS, TargetAssigner
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.cell_caller.cell_caller import CellCaller
from carmack.fastq_tools.fastq_filter import FastqFilter
from carmack.prepare_reads.read_preparer import DEFAULT_MAX_WORKERS as PREPARE_READS_DEFAULT_MAX_WORKERS
from carmack.prepare_reads.read_preparer import ReadPreparer
from carmack.split_reads.split_reads import BamSplitter
from carmack.tag_dedup.tag_dedup import TagDedup
from carmack.umi.umi_extractor import UmiExtractor
from carmack.utils import format_duration, get_bai, get_cpu_count

# Set up logging as the root logger
# Submodules should all traverse back to this
log = logging.getLogger()

# # Set up nicer formatting of click cli help messages
click.rich_click.MAX_WIDTH = 120
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.COMMAND_GROUPS = {
    "carmack": [
        {
            "name": "Commands for users",
            "commands": [
                "extract-barcodes",
                "extract-umis",
                "assign-targets",
                "prepare-reads",
                "fastq-filter",
                "bam-tag-deduplicate",
                "call-cells",
            ],
        },
        {
            "name": "Additional utility commands",
            "commands": ["split-bam"],
        },
    ]
}

# Set up rich stderr console
stderr = rich.console.Console(stderr=True)
stdout = rich.console.Console()

# Set up the rich traceback
rich.traceback.install(console=stderr, width=200, word_wrap=True, extra_lines=1)


def run_carmack():
    """
    Print programme header and then use to click for the command line interface.
    """
    # Time logging
    start_time = time.perf_counter()

    def log_runtime() -> None:
        elapsed = time.perf_counter() - start_time
        log.info(f"Wall time: {format_duration(elapsed)}")

    # Register the log_runtime function to be called on exit
    atexit.register(log_runtime)

    # Print carmack header (ANSI Shadow)
    stderr.print("\n\n", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░", highlight=False)
    stderr.print("[white]░░░░░█████╗░░█████╗░██████╗░███╗░░░███╗░█████╗░░█████╗░██╗░░██╗░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██╔══██╗██╔══██╗██╔══██╗████╗░████║██╔══██╗██╔══██╗██║░██╔╝░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██║░░╚═╝███████║██████╔╝██╔████╔██║███████║██║░░╚═╝█████═╝░░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██║░░██╗██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║░░██╗██╔═██╗░░░░░[white]", highlight=False)
    stderr.print("[white]░░░░╚█████╔╝██║░░██║██║░░██║██║░╚═╝░██║██║░░██║╚█████╔╝██║░╚██╗░░░░[white]", highlight=False)
    stderr.print("[white]░░░░░╚════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░░[white]", highlight=False)
    stderr.print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("\n", highlight=False)
    stderr.print(
        f"[grey25]    carmack version {carmack.__version__} - [link=https://github.com/briscoelab/carmack]https://github.com/briscoelab/carmack[/]",
        highlight=False,
    )
    stderr.print("\n", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("\n\n", highlight=False)

    # Launch the click cli
    carmack_cli()


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(carmack.__version__)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print verbose output to the console.")
@click.option("--hide-progress", is_flag=True, default=False, help="Don't show progress bars.")
@click.option("-l", "--log-file", help="Save a verbose log to a file.", metavar="<filename>")
@click.pass_context
def carmack_cli(ctx, verbose, hide_progress, log_file):
    """
    carmack provides helper tools for the analysis of single-cell mutli-omic data.

    This python module enables the extraction of valid cell barcodes and can filter reads with valid barcodes from fastq files.
    """
    # Set the base logger to output DEBUG
    log.setLevel(logging.DEBUG)

    # Set up logs to the console
    log.addHandler(
        rich.logging.RichHandler(
            level=logging.DEBUG if verbose else logging.INFO,
            console=rich.console.Console(stderr=True),
            show_time=False,
            show_path=verbose,  # True if verbose, false otherwise
            markup=True,
        )
    )

    # Set up logs to a file if we asked for one
    if log_file:
        log_fh = logging.FileHandler(log_file, encoding="utf-8")
        log_fh.setLevel(logging.DEBUG)
        log_fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)-20s [%(levelname)s]  %(message)s"))
        log.addHandler(log_fh)

    ctx.obj = {
        "verbose": verbose,
        "hide_progress": hide_progress or verbose,  # Always hide progress bar with verbose logging
    }


@carmack_cli.command("extract-barcodes")
@click.argument("fastq", required=True, nargs=1, type=click.Path(exists=True), metavar="<fastq>", help="Path to FASTQ file")
@click.option("-c", "--chemistry", required=True, type=str, help="Chemistry name for barcode layout")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option("-n", "--cpu_count", required=False, type=int, default=get_cpu_count(), show_default=True, help="Number of CPU workers to use. Default is all available CPUs minus 1.")
@click.option("--fast", is_flag=True, default=False, help="Skip local alignment fallback for faster extraction, at the cost of reduced sensitivity.")
def extract_barcodes(fastq, chemistry, output_dir, prefix, cpu_count, fast):
    """
    Extract cell barcodes from FASTQ reads using hybrid matching strategy.

    Uses a staged extraction pipeline:
    1. Fixed position matching (exact match at expected positions)
    2. Kmer seed-and-extend (handles indels within tolerance)
    3. Local alignment (handles complex errors unless --fast is set)
    """

    log.info("Extracting barcodes from FASTQ file...")
    extractor = BarcodeExtractor(fastq, chemistry, n_workers=cpu_count, fast=fast)
    extractor.extract_barcodes(output_dir, prefix)


@carmack_cli.command("extract-umis")
@click.argument("r1_annotated_fastq", required=True, nargs=1, type=click.Path(exists=True), metavar="<r1_annotated_fastq>")
@click.option("-c", "--chemistry", required=True, type=str, help="Chemistry name for UMI layout")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option("--raw", is_flag=True, default=False, help="Stop after raw extraction; do not correct UMIs.")
@click.option("--temp-dir", required=False, type=click.Path(exists=True, file_okay=False, writable=True), default=None, help="Directory UMI correction spills its shard files to (default: system temp, which may be RAM-backed tmpfs). The spill takes roughly 100 bytes per accepted read (~36 GB at 400M reads), so size the volume before the run.")
@click.option("--shard-count", required=False, type=click.IntRange(1, 1024), default=256, show_default=True, help="Number of shards UMI correction spreads reads over; more shards means lower peak memory, though past about 256 the peak stops improving. Each shard costs an open file descriptor, so a count near the cap needs a raised ulimit.")
def extract_umis(r1_annotated_fastq, chemistry, output_dir, prefix, raw, temp_dir, shard_count):
    """
    Extract raw UMIs from an annotated R1 FASTQ.

    For each annotated read the raw UMI is extracted between its left anchor (BC1, read from the
    header) and the downstream poly-G run, then annotated onto the read with UMI / UMI_POS tags. The
    UMI length distribution is written to a stats report. With --raw this is the terminal step; UMI
    correction is not performed.
    """

    log.info("Extracting UMIs from annotated FASTQ file...")
    extractor = UmiExtractor(r1_annotated_fastq, chemistry)
    extractor.extract_umis(output_dir, prefix, raw=raw, temp_dir=temp_dir, shard_count=shard_count)


@carmack_cli.command("assign-targets")
@click.argument("r1_umi_fastq", required=True, nargs=1, type=click.Path(exists=True), metavar="<r1_umi_fastq>")
@click.option("-c", "--chemistry", required=True, type=str, help="Chemistry name for target index layout")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option("-n", "--cpu_count", required=False, type=int, default=min(DEFAULT_MAX_WORKERS, get_cpu_count()), show_default=True, help="Number of CPU workers to use. Throughput saturates at about 16 workers: past that this stage's own single-threaded parse-and-write loop is the bound and more workers measure no faster, so the default is capped there rather than at every available CPU.")
def assign_targets(r1_umi_fastq, chemistry, output_dir, prefix, cpu_count):
    """
    Assign target indices from a UMI-annotated R1 FASTQ.

    For each annotated read a bounded window is taken off the end of the poly-G run recorded by UMI
    extraction, and the target index inside that window is matched against the chemistry whitelist.
    Every read is re-emitted carrying a TGIDX tag: either a whitelist entry with its TGIDX_POS span,
    or NONE. An unassigned read is an expected outcome rather than a failure - in a mixed library
    NONE is the correct answer for every scRNA read.
    """

    log.info("Assigning target indices from UMI-annotated FASTQ file...")
    assigner = TargetAssigner(r1_umi_fastq, chemistry, n_workers=cpu_count)
    assigner.assign_targets(output_dir, prefix)


@carmack_cli.command("prepare-reads")
@click.argument(
    "r1_annotated_fastq",
    required=True,
    nargs=1,
    type=click.Path(exists=True),
    metavar="<r1_annotated_fastq>",
)
@click.argument(
    "r2_fastq", required=True, nargs=1, type=click.Path(exists=True), metavar="<r2_fastq>"
)
@click.option("-c", "--chemistry", required=True, type=str, help="Chemistry name for read layout")
@click.option(
    "-o",
    "--output_dir",
    required=False,
    type=click.Path(exists=True),
    default=".",
    help="Output directory to save generated files",
)
@click.option(
    "-p",
    "--prefix",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Prefix for generated files",
)
@click.option(
    "-n",
    "--cpu_count",
    required=False,
    type=int,
    default=min(PREPARE_READS_DEFAULT_MAX_WORKERS, get_cpu_count()),
    show_default=True,
    help="Number of CPU workers to use. This stage's own saturation point has not been independently measured; the default provisionally mirrors assign-targets's measured cap, pending benchmarking.",
)
def prepare_reads(r1_annotated_fastq, r2_fastq, chemistry, output_dir, prefix, cpu_count):
    """
    Trim and dispatch every read from an annotated R1 FASTQ into its scRNA or scTIP output.

    The R1 FASTQ (paired with its raw R2) is read in lockstep and every read is trimmed
    down to its genomic/cDNA insert and dispatched to exactly one output. A read carrying
    no real target index - whether its TGIDX tag is NONE or the chemistry supports no
    target index at all - goes to the scRNA arm's three fixed output files, untrimmed
    past its UMI span. A read carrying a real target index is trimmed off the end of its
    TGIDX_POS span and written to that target's own scTIP (R1, R2) file pair. Every input
    read is written exactly once, to exactly one of the two arms.
    """

    log.info("Preparing reads from annotated FASTQ file...")
    preparer = ReadPreparer(r1_annotated_fastq, r2_fastq, chemistry, n_workers=cpu_count)
    preparer.prepare_reads(output_dir, prefix)


@carmack_cli.command("fastq-filter")
@click.argument("read1", required=True, nargs=1, type=click.Path(exists=True), metavar="<read1>")
@click.argument("read2", required=True, nargs=1, type=click.Path(exists=True), metavar="<read2>")
@click.argument("valid_barcodes", required=True, nargs=1, type=click.Path(exists=True), metavar="<valid_barcodes>")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option("--trim-r1", required=False, type=int, default=0, show_default=True, help="Trim this many bases from start of read1 if matched.")
@click.option("--trim-r2", required=False, type=int, default=0, show_default=True, help="Trim this many bases from start of read2 if matched.")
def fastq_filter(read1, read2, valid_barcodes, output_dir, prefix, trim_r1, trim_r2):
    """
    Filter fastq files for reads containing valid barcodes.

    The total set of cell barcodes and valid cell barcodes are saved to separate files in the output directory.
    Additional files containing barcode stats and counts are also saved to the output directory.
    """

    log.info("Filtering fastq files for valid barcodes...")
    fastq_filter = FastqFilter(read1, read2)
    fastq_filter.filter_valid_reads(valid_barcodes, output_dir, prefix, trim_r1=trim_r1, trim_r2=trim_r2)


@carmack_cli.command("bam-tag-deduplicate")
@click.argument("bam", required=True, nargs=1, type=click.Path(exists=True), metavar="<bam>")
@click.argument("bai", required=False, nargs=1, type=click.Path(exists=True), default=None, metavar="<bai>")
@click.argument("valid_barcodes", required=True, nargs=1, type=click.Path(exists=True), metavar="<valid_barcodes>")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-d", "--dedup", is_flag=True, default=False, help="Flag describing whether or not to reads should be deduplicated during barcode tagging")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option("--umi-map", required=False, type=click.Path(exists=True), default=None, help="Corrected UMI map (TSV: read_id, barcode, UR, UB) for UMI-aware tagging and deduplication.")
def bam_tag_deduplicate(bam, bai, valid_barcodes, output_dir, dedup, prefix, umi_map):
    """
    Tag reads with barcodes and deduplicate.

    The reads are tagged with their corresponding barcodes and written to an output BAM file.
    If dedup is set to True, reads are also deduplicated based on the start position, end position and barcode of the read pairs.
    An additional file containing the number of unique and duplicate read pairs is also saved to the output directory.
    If a UMI map is supplied, reads are additionally tagged with their raw (UR) and corrected (UB) UMI and deduplicated on the corrected UMI.
    """
    if bai is None:
        bai = get_bai(bam)

    log.info("Tagging reads with barcodes and deduplicating if requested...")
    tag_dedup = TagDedup(bam, bai, valid_barcodes, umi_map=umi_map)
    tag_dedup.tag_dedup_reads(dedup, output_dir, prefix)


@carmack_cli.command("split-bam")
@click.argument("bam", required=True, nargs=1, type=click.Path(exists=True), metavar="<tagged_bam>")
@click.argument("bai", required=False, nargs=1, type=click.Path(exists=True), default=None, metavar="<bai>")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
@click.option(
    "-n",
    "--cpu_count",
    required=False,
    type=int,
    default=get_cpu_count(),
    show_default=True,
    help="Number of CPUs to use for sorting and indexing of split BAM files. Default is all available CPUs minus 1.",
)
def split_bam(bam, bai, output_dir, prefix, cpu_count):
    """
    Split barcode-tagged BAM file into separate files based on barcode tag (BC) value.

    Reads are split into separate alignment files with each file containing reads with the same barcode tag value.
    The output files are saved to the output directory with corresponding sorted BAM and index BAI files.
    Each file is named according to the barcode tag (BC) value.
    Additionally, creates a CSV file in the output directory containing the barcode counts.
    """
    if bai is None:
        bai = get_bai(bam)

    splitter = BamSplitter(bam, bai)
    splitter.split(output_dir, prefix, cpu_count)


@carmack_cli.command("call-cells")
@click.argument("bed", required=True, nargs=1, type=click.Path(exists=True), metavar="<peaks_bed>")
@click.argument("bam", required=True, nargs=1, type=click.Path(exists=True), metavar="<tagged_bam>")
@click.argument("bai", required=False, nargs=1, type=click.Path(exists=True), default=None, metavar="<bai>")
@click.option("-c", "--force_n", required=False, type=int, default=None, help="Force selection of top n cells")
@click.option("-m", "--min_overlap", required=False, type=int, default=1, show_default=True, help="Minimum number of basepairs overlapping a peak to be considered")
@click.option("-g", "--visualise", is_flag=True, default=False, help="Save barcode rank plot with threshold")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default=None, show_default=True, help="Prefix for generated files")
def call_cells(bed, bam, bai, force_n, min_overlap, visualise, output_dir, prefix):
    """
    Filter and export cells and peaks to standard single-cell format based on the number of
    overlapping peaks per cell.

    All instances of peak-barcode overlaps are counted and saved to a matrix, which is then used to
    filter cells based on knee point, or a fixed number of cells with most overlaps (if force_n is
    set). The output files (barcodes, peaks and peak-barcode matrix) are saved to the output
    directory. If visualise is set, a plot of the barcode rank is saved to the output directory.
    """
    log.info("Calling cells based on peak overlaps...")
    if bai is None:
        bai = get_bai(bam)

    cell_caller = CellCaller(bed, bam, bai)
    cell_caller.compute_matrix(min_overlap=min_overlap)

    if visualise:
        plot = cell_caller.make_plot(force_n=force_n)
        prefix = f"{prefix}_" if prefix else ""
        plot.savefig(os.path.join(output_dir, f"{prefix}barcode_matrix.png"))

    cell_caller.export(output_dir, prefix, force_n)


# Main script is being run - launch the CLI
if __name__ == "__main__":
    run_carmack()
