#!/usr/bin/env python
""" carmack: Helper tools for analysis of single-cell mutli-omic data """
import logging
import os
import sys

import rich
import rich.console
import rich.logging
import rich.traceback
import rich_click as click

import carmack
from carmack.barcode.barcode_extractor import BarcodeExtractor
from carmack.fastq_tools.fastq_filter import FastqFilter
from carmack.duplicate_removal.remove_duplicates import DuplicateRemoval

# Set up logging as the root logger
# Submodules should all traverse back to this
log = logging.getLogger()

# # Set up nicer formatting of click cli help messages
click.rich_click.MAX_WIDTH = 120
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.COMMAND_GROUPS = {
    "carmack": [
        {
            "name": "Commands for users",
            "commands": ["extract-cell-barcodes", "fastq-filter", "bam-tag-deduplicate"],
        }
    ]
}
# click.rich_click.OPTION_GROUPS = {
#     "carmack extract-cell-barcodes": [{"options": ["--chemistry", "--maxdist", "--line_count", "--output_dir", "--prefix"]}],
#     "carmack fastq-filter": [{"options": ["--output_dir", "--prefix"]}]
# }

# Set up rich stderr console
stderr = rich.console.Console(stderr=True)
stdout = rich.console.Console()

# Set up the rich traceback
rich.traceback.install(console=stderr, width=200, word_wrap=True, extra_lines=1)


def run_carmack():
    """
    Print programme header and then use to click for the command line interface.
    """

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
        log_fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)-20s [%(levelname)-7s]  %(message)s"))
        log.addHandler(log_fh)

    ctx.obj = {
        "verbose": verbose,
        "hide_progress": hide_progress or verbose,  # Always hide progress bar with verbose logging
    }

@carmack_cli.command("extract-cell-barcodes")
@click.argument("read1", required=True, nargs=1, type=click.Path(exists=True), metavar="<read1>")
@click.argument("read2", required=True, nargs=1, type=click.Path(exists=True), metavar="<read2>")  
@click.argument("barcodes", required=True, nargs=1, type=click.Path(exists=True), metavar="<barcodes>")  
@click.option("-c","--chemistry", required=True, type=str, help="Chemistry class for barcode extraction")
@click.option("-d", "--max_dist", required=True, type=int, help="Maximal Hamming distance for barcode extraction")
@click.option("-s", "--print_stats", is_flag=True, default=False, help="Flag describing whether or not to print summary stats during barcode extraction")
@click.option("-l", "--log_freq", required=False, type=int, default=10000, help="Number of lines after which stats are logged during barcode extraction")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")   
@click.option("-p", "--prefix", required=False, type=str, default="", show_default=True, help="Prefix for generated files")
def extract_cell_barcodes(read1, read2, barcodes, chemistry, max_dist, print_stats, log_freq, output_dir, prefix): 
    """
    Extracts valid cell barcodes by correcting for indels and sequencing errors, using a specified maximal Hamming distance and barcode chemistry.

    The total set of cell barcodes and valid cell barcodes are saved to separate files in the output directory. 
    Additional files containing barcode stats and counts are also saved to the output directory. 
    """
    
    barcode_ext = BarcodeExtractor(read1, read2, barcodes, chemistry)
    barcode_ext.extract_cell_barcodes(max_dist, print_stats, log_freq, output_dir, prefix)


@carmack_cli.command("fastq-filter")
@click.argument("read1", required=True, nargs=1, type=click.Path(exists=True), metavar="<read1>")
@click.argument("read2", required=True, nargs=1, type=click.Path(exists=True), metavar="<read2>")  
@click.argument("valid_barcodes", required=True, nargs=1, type=click.Path(exists=True), metavar="<valid_barcodes>")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files")
@click.option("-p", "--prefix", required=False, type=str, default="", show_default=True, help="Prefix for generated files")
def fastq_filter(read1, read2, valid_barcodes, output_dir, prefix):
    """
    Filter fastq files for reads containing valid barcodes.

    The total set of cell barcodes and valid cell barcodes are saved to separate files in the output directory.
    Additional files containing barcode stats and counts are also saved to the output directory.
    """

    fastq_filter = FastqFilter(read1, read2)
    fastq_filter.filter_valid_reads(valid_barcodes, output_dir, prefix)

@carmack_cli.command("bam-tag-deduplicate")
@click.argument("bam", required=True, nargs=1, type=click.Path(exists=True), metavar="<bam>")
@click.argument("bai", required=True, nargs=1, type=click.Path(exists=True), metavar="<bai>")  
@click.argument("valid_barcodes", required=True, nargs=1, type=click.Path(exists=True), metavar="<valid_barcodes>")
@click.option("-l", "--log_progress", is_flag=True, default=False, help="Flag describing whether or not to log progress during barcode tagging and deduplication")
@click.option("-o", "--output_dir", required=False, type=click.Path(exists=True), default=".", help="Output directory to save generated files") 
@click.option("-d", "--dedup", is_flag=True, default=False, help="Flag describing whether or not to reads should be deduplicated during barcode tagging")
@click.option("-p", "--prefix", required=False, type=str, default="", show_default=True, help="Prefix for generated files")
def bam_tag_deduplicate(bam, bai, valid_barcodes, log_progress, output_dir, dedup, prefix):
    """
    Tag reads with barcodes and deduplicate.

    The reads are tagged with their corresponding barcodes and written to an output BAM file. 
    If dedup is set to True, reads are also deduplicated based on the start position, end position and barcode of the read pairs.
    An additional file containing the number of unique and duplicate read pairs is also saved to the output directory.
    """

    duplicate_rem = DuplicateRemoval(bam, bai, valid_barcodes)
    duplicate_rem.tag_and_deduplicate_reads(log_progress, output_dir, dedup, prefix)
    

# Main script is being run - launch the CLI
if __name__ == "__main__":
    run_carmack()
