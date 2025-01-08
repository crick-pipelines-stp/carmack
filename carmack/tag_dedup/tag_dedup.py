import os
import logging
import pysam
import csv
from typing import Optional, List, Tuple
from contextlib import ExitStack
from tqdm import tqdm
from carmack.utils import get_prefix

log = logging.getLogger(__name__)


class TagDedup:
    """
    Class that tags reads in BAM file with barcode and optionally removes
    duplicates.
    """

    def __init__(self, bam: str, bai: str, bc_valid_csv: str) -> None:
        self.bam = bam
        self.bai = bai
        self.bc_valid_csv = bc_valid_csv

        log.debug(
            f"TagDedup object created with BAM: {bam}, BAI: {bai},"
            f" and valid barcodes CSV: {bc_valid_csv}"
        )

    def _progressbar(self, alignment_file: pysam.AlignmentFile, **kwargs):
        """
        Curry tqdm to work with pysam.AlignmentFile objects
        """
        return tqdm(
            alignment_file.fetch(),
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} reads",
            total=alignment_file.count(),
            **kwargs,
        )

    def _tag(
        self, untagged_bam: pysam.AlignmentFile, tagged_bam: pysam.AlignmentFile, bc_dict: dict
    ) -> None:
        """
        Tag reads in untagged BAM file with barcode, duplicates and write to
        tagged BAM file.
        """
        log.info(
            f"Starting to tag reads in BAM file. Total reads to process: "
            f"{untagged_bam.count()}"
        )
        tag_count: int = 0

        # [(chromosome, start, template_len, barcode), ...]
        dup_index: List[Tuple[str, int, int, str]] = []

        for read in self._progressbar(untagged_bam, desc="Tagging reads"):
            read_name = read.query_name

            # Check if BC tag already exists
            if read.has_tag("BC"):
                log.error(
                    f"Input BAM file read with BC tag detected (read name: {read_name}). Exiting."
                )
                raise ValueError("Input BAM file reads already has 'BC' tags.")

            # Log unpaired reads
            if not read.is_paired:
                log.warning(f"Read {read_name} is not paired.")

            # Check if read has a barcode
            if read_name not in bc_dict:
                log.warning(
                    f"Read {read_name} does not have a barcode in barcodes CSV",
                    " and will be skipped.",
                )
                continue

            # Get barcode and tag read
            barcode = bc_dict[read_name]
            read.set_tag("BC", barcode)
            log.debug(
                f"Tagged read {read_name} (paired: {read.is_paired})" f" with barcode {barcode}"
            )

            # Tag duplicates
            chr = read.reference_name
            start = read.reference_start
            seq_len = read.template_length  # Don't want absolute value

            if (chr, start, seq_len, barcode) in dup_index:
                # Only a duplicate if chr, start, seq_len and barcode match
                log.debug(f"Duplicate read detected: {read_name}")
                read.set_tag("DU", True)
            else:
                read.set_tag("DU", False)
                dup_index.append((chr, start, seq_len, barcode))

            # Write tagged read to tagged BAM file
            tagged_bam.write(read)
            tag_count += 1

        log.info(f"Finished tagging reads. Total reads tagged:" f"{tag_count}")

    def _dedup(
        self,
        tagged_bam: pysam.AlignmentFile,
        dedup_bam: pysam.AlignmentFile,
        multiqc_log: Optional[csv.writer] = None,
    ):
        """
        Filter reads in tagged BAM file to remove duplicates.
        """
        log.info(
            f"Starting to deduplicate reads in tagged BAM file. Total "
            f"reads to process: {tagged_bam.count()}"
        )

        unique_count: int = 0

        def get_dup_tag(read: pysam.AlignedSegment) -> bool:
            """
            Check and get duplicate tag from read.
            """
            if not read.has_tag("DU"):
                log.error("Read does not have a DU tag. Exiting.")
                raise ValueError("Read does not have a DU tag.")

            dup_tag = read.get_tag("DU")
            if not isinstance(dup_tag, int):
                log.error("Read has an invalid DU tag (non-int). Exiting.")
                raise ValueError("Read has an invalid DU tag (non-int).")

            if dup_tag not in [0, 1]:
                log.error("Read has an invalid DU tag (not 0 or 1). Exiting.")
                raise ValueError("Read has an invalid DU tag (not 0 or 1).")

            # Return True if duplicate, False if not
            return dup_tag == 1

        for read in self._progressbar(tagged_bam, desc="Deduplicating reads"):
            read_dup = get_dup_tag(read)

            if not read_dup:
                dedup_bam.write(read)
                unique_count += 1

        # MultiQC log
        # Two reads for one read-pair
        if multiqc_log is not None:
            multiqc_log.writerow(["unique_reads", "duplicate_reads"])
            multiqc_log.writerow([unique_count, tagged_bam.count() - unique_count])

    def tag_dedup_reads(self, dedup: bool, output_dir: str, prefix: Optional[str] = None) -> None:
        """
        Generate a tagged BAM file and TSV file with filtered reads coordinates
        with associated barcodes. If `dedup = TRUE`, also generate a
        deduplicated BAM file from the tagged BAM file and log duplicate counts.
        """
        # Set prefix, if non specified
        if prefix is None:
            prefix = get_prefix(self.bam)

        # Init
        BAM_TAGGED_PATH = os.path.join(output_dir, prefix + ".tagged.bam")
        BAM_DEDUP_PATH = os.path.join(output_dir, prefix + ".dedup.tagged.bam")
        MULTIQC_CSV_PATH = os.path.join(output_dir, prefix + ".dedup.stats_mqc.log")

        # Read barcodes from CSV
        with open(self.bc_valid_csv, "r") as valid_barcodes:
            csv_reader = csv.reader(valid_barcodes)
            bc_dict = {line[0].split(" ", 1)[0]: line[1] for line in csv_reader}
        log.debug(f"Loaded {len(bc_dict)} valid barcodes.")

        # Tag and write to tagged BAM file
        with ExitStack() as stack:
            input_bam = stack.enter_context(
                pysam.AlignmentFile(self.bam, "rb", index_filename=self.bai)
            )
            bam_tagged = stack.enter_context(
                pysam.AlignmentFile(BAM_TAGGED_PATH, "wb", header=input_bam.header)
            )

            # Tag
            self._tag(input_bam, bam_tagged, bc_dict)

        pysam.index(BAM_TAGGED_PATH)
        log.info(
            f"Tagged BAM file written to {BAM_TAGGED_PATH} with a corresponding BAI index file."
        )

        if not dedup:
            return

        # Deduplicate and write to deduplicated BAM file
        with ExitStack() as stack:
            bam_tagged = stack.enter_context(pysam.AlignmentFile(BAM_TAGGED_PATH, "rb"))
            bam_dedup = stack.enter_context(
                pysam.AlignmentFile(BAM_DEDUP_PATH, "wb", header=bam_tagged.header)
            )
            multiqc_csv = stack.enter_context(open(MULTIQC_CSV_PATH, mode="w", newline=""))
            multiqc_csv_writer = csv.writer(multiqc_csv)

            # Deduplicate
            self._dedup(bam_tagged, bam_dedup, multiqc_csv_writer)

        pysam.index(BAM_DEDUP_PATH)
        log.info(
            f"Deduplicated BAM file written to {BAM_DEDUP_PATH} with "
            "a corresponding BAI index file."
        )
        log.info(f"MultiQC log written to {MULTIQC_CSV_PATH}.")
