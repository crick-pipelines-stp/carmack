#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
========================================================================================
    VALIDATE INPUTS
========================================================================================
*/

if (params.blacklist) {
    ch_blacklist = Channel.from( file(params.blacklist) )
}
else {
    ch_blacklist = Channel.empty()
}

if (params.samplesheet) { ch_input = Channel.from( file(params.samplesheet) ) } else { exit 1, "Input samplesheet not specified!" }

/*
========================================================================================
    INIALISE PARAMETERS AND OPTIONS
========================================================================================
*/

// Init aligners
def prepare_tool_indices = ["bowtie2"]

/*
========================================================================================
    IMPORT LOCAL MODULES/SUBWORKFLOWS
========================================================================================
*/

/*
* MODULES
*/
include { EXTRACT_BARCODES } from "./modules/local/python/extract_barcodes"
include { FILTER_FASTQ     } from "./modules/local/python/filter_fastq"

/*
* SUBWORKFLOWS
*/
include { PREPARE_GENOME } from "./subworkflows/local/prepare_genome"

/*
========================================================================================
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
========================================================================================
*/

/*
* MODULES
*/
include { FASTQC            } from './modules/nf-core/fastqc/main'
include { BOWTIE2_ALIGN     } from './modules/nf-core/bowtie2/align/main'
include { BEDTOOLS_BAMTOBED } from "./modules/nf-core/bedtools/bamtobed/main"

/*
* SUBWORKFLOWS
*/
include { BAM_SORT_STATS_SAMTOOLS          } from './subworkflows/nf-core/bam_sort_stats_samtools/main'
include { FASTQ_FASTQC_UMITOOLS_TRIMGALORE } from './subworkflows/nf-core/fastq_fastqc_umitools_trimgalore/main'

/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {
    // Init
    ch_software_versions = Channel.empty()

     // Parse samplesheet
    ch_input_parsed = ch_input.splitCsv ( header:true, sep:"," )
        .map { 
            it -> [[id:it.id], file(it.fastq_1), file(it.fastq_2), file(it.cell_barcodes)]
        }
    // EXAMPLE CHANNEL STRUCT: [META, READ1, READ2, BARCODES]
    // ch_input_parsed | view

    /*
     * SUBWORKFLOW: Uncompress and prepare reference genome files
     */
    PREPARE_GENOME (
        prepare_tool_indices,
        ch_blacklist
    )
    ch_software_versions = ch_software_versions.mix(PREPARE_GENOME.out.versions)
    ch_bowtie2_index     = PREPARE_GENOME.out.bowtie2_index
    // EXAMPLE CHANNEL STRUCT: [META, [INDEX] ]
    // ch_bowtie2_index | view

    /*
     * MODULE: Extract cell barcodes
     */
    EXTRACT_BARCODES (
        ch_input_parsed, 
        params.max_corrections, 
        params.chemistry 
    )

    ch_reads = ch_input_parsed.map { it -> [it[0], [it[1][0], it[1][1]]] }
    // EXAMPLE CHANNEL STRUCT: [META, [READ1, READ2]]
    // ch_reads | view

    /*
     * MODULE: Filter valid reads
     */
    // FILTER_FASTQ (
    //     ch_reads,
    //     EXTRACT_BARCODES.valid
    // )

    // Channnel with valid reads (after implementing EXTRACT_BARCODES and FILTER_FASTQ)
    // ch_valid_reads = FILTER_FASTQ.out.read1_valid.merge ( FILTER_FASTQ.out.read2_valid )
    //     .map { valid_read1, valid_read2 -> [meta, [valid_read1, valid_read2]] }

    // EXAMPLE CHANNEL STRUCT: [META, [VALID_READ1, VALID_READ1]]
    // ch_valid_reads | view

    /*
     * SUBWORKFLOW: Read QC, extract UMI and trim adapters with TrimGalore!
     */
    ch_trimmed_reads       = Channel.empty()
    ch_fastqc_raw_multiqc  = Channel.empty()
    ch_fastqc_trim_multiqc = Channel.empty()
    ch_trim_log_multiqc    = Channel.empty()
    ch_trim_read_count     = Channel.empty()
    // FASTQ_FASTQC_UMITOOLS_TRIMGALORE ( 
    //     // ch_valid_reads // TODO: Set only valid reads (ch_valid_reads) as input in FASTQ_FASTQC_UMITOOLS_TRIMGALORE, not ch_reads
    //     params.skip_fastqc,
    //     params.with_umi,
    //     params.skip_umi_extract,
    //     params.skip_trimming,
    //     params.umi_discard_read,
    //     params.min_trimmed_reads
    //  )
    // ch_valid_trimmed_reads = FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.reads
    // ch_fastqc_raw_multiqc  = FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.fastqc_zip 
    // ch_fastqc_trim_multiqc = FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.trim_zip
    // ch_trim_log_multiqc    = FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.trim_log
    // ch_trim_read_count     = FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.trim_read_count
    // ch_software_versions   = ch_software_versions.mix(FASTQ_FASTQC_UMITOOLS_TRIMGALORE.out.versions)

    // ch_index = ch_bowtie2_index.map { [[id:it.baseName], it] }
    // ch_index | view
    
    // // Run BOWTIE2_ALIGN
    // BOWTIE2_ALIGN (
    //     ch_valid_trimmed_reads, // valid trimmed reads
    //     ch_index.collect{ it[1] },
    //     params.save_unaligned,
    //     false
    // )
    // ch_software_versions = ch_software_versions.mix(BOWTIE2_ALIGN.out.versions)
    // // ch_software_versions | view
    // // BOWTIE2_ALIGN.out.bam | view

    // // Sort, index BAM file and run samtools stats, flagstat and idxstats
    // // (maybe should do after barcode tagging and duplicate removal as sorting destroys read pair order)
    // BAM_SORT_STATS_SAMTOOLS ( 
    //     BOWTIE2_ALIGN.out.bam, 
    //     PREPARE_GENOME.out.fasta.collect{ it[1] } )
    // ch_software_versions = ch_software_versions.mix(BAM_SORT_STATS_SAMTOOLS.out.versions)

    // // BAM_SORT_STATS_SAMTOOLS.out.bam | view
    // // BAM_SORT_STATS_SAMTOOLS.out.bai | view

    // // Filtering reads
    // // ...

    // // Tagging bam files with cell barcodes
    // // ...

    // // Deduplication
    // // ...

    // // Run bedtools bam_to_bed
    // // ch_tagged_bam with meta
    // // ch_tagged_bam_meta = ch_tagged_bam.map { bam -> [meta, bam] } (og channel from file in tests/data)
    // // ch_tagged_bam_meta| view
    // // [[id:hydrop_scatac_1_S1_R1, group:hydrop_scatac_1_S1, replicate:1, single_end:false], /Users/hodgett/dev/repos/carmack/nextflow/tests/data/output.bam]

    // BEDTOOLS_BAMTOBED (
    //     ch_tagged_bam_meta
    //     // SAMTOOLS_SORT.out.bam
    // )
    // // BEDTOOLS_BAMTOBED.out.bed | view

    // // Run MULTIQC (incl. ch_fastqc_raw_multiqc, ch_fastqc_trim_multiqc and ch_trim_log_multiqc, see rnaseq)

}

