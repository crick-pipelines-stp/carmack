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

if (params.samplesheet) { ch_input = Channel.from( file(params.samplesheet) ) } else { exit 1, 'Input samplesheet not specified!' }

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
include { EXTRACT_BARCODES } from './modules/local/python/extract_barcodes'
include { FILTER_FASTQ     } from './modules/local/python/filter_fastq'

/*
* SUBWORKFLOWS
*/
include { PREPARE_GENOME    } from './subworkflows/local/prepare_genome'
include { FASTQC_TRIMGALORE } from './subworkflows/local/fastqc_trimgalore'

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
include { BEDTOOLS_BAMTOBED } from './modules/nf-core/bedtools/bamtobed/main'

/*
* SUBWORKFLOWS
*/
include { BAM_SORT_STATS_SAMTOOLS          } from './subworkflows/nf-core/bam_sort_stats_samtools/main'

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
            it -> [[id:it.sample_id], file(it.fastq_1), file(it.fastq_2), file(it.cell_barcodes)]
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

    ch_reads = ch_input_parsed.map { it -> [it[0], it[1], it[2]] }
    // EXAMPLE CHANNEL STRUCT: [META, READ1, READ2]
    // ch_reads | view

    /*
     * MODULE: Filter valid reads
     */
    FILTER_FASTQ (
        ch_reads,
        EXTRACT_BARCODES.out.valid
    )
    // EXAMPLE CHANNEL STRUCT: [META, [VALID_READ1, VALID_READ2]]
    // FILTER_FASTQ.out.valid_reads | view

    /*
     * SUBWORKFLOW: Read QC, trim adapters and perform post-trim read QC
     */
    FASTQC_TRIMGALORE (
        FILTER_FASTQ.out.valid_reads, 
        params.skip_fastqc, 
        params.skip_trimming
    )
    ch_software_versions = ch_software_versions.mix(FASTQC_TRIMGALORE.out.versions)
    ch_trimmed_reads     = FASTQC_TRIMGALORE.out.reads
    // ch_trimmed_reads | view

    ch_index = ch_bowtie2_index.map { [[id:it.baseName], it] }
    // ch_index | view
    
    /*
     * MODULE: Map reads with BOWTIE2 to target genome
     */
    BOWTIE2_ALIGN (
        ch_trimmed_reads, 
        ch_index.collect{ it[1] },
        params.save_unaligned,
        false
    )
    ch_software_versions = ch_software_versions.mix(BOWTIE2_ALIGN.out.versions)
    // BOWTIE2_ALIGN.out.bam | view

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

