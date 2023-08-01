
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
 * SUBWORKFLOWS
 */
include { PREPARE_GENOME } from "../subworkflows/local/prepare_genome"

/*
========================================================================================
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
========================================================================================
*/

/*
 * MODULES
 */
include { FASTQC            } from '../modules/nf-core/fastqc/main'
include { BOWTIE2_ALIGN     } from '../modules/nf-core/bowtie2/align/main'
include { BEDTOOLS_BAMTOBED } from "../modules/nf-core/bedtools/bamtobed/main"


/*
 * SUBWORKFLOWS
 */
include { BAM_SORT_STATS_SAMTOOLS } from '../subworkflows/nf-core/bam_sort_stats_samtools/main'

// Init channels
ch_fastq_read1    = Channel.from( file(params.fastq_read1) )
ch_fastq_read2    = Channel.from( file(params.fastq_read2) )
ch_fastq_barcodes = Channel.from( file(params.fastq_barcodes) )

ch_fastq_valid_read1 = Channel.from( file(params.fastq_valid_read1) )
ch_fastq_valid_read2 = Channel.from( file(params.fastq_valid_read2) )

ch_tagged_bam = Channel.from( file(params.tagged_bam) )

meta = [id:'hydrop_scatac_1_S1_R1', group:'hydrop_scatac_1_S1', replicate:1, single_end:false]

workflow CARMACK {
    // Init
    ch_software_versions = Channel.empty()

    /*
     * SUBWORKFLOW: Uncompress and prepare reference genome files
     */
    PREPARE_GENOME (
        prepare_tool_indices,
        ch_blacklist
    )
    ch_software_versions = ch_software_versions.mix(PREPARE_GENOME.out.versions)
    ch_bowtie2_index     = PREPARE_GENOME.out.bowtie2_index

    // EXAMPLE CHANNEL STRUCT: [ val(meta), [ path(bt2/index/) ] ]
    // ch_bowtie2_index | view
    
    // Create a channel for fastqc with meta, r1, r2 and bcs
    ch_fastq_read1.merge ( ch_fastq_read2 )
    .map { read1, read2 -> [meta, [read1, read2]] }
    .set { ch_fastq }

    // EXAMPLE CHANNEL STRUCT: [[META], [READS]]
    // ch_fastq | view

    // Run FASTQC
    FASTQC ( ch_fastq )
    fastqc_html          = FASTQC.out.html              
    fastqc_zip           = FASTQC.out.zip
    ch_software_versions = ch_software_versions.mix(FASTQC.out.versions)

    // EXAMPLE CHANNEL STRUCT: [ val(meta), [ html ] ]
    // fastqc_html | view

    // EXAMPLE CHANNEL STRUCT: [ val(meta), [ zip ] ]
    // fastqc_zip | view

    // EXAMPLE CHANNEL STRUCT: [ versions.yml ]
    // ch_software_versions | view

    // Create barcode extraction channel
    ch_fastq_read1.merge ( ch_fastq_read2 )
    .merge ( ch_fastq_barcodes )
    .map { read1, read2, barcodes -> [meta, [read1, read2, barcodes]] }
    .set { ch_bc_ext }

    // Run barcode extraction
    // ...

    // EXAMPLE CHANNEL STRUCT: [[META], [READS]]
    // ch_bc_ext | view

    // Run fastq filter
    // ...

    // Trim reads by running trimgalore (or fastq_trimgalore subworkflow)
    // ...

    // Run BOWTIE2
    ch_fastq_valid_read1.merge ( ch_fastq_valid_read2 )
    .map { valid_read1, valid_read2 -> [meta, [valid_read1, valid_read2]] }
    .set { ch_valid_reads }
    // ch_valid_reads | view

    ch_index = ch_bowtie2_index.map { [[id:it.baseName], it] }
    ch_index | view
    
    BOWTIE2_ALIGN (
        ch_valid_reads,
        ch_index.collect{ it[1] },
        params.save_unaligned,
        false
    )
    ch_software_versions = ch_software_versions.mix(BOWTIE2_ALIGN.out.versions)
    // ch_software_versions | view
    // BOWTIE2_ALIGN.out.bam | view

    // Sort, index BAM file and run samtools stats, flagstat and idxstats
    // (maybe should fo after tagging and duplicate removal as sorting destroys read pair order)
    BAM_SORT_STATS_SAMTOOLS ( 
        BOWTIE2_ALIGN.out.bam, 
        PREPARE_GENOME.out.fasta.collect{ it[1] } )
    ch_software_versions = ch_software_versions.mix(BAM_SORT_STATS_SAMTOOLS.out.versions)

    // BAM_SORT_STATS_SAMTOOLS.out.bam | view
    // BAM_SORT_STATS_SAMTOOLS.out.bai | view

    // Tagging bam files with cell barcodes
    // ...

    // Run bedtools bam_to_bed
    // ch_tagged_bam with meta
    ch_tagged_bam.map { bam -> [meta, bam] }
    .set { ch_tagged_bam_meta }
    ch_tagged_bam_meta| view
    // [[id:hydrop_scatac_1_S1_R1, group:hydrop_scatac_1_S1, replicate:1, single_end:false], /Users/hodgett/dev/repos/carmack/nextflow/tests/data/output.bam]

    BEDTOOLS_BAMTOBED (
        ch_tagged_bam_meta
        // SAMTOOLS_SORT.out.bam
    )
    // BEDTOOLS_BAMTOBED.out.bed | view

}