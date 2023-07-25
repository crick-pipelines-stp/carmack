
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
include { FASTQC        } from '../modules/nf-core/fastqc/main'
include { BOWTIE2_ALIGN } from '../modules/nf-core/bowtie2/align/main'


// Init channels
ch_fastq_read1    = Channel.from( file(params.fastq_read1) )
ch_fastq_read2    = Channel.from( file(params.fastq_read2) )
ch_fastq_barcodes = Channel.from( file(params.fastq_barcodes) )
// ch_fasta          = Channel.from( file(params.fasta) )
// ch_bowtie2        = Channel.from( file(params.bowtie2) )

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
    ch_bowtie2_index | view
    
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
    // ch_versions | view

    // Run barcode extraction
    ch_fastq_read1.merge ( ch_fastq_read2 )
    .merge ( ch_fastq_barcodes )
    .map { read1, read2, barcodes -> [meta, [read1, read2, barcodes]] }
    .set { ch_bc_ext }

    // EXAMPLE CHANNEL STRUCT: [[META], [READS]]
    ch_bc_ext | view

    // Run fastq filter

    // Run bowtie2
    // BOWTIE2_ALIGN (
    //     // filtered_reads,
    //     // ch_bowtie_index.collect{ it[1] },
    //     // params.save_unaligned,
    //     // false
    // )
    // ch_versions = ch_versions.mix(BOWTIE2_ALIGN.out.versions)

}