include { FASTQC } from '../modules/nf-core/fastqc/main'

ch_fasta_read1    = Channel.from( file(params.fasta_read1) )
ch_fasta_read2    = Channel.from( file(params.fasta_read2) )
ch_fasta_barcodes = Channel.from( file(params.fasta_barcodes) )
meta = [id:'hydrop_scatac_1_S1_R1', group:'hydrop_scatac_1_S1', replicate:1, single_end:false]

workflow CARMACK {
    // Create a channel for fastqc with meta, r1, r2 and bcs
    // ch_fasta_read1.set { ch_fastq }

    ch_fasta_read1.merge ( ch_fasta_read2 )
    .map { read1, read2 -> [meta, [read1, read2]] }
    .set { ch_fastq }

    // EXAMPLE CHANNEL STRUCT: [[META], [READS]]
    // ch_fastq | view

    // Run FASTQC
    ch_versions = Channel.empty()

    FASTQC ( ch_fastq )
    fastqc_html    = FASTQC.out.html              
    fastqc_zip     = FASTQC.out.zip
    ch_versions    = ch_versions.mix(FASTQC.out.versions)

    // EXAMPLE CHANNEL STRUCT: [ val(meta), [ html ] ]
    // fastqc_html | view

    // EXAMPLE CHANNEL STRUCT: [ val(meta), [ zip ] ]
    // fastqc_zip | view

    // EXAMPLE CHANNEL STRUCT: [ versions.yml ]
    // ch_versions | view

    // Run barcode extraction
    ch_fasta_read1.merge ( ch_fasta_read2 )
    .merge ( ch_fasta_barcodes )
    .map { read1, read2, barcodes -> [meta, [read1, read2, barcodes]] }
    .set { ch_bc_ext }

    // EXAMPLE CHANNEL STRUCT: [[META], [READS]]
    ch_bc_ext | view
}