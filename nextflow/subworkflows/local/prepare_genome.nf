/*
 * Uncompress and prepare reference genome files
*/

include { GUNZIP as GUNZIP_FASTA } from '../../modules/nf-core/gunzip/main.nf'
include { SAMTOOLS_FAIDX         } from '../../modules/nf-core/samtools/faidx/main'
include { UNTAR                  } from '../../modules/nf-core/untar/main.nf'
include { BOWTIE2_BUILD          } from '../../modules/nf-core/bowtie2/build/main'

workflow PREPARE_GENOME {
    take:
    prepare_tool_indices // list: tools to prepare indices for
    blacklist            // channel: blacklist file or empty channel

    main:
    ch_versions      = Channel.empty()
    ch_spikein_fasta = Channel.empty()

    /*
    * Uncompress genome fasta file if required
    */
    if (params.fasta.endsWith(".gz")) {
        ch_fasta    = GUNZIP_FASTA ( [ [id:"target_fasta"], params.fasta ] ).gunzip
        ch_versions = ch_versions.mix(GUNZIP_FASTA.out.versions)
    } else {
        ch_fasta = Channel.from( file(params.fasta) ).map { row -> [[id:"spikein_fasta"], row] }
    }

    /*
    * Uncompress GTF annotation file
    */
    // ch_gtf = Channel.empty()
    // if (params.gtf.endsWith(".gz")) {
    //     ch_gtf      = GUNZIP_GTF ( [ [:], params.gtf ] ).gunzip.map { it[1] }
    //     ch_versions = ch_versions.mix(GUNZIP_GTF.out.versions)
    // } else {
    //     ch_gtf = Channel.from( file(params.gtf) )
    // }

    /*
    * Index genome fasta file
    */
    ch_fasta_index = SAMTOOLS_FAIDX ( ch_fasta ).fai
    ch_versions    = ch_versions.mix(SAMTOOLS_FAIDX.out.versions)

    /*
    * Uncompress Bowtie2 index or generate from scratch if required for both genomes
    */
    ch_bt2_index         = Channel.empty()
    ch_bt2_spikein_index = Channel.empty()
    ch_bt2_versions      = Channel.empty()
    if ("bowtie2" in prepare_tool_indices) {
        if (params.bowtie2) {
            if (params.bowtie2.endsWith(".tar.gz")) {
                ch_bt2_index = UNTAR ( [ [], params.bowtie2 ] ).untar.map{ row -> [ [id:"target_index"], row[1] ] }
                ch_versions  = ch_versions.mix(UNTAR.out.versions)
            } else {
                ch_bt2_index = [ [id:"target_index"], file(params.bowtie2) ]
            }
        } else {
            ch_bt2_index = BOWTIE2_BUILD ( ch_fasta ).index.map{ row -> [ [id:"target_index"], row[1] ] }
            ch_versions  = ch_versions.mix(BOWTIE2_BUILD.out.versions)
        }
    }


    emit:
    fasta                  = ch_fasta                    // path: genome.fasta
    fasta_index            = ch_fasta_index              // path: genome.fai
    // gtf                    = ch_gtf                      // path: genome.gtf
    bowtie2_index          = ch_bt2_index                // path: bt2/index/
    versions               = ch_versions                 // channel: [ versions.yml ]

}