process BAM_TAG_DEDUP {
    label 'process_medium'

    container "docker.io/luslab/carmack:latest"

    input:
    tuple val(meta), path(bam)
    tuple val(meta), path(bai)
    path valid_barcodes
    
    output:
    path "*.tagged.bam" , emit: bam
    path "*_mqc.log"    , emit: mqc
    path  "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: ''
    """
    python -m carmack bam-tag-deduplicate $bam $bai $valid_barcodes \\
        -o . \\
        -p $prefix \\
        $args 

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | grep -E -o \"([0-9]{1,}\\.)+[0-9]{1,}\")
    END_VERSIONS
    """
}