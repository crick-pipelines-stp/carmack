process FILTER_FASTQ {
    label 'process_medium'

    container "docker.io/luslab/carmack:latest"

    input:
    tuple val(meta), path(read1), path(read2)
    path valid_barcodes
    
    output:
    tuple val(meta), path("*_valid.fastq.gz"), emit: valid_reads
    path  "versions.yml"                     , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: ''
    """
    python -m carmack fastq-filter $read1 $read2 $valid_barcodes \\
        -o . \\
        -p $prefix \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | grep -E -o \"([0-9]{1,}\\.)+[0-9]{1,}\")
    END_VERSIONS
    """
}