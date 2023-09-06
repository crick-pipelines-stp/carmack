process EXTRACT_BARCODES {
    label 'process_medium'

    container "docker.io/luslab/carmack:latest"

    input:
    tuple val(meta), path(read1), path(read2), path(barcodes)
    val max_corrections
    val chemistry
    
    output:
    path '*.bc_all.csv'         , emit: all
    path '*.bc_valid.csv'       , emit: valid
    path '*.bc_counts_stats.csv', emit: bc_stats
    path '*.bc_counts.csv'      , emit: bc_counts
    path  "versions.yml"        , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: ''
    """
    python -m carmack extract-cell-barcodes $read1 $read2 $barcodes \\
        -c $chemistry \\
        -d $max_corrections \\
        $args \\
        -o . \\
        -p $prefix

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | grep -E -o \"([0-9]{1,}\\.)+[0-9]{1,}\")
    END_VERSIONS
    """
}