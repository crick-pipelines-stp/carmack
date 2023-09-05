process FILTER_FASTQ {
    label 'process_medium'

    conda "conda-forge::python=3.8.3"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-f42a44964bca5225c7860882e231a7b5488b5485:47ef981087c59f79fdbcab4d9d7316e9ac2e688d-0' :
        'quay.io/biocontainers/mulled-v2-f42a44964bca5225c7860882e231a7b5488b5485:47ef981087c59f79fdbcab4d9d7316e9ac2e688d-0' }"

    input:
    tuple val(meta), path(read1), path(read2)
    path valid_barcodes
    
    output:
    path '*.r1_valid.fastq.gz', emit: read1_valid
    path '*.r2_valid.fastq.gz', emit: read2_valid
    path  "versions.yml"      , emit: versions

    script:
    def prefix = task.ext.prefix ?: ''
    """
    fastq_filter.py $read1 $read2 $valid_barcodes \\
        -o $params.outdir \\
        -p $prefix 

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | grep -E -o \"([0-9]{1,}\\.)+[0-9]{1,}\")
    END_VERSIONS
    """
}