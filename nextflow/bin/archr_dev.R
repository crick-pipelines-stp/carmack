#!/usr/bin/env Rscript

################################################
################################################
## Functions                                  ##
################################################
################################################
# ...

################################################
################################################
## Parse nextflow and CL params               ##
################################################
################################################
# opt <- list(
#   cores = "!{cores}",    # Number of cores to use
#   bam_file = "!{bam}",   # The bam file
#   prefix = "!{prefix}",  # Output file prefix
#   genome = "!{genome}"
# )

# checks ...

################################################
################################################
## Finish loading libraries                   ##
################################################
################################################
# library(ArchR)
# ArchR::installExtraPackages() 
library(devtools)
library(BiocManager)
library(Rsamtools)
library(irlba)
# library(Cairo)

################################################
################################################
## Load and prepare data                      ##
################################################
################################################

# Set the number of threads
# addArchRThreads(threads = opt$cores) 
addArchRThreads(threads = 2) 

# Set genome
# addArchRGenome(opt$genome)
addArchRGenome("hg38")

# Load the input data
inputFiles <- list.files('.', pattern = '*.bam$', full.names = T)
# inputFiles

################################################
################################################
## ArchR analysis                             ##
################################################
################################################


## Create arrow files
arrow_files <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = "hydrop_scatac_1_S1_R1_001",
  minTSS = 0, # Can always increase this later
  minFrags = 1, 
  addTileMat = TRUE,
  addGeneScoreMat = FALSE # Originally TRUE (produces metadata), but set to false as caused errors
  # evt. Set to true only if more than 1 sample
)


## Inspect arrow files
arrow_files
# "hydrop_scatac_1_S1_R1_001.arrow"
# rm(arrow_files)

## Per-cell QC
# Above command outomatically generates a QualityControl folder with 
# a Fragment_Size_Distribution.pdf and a TSS_by_Unique_Frags.pdf file

## Doublet inference (should only be run for droplet-based single-cell chemistries e.g. hydrop)
doubScores <- addDoubletScores(
  input = arrow_files,
  k = 10, #Refers to how many cells near a "pseudo-doublet" to count.
  knnMethod = "UMAP", #Refers to the embedding to use for nearest neighbor search with doublet projection.
  LSIMethod = 1
)

# Alternative settings suggested to be  more appropriate when you have more homogeneous cells (e.g. one sample)
# doubScores <- addDoubletScores(
#   input = arrow_files,
#   k = 10, #Refers to how many cells near a "pseudo-doublet" to count.
#   knnMethod = "LSI", #Refers to the embedding to use for nearest neighbor search with doublet projection.
#   LSIMethod = 1,
#   force = TRUE
# )

## Create an ArchRProject
proj_carmack_test <- ArchRProject(
  ArrowFiles = arrow_files, 
  outputDirectory = "carmack_test",
  copyArrows = TRUE #This is recommened so that if you modify the Arrow files you have an original copy for later usage.
)

proj_carmack_test

# check memory size
paste0("Memory Size = ", round(object.size(proj_carmack_test) / 10^6, 3), " MB")
# "Memory Size = 58.615 MB"

getAvailableMatrices(proj_carmack_test)
# "TileMatrix"

# Plot fragment size distribution
p1 <- plotFragmentSizes(ArchRProj = proj_carmack_test)
p1

# Plot TSS enrichment
p2 <- plotTSSEnrichment(ArchRProj = proj_carmack_test)
p2

plotPDF(p1,p2, name = "QC-Sample-FragSizes-TSSProfile.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE, width = 5, height = 5)

# proj_carmack_test_filtered <- filterDoublets(proj_carmack_test)


## Dimensionality reduction
## Iterative Latent Semantic Indexing (LSI)
# proj_carmack_test_LSI <- addIterativeLSI(
#   ArchRProj = proj_carmack_test,
#   useMatrix = "TileMatrix", 
#   name = "IterativeLSI", 
#   iterations = 2, 
#   clusterParams = list( #See Seurat::FindClusters
#     resolution = c(0.2), 
#     sampleCells = 10000, 
#     n.start = 10
#   ), 
#   varFeatures = 25000, 
#   dimsToUse = 1:30
# )

## Clustering
# proj_carmack_test_cluster <- addClusters(
#   input = proj_carmack_test,
#   reducedDims = "IterativeLSI",
#   method = "Seurat",
#   name = "Clusters",
#   resolution = 0.8
# )

## Single-cell embeddings
# proj_carmack_test_sc <- addUMAP(
#   ArchRProj = proj_carmack_test, 
#   reducedDims = "IterativeLSI", 
#   name = "UMAP", 
#   nNeighbors = 30, 
#   minDist = 0.5, 
#   metric = "cosine"
# )

# proj_carmack_test_sc <- addUMAP(
#   ArchRProj = proj_carmack_test, 
#   reducedDims = "Harmony", 
#   name = "UMAPHarmony", 
#   nNeighbors = 30, 
#   minDist = 0.5, 
#   metric = "cosine"
# )


## Gene scores and marker genes (requires GeneScoreMatrix)

## Defining Cluster identity

## Pseudo-bulk replicates

## Peak calling
################################################
################################################
## R SESSION INFO                             ##
################################################
################################################




################################################
################################################
## VERSIONS FILE                              ##
################################################
################################################