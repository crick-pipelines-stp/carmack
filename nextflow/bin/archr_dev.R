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
library(ArchR)
ArchR::installExtraPackages() 
library(devtools)
library(BiocManager)
library(Rsamtools)
library(irlba)

# install.packages("Cairo")
#library(Cairo)

# install.packages("svg")
# install.packages("cairo_pdf")
# install.packages("cairo_ps")

install.packages("grDevices")
library(grDevices)


################################################
################################################
## Load and prepare data                      ##
################################################
################################################

# Set the number of threads
# addArchRThreads(threads = opt$cores) 
addArchRThreads(threads = 1) 

# Set genome
# addArchRGenome(opt$genome)
addArchRGenome("hg38")

# Load the input data
# inputFiles <- list.files('./mount/data', pattern = '*.tagged.bam$', full.names = T)
# inputFiles

inputFiles <- list.files('./mount/data', pattern = '*new.tsv.gz$', full.names = T)
inputFiles

################################################
################################################
## ArchR analysis                             ##
################################################
################################################

## Create arrow files
arrow_files <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = "hydrop_scatac_1_S1",
  minTSS = 0, # Can always increase this later (originally 4)
  minFrags = 1,  # originally 1000
  addTileMat = TRUE,
  addGeneScoreMat = FALSE, # Originally TRUE (produces metadata), but set to false as caused errors
  # evt. Set to true only if more than 1 sample
  force = TRUE
)

# Cannot generate the GeneScoreMatrix
# arrow_files2 <- createArrowFiles(
#   inputFiles = inputFiles,
#   sampleNames = "hydrop_scatac_1_S1_2",
#   minTSS = 0, # Can always increase this later (originally 4)
#   minFrags = 1,  # originally 1000
#   addTileMat = TRUE,
#   addGeneScoreMat = TRUE, # Originally TRUE (produces metadata), but set to false as caused errors
#   # evt. Set to true only if more than 1 sample
#   force = TRUE
# )

# No arrow files generated, likely because the nr of fragments per cell and the TS enrichment scores are much lower for our data
# arrow_files_minTSS4_minFrags10 <- createArrowFiles(
#   inputFiles = inputFiles,
#   sampleNames = "hydrop_scatac_1_S1_minTSS4_minFrags10",
#   minTSS = 4, # Can always increase this later (originally 4)
#   minFrags = 10,  # originally 1000
#   addTileMat = TRUE,
#   addGeneScoreMat = FALSE, # Originally TRUE (produces metadata), but set to false as caused errors
#   # evt. Set to true only if more than 1 sample
#   force = TRUE
# )


## Inspect arrow files
arrow_files
# "hydrop_scatac_1_S1_R1_001.arrow"
# rm(arrow_files)


## Create an ArchRProject
proj_carmack_test <- ArchRProject(
  ArrowFiles = arrow_files, 
  outputDirectory = "./mount/data/archr",
  copyArrows = TRUE #This is recommended so that if you modify the Arrow files you have an original copy for later usage.
)

proj_carmack_test
# rm(proj_carmack_test)

# check memory size
paste0("Memory Size = ", round(object.size(proj_carmack_test) / 10^6, 3), " MB")
# "Memory Size = 58.631 MB"

getAvailableMatrices(proj_carmack_test)
# "TileMatrix"

head(proj_carmack_test$cellNames)
# [1] "hydrop_scatac_1_S1#GTCAAGCCAAGAGTAACAGGATCCAGTGCA" "hydrop_scatac_1_S1#TGTAGCAAGTGACGGACGGTACTAATGACG" "hydrop_scatac_1_S1#GAACAGTAGTCAACCAACGGGAGCGGTAAC"
# [4] "hydrop_scatac_1_S1#CACAAGCATTATGAGCCAGAAGGACGTTCG" "hydrop_scatac_1_S1#CAAGGTCGATGTGATAGGAAAGTTGGAAGA" "hydrop_scatac_1_S1#GAACAGTAGTACGGTGGACTCAGTGTGGAA"

head(proj_carmack_test$Sample)
# [1] "hydrop_scatac_1_S1" "hydrop_scatac_1_S1" "hydrop_scatac_1_S1" "hydrop_scatac_1_S1" "hydrop_scatac_1_S1" "hydrop_scatac_1_S1"

quantile(proj_carmack_test$TSSEnrichment)
# 0%   25%   50%   75%  100% 
# 0.000 0.000 0.000 0.000 1.386 

df <- getCellColData(proj_carmack_test, select = c("log10(nFrags)", "TSSEnrichment"))
df

p <- ggPoint(
  x = df[,1], 
  y = df[,2], 
  colorDensity = TRUE,
  continuousSet = "sambaNight",
  xlabel = "Log10 Unique Fragments",
  ylabel = "TSS Enrichment",
  xlim = c(log10(500), quantile(df[,1], probs = 0.99)),
  ylim = c(0, quantile(df[,2], probs = 0.99))
) + geom_hline(yintercept = 4, lty = "dashed") + geom_vline(xintercept = 3, lty = "dashed")

p

plotPDF(p, name = "TSS-vs-Frags.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE)


## Per-cell QC
# Above command automatically generates a QualityControl folder with 
# a Fragment_Size_Distribution.pdf and a TSS_by_Unique_Frags.pdf file
# Plot fragment size distribution
p1 <- plotFragmentSizes(ArchRProj = proj_carmack_test)
p1

# Plot TSS enrichment
p2 <- plotTSSEnrichment(ArchRProj = proj_carmack_test)
p2

plotPDF(p1,p2, name = "QC-Sample-FragSizes-TSSProfile.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE, width = 5, height = 5)

# proj_carmack_test_filtered <- filterDoublets(proj_carmack_test)

## Plotting sample stats
# A ridge plot for each sample for the TSS enrichment scores
p3_test <- plotGroups(
  ArchRProj = proj_carmack_test, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "TSSEnrichment",
  plotAs = "ridges"
)

p3_test

# A violin plot for each sample for the TSS enrichment scores
p4_test <- plotGroups(
  ArchRProj = proj_carmack_test, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "TSSEnrichment",
  plotAs = "violin",
  alpha = 0.4,
  addBoxPlot = TRUE
)

p4_test

# A ridge plot for each sample for the log10(unique nuclear fragments)
p5_test <- plotGroups(
  ArchRProj = proj_carmack_test, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "log10(nFrags)",
  plotAs = "ridges"
)

p5_test

# A violin plot for each sample for the log10(unique nuclear fragments)
p6_test <- plotGroups(
  ArchRProj = proj_carmack_test, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "log10(nFrags)",
  plotAs = "violin",
  alpha = 0.4,
  addBoxPlot = TRUE
)

p6_test

# Save plots
plotPDF(p3_test,p4_test,p5_test,p6_test, name = "QC-Sample-Statistics.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE, width = 4, height = 4)

# Save archr project
saveArchRProject(ArchRProj = proj_carmack_test, outputDirectory = "Save-proj_carmack_test", load = FALSE)

## Doublet inference (should only be run for droplet-based single-cell chemistries e.g. hydrop)
doubScores <- addDoubletScores(
  input = arrow_files,
  k = 10, #Refers to how many cells near a "pseudo-doublet" to count.
  knnMethod = "UMAP", #Refers to the embedding to use for nearest neighbor search with doublet projection.
  LSIMethod = 1,
  # force = TRUE
  # Sometimes R^2 is low for a sample, which means that cellular heterogeneity is low and so it is difficult to call doublets
)

## Filtering doublets
proj_carmack_test <- filterDoublets(ArchRProj = proj_carmack_test, filterRatio=1) 
# Can increase filterRatio to remove more cells (i.e. filterRatio = The maximum ratio of predicted doublets to filter based on the number of pass-filter cells.)


# Alternative settings suggested to be more appropriate when you have more homogeneous cells (e.g. one sample)
# doubScores <- addDoubletScores(
#   input = arrow_files,
#   k = 10, #Refers to how many cells near a "pseudo-doublet" to count.
#   knnMethod = "LSI", #Refers to the embedding to use for nearest neighbor search with doublet projection.
#   LSIMethod = 1,
#   force = TRUE
# )


## Dimensionality reduction
## Iterative Latent Semantic Indexing (LSI)
proj_carmack_test <- addIterativeLSI(
  ArchRProj = proj_carmack_test,
  useMatrix = "TileMatrix", 
  name = "IterativeLSI", 
  iterations = 2, 
  clusterParams = list( #See Seurat::FindClusters
    resolution = c(0.2), 
    # sampleCells = 10000, 
    n.start = 10
  ), 
  varFeatures = 5000, # originally 25000
  dimsToUse = 1:30 
)



# # Dimensionality reduction using Latent Semantic Indexing (LSI)
# proj_carmack_test <- addIterativeLSI(ArchRProj = proj_carmack_test, useMatrix = "TileMatrix", name = "IterativeLSI")
# 
# 
# # Call clusters in the reduced dimension sub-space using Seurat's graph clustering by default
# proj_carmack_test <- addClusters(input = proj_carmack_test, reducedDims = "IterativeLSI")
# 
# 
# # Visualise 2D UMAP embedding
# proj_carmack_test <- addUMAP(ArchRProj = proj_carmack_test, reducedDims = "IterativeLSI", name = "UMAP", nNeighbors = 30, minDist = 0.5, metric = "cosine")
# 
# # Plot UMAP embedding coloured by sample
# p3 <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Sample", embedding = "UMAP")
# 
# # Plot UMAP embedding coloured by sample
# p4 <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Clusters", embedding = "UMAP")
# 
# ggAlignPlots(p3, p4, type = "h")
# 
# # Save plots in1 pdf file
# plotPDF(p3,p4, name = "Plot-UMAP-Sample-Clusters.pdf",
#         ArchRProj = proj_carmack_test, addDOC = FALSE, width = 5, height = 5)


# Can add more LSI iterations and start from a lower initial clustering resolution to detect more subtle batch effects.
# Can additionally reduce the nr of variable features to focus on the most variable features
# e.g. iterations = 4, resolution = c(0.1, 0.2, 0.4), varFeatures = 15000


# For extremely large datasets, ArchR can estimate the LSI dimensionality reduction with LSI projection.
# This can be done by setting sampleCellsFinal and projectCellsPre when running addIterativeLSI


# If batch effects are very strong that the iterative LSI approach isn't enough to correct these batch effects, 
# we can use Harmony. Pass a dimensionality reduction object from ArchR directly to the HarmonyMatrix() function
# proj_carmack_test <- addHarmony(
#   ArchRProj = proj_carmack_test,
#   reducedDims = "IterativeLSI",
#   name = "Harmony",
#   groupBy = "Sample"
# )



## Clustering
# proj_carmack_test <- addClusters(
#   input = proj_carmack_test,
#   reducedDims = "IterativeLSI",
#   method = "Seurat",
#   name = "Clusters",
#   resolution = 0.8
# )

## Single-cell embeddings
# proj_carmack_test <- addUMAP(
#   ArchRProj = proj_carmack_test, 
#   reducedDims = "IterativeLSI", # or "Harmony" if Harmony was used instead)
#   name = "UMAP", # could use a different name if Harmony is used (e.g. "UMAPHarmony)
#   nNeighbors = 30, 
#   minDist = 0.5, 
#   metric = "cosine"
# )

# Plot UMAP embedding coloured per sample and per cluster
# p7_test <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Sample", embedding = "UMAP")
 
# p8_test <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Clusters", embedding = "UMAP")

# ggAlignPlots(p7_test, p8_test, type = "h")

# Save plot
# plotPDF(p7_test, p8_test, name = "Plot-UMAP-Sample-Clusters.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE, width = 5, height = 5)


## t-SNE
# Alternative to UMAP embedding, you can use t-Stochastic Neighbour Embedding (t-SNE)
# proj_carmack_test <- addTSNE(ArchRProj = proj_carmack_test, reducedDims = "IterativeLSI", name = "TSNE", perplexity = 30)
# 
# p9_test <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Sample", embedding = "TSNE")
# 
# p10_test <- plotEmbedding(ArchRProj = proj_carmack_test, colorBy = "cellColData", name = "Clusters", embedding = "TSNE")
# 
# ggAlignPlots(p9_test, p10_test, type = "h")
# 
# # Save plot
# plotPDF(p9_test, p10_test, name = "Plot-TSNE-Sample-Clusters.pdf", ArchRProj = proj_carmack_test, addDOC = FALSE, width = 5, height = 5)


## Gene scores and marker genes (requires GeneScoreMatrix)
# This is useful in inferring which cell type is represented by each cluster, 
# using a priori knowledge of cell type-specific marker genes and estimating 
# the expression of these genes based on the chromatin accessibility data.
# The gene scores in the GeneScoreMatrix are a measurement of how highly expressed a gene will be based on the accessibility of regulatory elements in the vicinity of the gene.

# Identifying marker genes
# markersGS <- getMarkerFeatures(
#   ArchRProj = proj_carmack_test, 
#   useMatrix = "GeneScoreMatrix", 
#   groupBy = "Clusters",
#   bias = c("TSSEnrichment", "log10(nFrags)"),
#   testMethod = "wilcoxon"
# )
# 
# markerList <- getMarkers(markersGS, cutOff = "FDR <= 0.01 & Log2FC >= 1.25")
# # markerList$C6

# optionally you can supply some marker genes to 
# label on the heatmap via the labelMarkers parameter.
# markerGenes  <- c(
#   "CD34", #Early Progenitor
#   "GATA1", #Erythroid
#   "PAX5", "MS4A1", "EBF1", "MME", #B-Cell Trajectory
#   "CD14", "CEBPB", "MPO", #Monocytes
#   "IRF8", 
#   "CD3D", "CD8A", "TBX21", "IL7R" #TCells
# )

# heatmapGS <- markerHeatmap(
#   seMarker = markersGS, 
#   cutOff = "FDR <= 0.01 & Log2FC >= 1.25", 
#   labelMarkers = NULL, # Alt. markerGenes
#   transpose = TRUE
# )
# 
# ComplexHeatmap::draw(heatmapGS, heatmap_legend_side = "bot", annotation_legend_side = "bot")
# plotPDF(heatmapGS, name = "GeneScores-Marker-Heatmap", width = 8, height = 6, ArchRProj = proj_carmack_test, addDOC = FALSE)

# Visualising the marker genes on an embedding
# p11_test <- plotEmbedding(
#   ArchRProj = proj_carmack_test, 
#   colorBy = "GeneScoreMatrix", 
#   name = markerGenes, 
#   embedding = "UMAP",
#   imputeWeights = getImputeWeights(proj_carmack_test) # or NULL
# )

# Can do this for all genes
# p12_test <- lapply(p, function(x){
#   x + guides(color = FALSE, fill = FALSE) + 
#     theme_ArchR(baseSize = 6.5) +
#     theme(plot.margin = unit(c(0, 0, 0, 0), "cm")) +
#     theme(
#       axis.text.x=element_blank(), 
#       axis.ticks.x=element_blank(), 
#       axis.text.y=element_blank(), 
#       axis.ticks.y=element_blank()
#     )
# })
# do.call(cowplot::plot_grid, c(list(ncol = 3),p12_test))
# 
# 
# plotPDF(plotList = p11_test, 
#         name = "Plot-UMAP-Marker-Genes-WO-Imputation.pdf", 
#         ArchRProj = proj_carmack_test, 
#         addDOC = FALSE, width = 5, height = 5)


## Pseudo-bulk replicates (requires clustering)
proj_carmack_test <- addGroupCoverages(ArchRProj = proj_carmack_test, groupBy = "Clusters")


## Peak calling
pathToMacs2 <- findMacs2()

proj_carmack_test <- addReproduciblePeakSet(
  ArchRProj = proj_carmack_test, 
  groupBy = "Clusters", 
  pathToMacs2 = pathToMacs2
)

getPeakSet(proj_carmack_test)

# Save ArchR project
saveArchRProject(ArchRProj = proj_carmack_test, outputDirectory = "Save-proj_carmack_test", load = FALSE)

# Add the peak matrix to the ArchR project to prepare for downstream analysis
proj_carmack_test <- addPeakMatrix(proj_carmack_test)

getAvailableMatrices(proj_carmack_test)

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