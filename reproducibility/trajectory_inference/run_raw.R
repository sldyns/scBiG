library(rhdf5)
library(SingleCellExperiment)
library(slingshot, quietly = FALSE)
library("FactoMineR")
library(grDevices)
library(uwot)
library(RColorBrewer)
library(Seurat)
library(TSCAN)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")

#read data
read <- function(data){
  root<-paste0("../reproducibility/datasets/trajectory/")
  h1<-H5Fopen(paste0(root,data,".h5"))
  h5dump(h1,load=FALSE)
  counts<-h1$X
  dim(counts)
  label<-h1$Y
  sce <- SingleCellExperiment(assays = List(counts = counts))
  output <- list(sce,counts,label)
  return(output)
}

#Normalization
FQnorm <- function(counts){
  rk <- apply(counts,2,rank,ties.method='min')
  counts.sort <- apply(counts,2,sort)
  refdist <- apply(counts.sort,1,median)
  norm <- apply(rk,2,function(r){ refdist[r] })
  rownames(norm) <- rownames(counts)
  return(norm)
}

##Raw
method<-"raw"

data<-'DPT'
set.seed(0)
output <-read(data);sce <- output[[1]]; counts <- output[[2]];
label <- output[[3]]; k<-max(label)+1
# filter genes down to potential cell-type markers
geneFilter <- apply(assays(sce)$counts,1,function(x){
  sum(x >= 1) >= 3
})
sce <- sce[geneFilter, ]
assays(sce)$norm <- FQnorm(assays(sce)$counts)
#Normalization
norm<-t(log1p(assays(sce)$norm)); dim(norm)
rd2 <- uwot::umap(norm)
colnames(rd2) <- c('UMAP1', 'UMAP2')
reducedDims(sce) <- SimpleList(UMAP = rd2)
cl2 <- kmeans(rd2, centers = k)$cluster; cl2<-as.integer(cl2)
colData(sce)$kmeans <- cl2

#save
saveroot <- '../reproducibility/results/trajectory_inference/'
plot <-paste0(data,"_Raw")
np$savez(paste0(saveroot,data,'/', plot,".npz"),
         umap=rd2,
         true=label,
         data=t(counts),
         cluster=cl2)

data<-'YAN'
set.seed(5)
output <-read(data);sce <- output[[1]]; counts <- output[[2]];
label <- output[[3]]; k<-max(label)+1
# filter genes down to potential cell-type markers
geneFilter <- apply(assays(sce)$counts,1,function(x){
  sum(x >= 1) >= 3
})
sce <- sce[geneFilter, ]
assays(sce)$norm <- FQnorm(assays(sce)$counts)
#Normalization
norm<-t(log1p(assays(sce)$norm)); dim(norm)
rd2 <- uwot::umap(norm)
colnames(rd2) <- c('UMAP1', 'UMAP2')
reducedDims(sce) <- SimpleList(UMAP = rd2)
cl2 <- kmeans(rd2, centers = k)$cluster; cl2<-as.integer(cl2)
colData(sce)$kmeans <- cl2

#save
saveroot <- '../reproducibility/results/trajectory_inference/'
plot <-paste0(data,"_Raw")
np$savez(paste0(saveroot,data,'/', plot,".npz"),
         umap=rd2,
         true=label,
         data=t(counts),
         cluster=cl2)

data<-'Deng'
set.seed(0)
output <-read(data);sce <- output[[1]]; counts <- output[[2]];
label <- output[[3]]; k<-max(label)+1
# filter genes down to potential cell-type markers
geneFilter <- apply(assays(sce)$counts,1,function(x){
  sum(x >= 3) >= 10
})
sce <- sce[geneFilter, ]
assays(sce)$norm <- FQnorm(assays(sce)$counts)
#Normalization
norm<-t(log1p(assays(sce)$norm)); dim(norm)
rd2 <- uwot::umap(norm)
colnames(rd2) <- c('UMAP1', 'UMAP2')
reducedDims(sce) <- SimpleList(UMAP = rd2)
cl2 <- kmeans(rd2, centers = k)$cluster; cl2<-as.integer(cl2)
colData(sce)$kmeans <- cl2

#save
saveroot <- '../reproducibility/results/trajectory_inference/'
plot <-paste0(data,"_Raw")
np$savez(paste0(saveroot,data,'/', plot,".npz"),
         umap=rd2,
         true=label,
         data=t(counts),
         cluster=cl2)

data<-'Buettner'
set.seed(0)
output <-read(data);sce <- output[[1]]; counts <- output[[2]];
label <- output[[3]]; k<-max(label)+1
# filter genes down to potential cell-type markers
geneFilter <- apply(assays(sce)$counts,1,function(x){
  sum(x >= 1) >= 3
})
sce <- sce[geneFilter, ]
assays(sce)$norm <- FQnorm(assays(sce)$counts)
#Normalization
norm<-t(log1p(assays(sce)$norm)); dim(norm)
rd2 <- uwot::umap(norm)
colnames(rd2) <- c('UMAP1', 'UMAP2')
reducedDims(sce) <- SimpleList(UMAP = rd2)
cl2 <- kmeans(rd2, centers = k)$cluster; cl2<-as.integer(cl2)
colData(sce)$kmeans <- cl2

#save
saveroot <- '../reproducibility/results/trajectory_inference/'
plot <-paste0(data,"_Raw")
np$savez(paste0(saveroot,data,'/', plot,".npz"),
         umap=rd2,
         true=label,
         data=t(counts),
         cluster=cl2)
  




