library(Seurat)
library(dplyr)
library(patchwork)
library(R.utils)
library(rhdf5)
library(hdf5r)
library(Matrix)
library(igraph) 
library(SeuratObject)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")

method <-'Seurat'
t<-c('DPT','YAN','Deng')
for ( i in 1:length(t)) {
  print(t[i])
name<-paste0(t[i])
h1=H5Fopen(paste0("../datasets/trajectory/",name,".h5"))
h5dump(h1,load=FALSE)
raw_count=h1$X
celltype= h1$Y
K=length(unique(celltype))
colnames <- colnames(raw_count, do.NULL = FALSE)
rownames <- rownames(raw_count, do.NULL = FALSE)

colnames(raw_count) <- colnames
rownames(raw_count) <- rownames

real.data <- raw_count

data <- CreateSeuratObject(counts = real.data,min.cells = 3, min.features = 200)
data

data <- NormalizeData(data, normalization.method = "LogNormalize", scale.factor = 10000)
data <- FindVariableFeatures(data, verbose = FALSE)
data <- ScaleData(data)

data <- RunPCA(data, features = VariableFeatures(object = data))
cellembeddings <- data@reductions$pca@cell.embeddings


root <-paste0("../results/trajectory_inference/")
plot <-paste0("embedding_",name,"_",method)
np$savez(paste0(root, name,'/',plot,".npz"),
         data=t(data@assays$RNA@scale.data),
         latent=cellembeddings,
         true=celltype)
}
