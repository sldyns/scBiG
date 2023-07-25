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

t<-c('10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts',
     'Adam','Human_pancreatic_islets','Macosko_mouse_retina')
for ( i in 1:length(t)) {
  print(t[i])
name<-paste0(t[i])
##NMI
NMI <- function(c,t){
  n <- length(c)
  r <- length(unique(c))
  g <- length(unique(t))
  N <- matrix(0,nrow = r , ncol = g)
  for(i in 1:r){
    for (j in 1:g){
      N[i,j] = sum(t[c == i] == j)
    }
  }
  N_t <- colSums(N)
  N_c <- rowSums(N)
  B <- (1/n)*log(t( t( (n*N) / N_c ) / N_t))
  W <- B*N
  I <- sum(W,na.rm = T) 
  H_c <- sum((1/n)*(N_c * log(N_c/n)) , na.rm = T)
  H_t <- sum((1/n)*(N_t * log(N_t/n)) , na.rm = T)    
  nmi <- I/sqrt(H_c * H_t)
  return (nmi)
}

h1=H5Fopen(paste0("../reproducibility/datasets/real/",name,".h5"))
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
data <- FindNeighbors(data, dims = 1:15)
data <- FindClusters(data, resolution = 1,algorithm =1)
data <- RunUMAP(data, dims = 1:15)

cluster = data@meta.data[["seurat_clusters"]]
cluster = as.numeric(cluster)

DimPlot(data, reduction = "umap")
LabelClusters(DimPlot(data, reduction = "umap"),id = 'ident')

UMAP = data@reductions[["umap"]]@cell.embeddings

library(cidr)
ARI<- adjustedRandIndex(cluster, celltype)
ARI
NMI = NMI(cluster, celltype)
NMI

root <-paste0("../reproducibility/results/visualization/")
plot <-paste0("record_",name,"_",method)
np$savez(paste0(root, name,'/',plot,".npz"),
         ari=round(ARI,4), 
         nmi=round(NMI,4),
         umap=UMAP,
         true=celltype,
         louvain=cluster)
}
