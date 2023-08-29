library(Seurat)
library(dplyr)
library(patchwork)
library(rhdf5)
library(hdf5r)
library(Matrix)
library(igraph) 
library(SeuratObject)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")
library(utils)
dir<-paste0('../datasets/time/')

t<-c('2000','4000','8000','16000','32000','64000')
for ( i in 1:length(t)) {
  print(t[i])
  name<-paste0("data_cell",t[i])
  name1<-paste0("cell",t[i])
  set.seed(0)
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
  
  h1=H5Fopen(paste0(dir,name,".h5"))
  h5dump(h1,load=FALSE)
  raw_count=h1$X
  celltype= h1$Y
  K=length(unique(celltype))
  colnames <- colnames(raw_count, do.NULL = FALSE)
  rownames <- rownames(raw_count, do.NULL = FALSE)
  
  colnames(raw_count) <- colnames
  rownames(raw_count) <- rownames
  
  s=Sys.time()
  real.data <- raw_count
  data <- CreateSeuratObject(counts = real.data,min.cells = 3, min.features = 200)
  print(data)
  data <- NormalizeData(data, normalization.method = "LogNormalize", scale.factor = 10000)
  data <- FindVariableFeatures(data, verbose = FALSE)
  data <- ScaleData(data)
  data <- RunPCA(data, npcs = 64,features = VariableFeatures(object = data))
  e=Sys.time()
  print(e-s)
  print(memory.size(max=TRUE))
  memory_usage=memory.size(max=TRUE)
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
  print(ARI)
  NMI = NMI(cluster, celltype)
  print(NMI)
  
  root <-paste0('../results/time_memory/')
  plot <-paste0("record_",name1,"_Seurat")
  np$savez(paste0(root, t[i],'/',plot,".npz"),
           time=e-s,memory_usage=memory_usage)
}
