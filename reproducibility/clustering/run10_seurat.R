library(Seurat)
library(dplyr)
library(patchwork)
library(R.utils)
library(rhdf5)
library(hdf5r)
library(Matrix)
library(igraph)
library(SeuratObject)

method <-'Seurat'

t0<-c('10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts',
     'Adam','Human_pancreatic_islets','Macosko_mouse_retina')

for ( i in 1:length(t0)) {
  print(t0[i])
  name<-paste0(t0[i])
  
  t <-1:10
  for ( i in t) {
    print(t[i])
    y<-paste0(name,'_',paste0(t[i]))
    h1=H5Fopen(paste0("../reproducibility/datasets/real/sample/",name,'/',y,".h5"))
    h5dump(h1,load=FALSE)
    raw_count=h1$X
    celltype= h1$Y
    celltype = as.numeric(celltype)
    K=length(unique(celltype))
    colnames <- colnames(raw_count, do.NULL = FALSE)
    rownames <- rownames(raw_count, do.NULL = FALSE)
    
    colnames(raw_count) <- colnames
    rownames(raw_count) <- rownames
    
    real.data <- raw_count
  
    data <- CreateSeuratObject(counts = real.data,min.cells = 3, min.features = 200)
    data
    
    data <- NormalizeData(data, normalization.method = "LogNormalize", scale.factor = 10000)
    data <- ScaleData(data)
    data <- FindVariableFeatures(data, verbose = FALSE)
    
    data <- RunPCA(data, features = VariableFeatures(object = data))
    data <- FindNeighbors(data, dims = 1:15)
    data <- FindClusters(data, resolution = 1,algorithm =1,random.seed =0)
    data <- RunUMAP(data, dims = 1:15)
    
    cluster = data@meta.data[["seurat_clusters"]]
    cluster = as.numeric(cluster)
    
    DimPlot(data, reduction = "umap")
    LabelClusters(DimPlot(data, reduction = "umap"),id = 'ident')
    
    # UMAP = data@reductions[["umap"]]@cell.embeddings
    
    library(cidr)
    ari<- adjustedRandIndex(cluster, celltype)
    ARI<-c(ARI,ari)
    
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
    nmi <- NMI(cluster, celltype)
    NMI1<- c(NMI1,nmi)
    
    }
    
    print(ARI)
    print(NMI1)
    
    root <-paste0("../reproducibility/results/clustering/")
    plot <-paste0("result_",name,"_",method)
    np$savez(paste0(root, name,'/',plot,".npz"),
             aril=ARI,
             nmil=NMI1)
}
