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
    cellembeddings <- data@reductions$pca@cell.embeddings
    
    root <-paste0("../results/clustering/")
    plot <-paste0("embedding_",t[i],'_',name,"_",method)
    np$savez(paste0(root, name,'/',plot,".npz"),
             latent=cellembeddings,
             true=celltype)
  }
}
