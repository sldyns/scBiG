library(rhdf5)
library(SingleCellExperiment)
library(slingshot, quietly = FALSE)
library("FactoMineR")
library(grDevices)
library(uwot)
library(RColorBrewer)
library(TSCAN)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")

run <- function(method){
  npz1 <- np$load(paste0(root,data,"_",method,".npz"))
  counts<-npz1$f[["data"]]
  counts<-t(counts)
  dim(counts)
  sce <- SingleCellExperiment(assays = List(counts = counts))
  rd1 <- npz1$f[["latent"]]
  rd2 <- npz1$f[["umap"]]
  colnames(rd2) <- c('UMAP1', 'UMAP2')
  reducedDims(sce) <- SimpleList(UMAP = rd2)
  reducedDims(sce) <- SimpleList(PCA = rd1, UMAP = rd2)
  #Clustering Cells
  cl3<- npz1$f[["louvain"]]
  cl3<-as.integer(cl3)
  colData(sce)$louvain <- cl3
  ##Using Slingshot
  sce <- slingshot(sce, clusterLabels = 'louvain', reducedDim = 'UMAP')
  summary(sce$slingPseudotime_1)
  Pseudo<-sce$slingPseudotime_1
  time<-npz1$f[["true"]]
  kendall = cor(Pseudo,time,method = "kendall", use = "complete.obs")[[1]]
  cor<-round(kendall,2)
  print(cor)
  subpopulation <- data.frame(cell = c(1:length(time)), sub = time)
  pseudo.df<-data.frame(sample_name=c(1:length(Pseudo)),State=time,Pseudotime=order(unlist(Pseudo)))
  POS <- orderscore(subpopulation, pseudo.df)
  print(POS[3])
  POS<-round(POS[3],2)
  print(POS)
  cor <- sprintf("%.2f", cor)
  POS <- sprintf("%.2f", POS)
  colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
  plotcol <- colors[cut(sce$slingPseudotime_1, breaks=100)]
  plt<-plot(reducedDims(sce)$UMAP, col = plotcol, pch=16, asp = 1,
            main = paste0(method," (cor:",cor," pos:",POS,")"),
            cex.main = 1.25,bty='l') 
  lines(SlingshotDataSet(sce), lwd=2, col='black')
  return(plt)
}


data<-'DPT'
root<-paste0("../results/trajectory_inference/",data,"/")
savefig<-paste0("../figures/")
svg(filename = paste0(savefig,data,".svg"),width=12,height=5)
opar <- par(no.readonly = TRUE)
par(mfrow=c(2,5),mar=c(4,4,2.5,1.5),oma=c(0,0,0,0),mgp = c(2.5,1,0),pty='m') 
method<-c('Seurat'); plt0<-run(method); 
method<-c('ICA'); plt2<-run(method); 
method<-c('ZIFA'); plt3<-run(method); 
method<-c('graph-sc'); plt4<-run(method); 
method<-c('scGAE'); plt5<-run(method); 
method<-c('scGNN'); plt6<-run(method); 
method<-c('DCA'); plt7<-run(method); 
method<-c('scVI'); plt8<-run(method); 
method<-c('SIMBA'); plt1<-run(method);
method<-c('scBiG'); plt9<-run(method); 
par(opar,pin = c(20,15))
dev.off()

data<-'YAN'
root<-paste0("../results/trajectory_inference/",data,"/")
savefig<-paste0("../figures/")
svg(filename = paste0(savefig,data,".svg"),width=12,height=5)
opar <- par(no.readonly = TRUE)
par(mfrow=c(2,5),mar=c(4,4,2.5,1.5),oma=c(0,0,0,0),mgp = c(2.5,1,0),pty='m') 
method<-c('Seurat'); plt0<-run(method); 
method<-c('ICA'); plt2<-run(method); 
method<-c('ZIFA'); plt3<-run(method); 
method<-c('graph-sc'); plt4<-run(method); 
method<-c('scGAE'); plt5<-run(method); 
method<-c('scGNN'); plt6<-run(method); 
method<-c('DCA'); plt7<-run(method); 
method<-c('scVI'); plt8<-run(method); 
method<-c('SIMBA'); plt1<-run(method);
method<-c('scBiG'); plt9<-run(method); 
par(opar,pin = c(20,15))
dev.off()

data<-'Deng'
root<-paste0("../results/trajectory_inference/",data,"/")
savefig<-paste0("../figures/")
svg(filename = paste0(savefig,data,".svg"),width=12,height=5)
opar <- par(no.readonly = TRUE)
par(mfrow=c(2,5),mar=c(4,4,2.5,1.5),oma=c(0,0,0,0),mgp = c(2.5,1,0),pty='m') 
method<-c('Seurat'); plt0<-run(method); 
method<-c('ICA'); plt2<-run(method); 
method<-c('ZIFA'); plt3<-run(method); 
method<-c('graph-sc'); plt4<-run(method); 
method<-c('scGAE'); plt5<-run(method); 
method<-c('scGNN'); plt6<-run(method); 
method<-c('DCA'); plt7<-run(method); 
method<-c('scVI'); plt8<-run(method); 
method<-c('SIMBA'); plt1<-run(method);
method<-c('scBiG'); plt9<-run(method); 
par(opar,pin = c(20,15))
dev.off()
