library(scImpute)

t<-c('0.8','0.85','0.9','0.95')
for ( i in 1:length(t)) {
  print(t[i])
  root<-paste0("../reproducibility/datasets/sim/")
  results<-paste0("../reproducibility/results/expression_recovery/")
  name<-paste0('data_',t[i])
  name1<-paste0(t[i],'/')

  scimpute(
    count_path = paste0(root,name,".csv"),
    infile = "csv",
    outfile = "csv",
    out_dir = paste0(results,name1),
    labeled = FALSE,
    drop_thre = 0.5,
    Kcluster =5,
    ncores = 10)
}

