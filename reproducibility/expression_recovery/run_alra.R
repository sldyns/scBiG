# Demo for running ALRA. 
source('../reproducibility/pkgs/ALRA/alra.R')
library(dplyr)
library(magrittr)
library(hdf5r)
library(rhdf5)
library(Matrix)
library(RcppCNPy)
library(reticulate)
np <- import("numpy")

set.seed(0)
t <- c("0.9","0.95","0.8","0.85")
method <- "ALRA"
for ( i in 1:length(t)) {
  print(t[i])
  name <-t[i]
  #A's row is cell
  h1=H5Fopen(paste0("../reproducibility/datasets/sim/","data_",name,".h5"))
  h5dump(h1,load=FALSE)
  A=t(h1$X)
  dim(A)
  A_norm <- normalize_data(A)
  k_choice <- choose_k(A_norm)
  A_norm_completed <- alra(A_norm,k=k_choice$k)[[3]]
  
  A0 =t(h1$X_true)
  dim(A0)
  
  totalUMIPerCell <- rowSums(A);
  if (any(totalUMIPerCell == 0)) {
    toRemove <- which(totalUMIPerCell == 0)
    A <- A[-toRemove,]
    A0 <- A0[-toRemove,]
    totalUMIPerCell <- totalUMIPerCell[-toRemove]
  }  
  YPred1=(exp(A_norm_completed) -1)/10E3
  YPred=sweep(YPred1, 1, totalUMIPerCell , "*")
  
  real = c(t(A0))
  pred = c(t(YPred))
  obser = c(A)
  
  X1 <- (real > 0)
  X2 <- (obser  == 0)
  X <- X1&X2
  
  rmse <- sqrt(mean((real-pred)^2))
  cor <- cor(real,pred, method = 'pearson')
  
  rmsed <- sqrt(mean((real[X]-pred[X])^2))
  cord <- cor(real[X],pred[X], method = 'pearson')
  
  root <-paste0("../reproducibility/results/expression_recovery/")
  plot <-paste0("result_",name,"_",method)
  np$savez(paste0(root, name,'/',plot,".npz"),
           rmse=round(rmse,4),
           rmsed=round(rmsed,4),
           pcc=round(cor,4),
           pccd=round(cord,4))
  rm()
  }