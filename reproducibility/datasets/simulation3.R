library("splatter")
library(rhdf5)
set.seed(0)

# ----------------Generate the simulation data using Splatter package------------------

run<-function(cells,genes){

params = newSplatParams()
d=5 

params = setParams(params, list(batchCells = cells,
                                nGenes = genes,
                                group.prob = c(0.05, 0.15, 0.2, 0.25, 0.35))
)
params

sim = splatSimulateGroups(params,
                          dropout.shape = c(-0.6,-0.6,-0.6,-0.6,-0.6),
                          dropout.mid = c(d,d,d,d,d),
                          dropout.type = "group", 
)

simtrue <- as.matrix(sim@assays@data@listData[["TrueCounts"]])
X <- as.matrix(assays(sim)$count)
label <- as.integer(substring(sim$Group,6))

#
counts<-X
zero.rate <- sum(counts==0)/(dim(counts)[1]*dim(counts)[2])
print(paste0("zero expression rate ", zero.rate))

#save
name <-paste0("../reproducibility/datasets/time/")
name1= paste0("data_cell", cells)
print(paste0("Creating ... ", name, name1,".h5"))
h5createFile(paste0(name, name1,".h5"))
h5write(X,paste0(name, name1,".h5"),"X")
h5write(simtrue,paste0(name, name1,".h5"),"X_true")
h5write(label, paste0(name, name1,".h5"),"Y")

}

run(cells=2000,genes=10000)
run(cells=4000,genes=10000)
run(cells=8000,genes=10000)
run(cells=16000,genes=10000)
run(cells=32000,genes=10000)
run(cells=64000,genes=10000)
