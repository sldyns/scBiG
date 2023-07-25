library("splatter")
library(rhdf5)
set.seed(0)

# ----------------Generate the simulation data using Splatter package------------------
# Set the number of cells to 5000
# Set the number of genes to 6000
# Set the number of groups to 5
params = newSplatParams()
cells = 5000
genes = 6000
d = 3.15 #0.8  # d=3.9 #0.85  # d=4.8 #0.9  # d=6.3 #0.95
Sparsity = 0.8

# Setting parameters
params = setParams(params, list(batchCells = cells,
                                nGenes = genes,
                                group.prob = c(0.05, 0.15, 0.2, 0.25, 0.35),
                                de.prob = c(0.3, 0.1, 0.2, 0.01, 0.1),
                                de.facLoc = c(0.3, 0.1, 0.1, 0.01, 0.2),
                                de.facScale = c(0.1, 0.4, 0.2, 0.5, 0.4))
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

## Save real count matrix,count matrix with dropout and real cell type
counts<-X
zero.rate <- sum(counts==0)/(dim(counts)[1]*dim(counts)[2])
print(paste0("zero expression rate ", zero.rate))

#save
name <-paste0("../reproducibility/datasets/sim/")
name1= paste0("data1_", Sparsity)
print(paste0("Creating ... ", name, name1,".h5"))
h5createFile(paste0(name, name1,".h5"))
h5write(X,paste0(name, name1,".h5"),"X")
h5write(simtrue,paste0(name, name1,".h5"),"X_true")
h5write(label, paste0(name, name1,".h5"),"Y")
