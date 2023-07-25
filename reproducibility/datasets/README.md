## Summary of the simulated data

In  gene expression recovery experiments, we generated the simulated scRNA-seq raw count matrix using the function `splatSimulate` in the R package splatter. We set the number of cells and genes to 5000 and 6000, respectively, and adjusted the parameter `dropout.mid` to generate four different sparsity levels (0.8, 0.85, 0.9 and 0.95) for the simulation data.

In cell trajectory inference experiments, we used PROSSTT to generate a simulated scRNA-seq datasets with a tree topology. The end result of the simulation is a counts matrix of 500 cells and 5000 genes, as well as a vector that contains the branch assignments and pseudotime values for each simulated cell. In addition, PROSSTT is cloned from https://github.com/soedinglab/prosstt.git

In runtime and memory usage experiments, we also used the R package splatter to generate six simulated  datasets  with all 10,000 genes and 2000, 4000, 8000, 16000, 32000, and 64000 cells, respectively.

## Summary of the all scRNA-seq datasets

####  Seven scRNA-seq datasets  for cell clustering and visualization tasks.

| Dataset                 | No. of cells | No. of genes | No. of cell types | Sparsity level |
| ----------------------- | ------------ | ------------ | ----------------- | -------------- |
| Human pancreatic islets | 1724         | 20125        | 14                | 90.59%         |
| Mouse ES cell           | 2717         | 24175        | 4                 | 65.76 %        |
| Mouse bladder cell      | 2746         | 20670        | 16                | 94.87%         |
| Adam                    | 3660         | 23797        | 8                 | 92.33%         |
| 10X PBMC                | 4271         | 16653        | 8                 | 92.24%         |
| Human kidney counts     | 5685         | 25215        | 11                | 92.92 %        |
| Macosko mouse retina    | 14653        | 11422        | 39                | 88.34%         |

#### Three scRNA-seq datasets  for cell trajectory inference tasks.

| Datasets | No. of cells | No. of genes | No. of Trajectory branches |
| -------- | ------------ | ------------ | -------------------------- |
| YAN      | 90           | 20214        | 6                          |
| Deng     | 268          | 22431        | 10                         |
| Buettner | 288          | 38390        | 3                          |

#### Two scRNA-seq datasets  for gene co-expression tasks.

| Datasets          | No. of cells | No. of genes | No. of  cell types |
| ----------------- | ------------ | ------------ | ------------------ |
| fetal_liver_atlas | 500          | 28790        | 7                  |
| Trachea           | 7193         | 18388        | 6                  |

## Download

All datasets can be downloaded from https://drive.google.com/drive/folders/1JX589QWH7N895cAMAdJR9KDaFEigy1r3 or downloaded from the following links.

| Dataset                 | Website                                                      |
| ----------------------- | ------------------------------------------------------------ |
| Human pancreatic islets | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230758 |
| Mouse ES cell           | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525  |
| Mouse bladder cell      | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097 |
| Adam                    | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333  |
| 10X PBMC                | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583  |
| Human kidney counts     | https://github.com/xuebaliang/scziDesk/tree/master/dataset/Young |
| Macosko mouse retina    | https://github.com/hemberg-lab/scRNA.seq.datasets/blob/master/R/macosko.R |
| YAN                     | https://github.com/hemberg-lab/scRNA.seq.datasets/blob/master/R/yan.R |
| Deng                    | https://github.com/hemberg-lab/scRNA.seq.datasets/blob/master/R/deng.R |
| Buettner                | https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2805 |
| fetal_liver_atlas       | https://github.com/quon-titative-biology/siVAE/blob/master/sample_fetal_liver_atlas_dataset.h5ad |
| Trachea                 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103354 |

In cell clustering task, we need to sample real scRNA-seq dataset. If you decide to perform sampling, please find the common folder and run the `sample` function in `read.py` to generate the sampled datasets for reproduciblity.

