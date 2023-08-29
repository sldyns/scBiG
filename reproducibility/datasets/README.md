## Summary of the simulated data

In  gene expression recovery experiments, we generated the simulated scRNA-seq raw count matrix using the function `splatSimulate` in the R package splatter. We set the number of cells and genes to 5000 and 10000, respectively, and adjusted the parameter `dropout.mid` to generate four different sparsity levels (0.8, 0.85, 0.9 and 0.95) for the simulation data.

In cell trajectory inference experiments, we used PROSSTT to generate a simulated scRNA-seq datasets with a tree topology. The end result of the simulation is a counts matrix of 500 cells and 10000 genes, as well as a vector that contains the branch assignments and pseudotime values for each simulated cell. In addition, PROSSTT is cloned from https://github.com/soedinglab/prosstt.git

In runtime and memory usage experiments, we also used the R package splatter to generate six simulated  datasets  with all 10,000 genes and 2000, 4000, 8000, 16000, 32000, and 64000 cells, respectively.

## Summary of the all scRNA-seq datasets

#### Seven scRNA-seq datasets  for cell clustering and visualization tasks.

| Dataset        | No. of cells | No. of genes | No. of cell types | Sparsity level |
| -------------- | ------------ | ------------ | ----------------- | -------------- |
| Human pancreas | 1724         | 20125        | 14                | 90.59%         |
| Mouse ES       | 2717         | 24175        | 4                 | 65.76 %        |
| Mouse bladder  | 2746         | 20670        | 16                | 94.87%         |
| Mouse kidney   | 3660         | 23797        | 8                 | 92.33%         |
| Human PBMC     | 4271         | 16653        | 8                 | 92.24%         |
| Human kidney   | 5685         | 25215        | 11                | 92.92 %        |
| Mouse retina   | 14653        | 11422        | 39                | 88.34%         |

#### Two scRNA-seq datasets for cell trajectory inference tasks.

| Datasets     | No. of cells | No. of genes | No. of Trajectory branches |
| ------------ | ------------ | ------------ | -------------------------- |
| Human embryo | 90           | 20214        | 6                          |
| Mouse embryo | 268          | 22431        | 10                         |

#### Two scRNA-seq datasets for gene co-expression tasks.

| Datasets      | No. of cells | No. of genes | No. of  cell types |
| ------------- | ------------ | ------------ | ------------------ |
| Human liver   | 500          | 28790        | 7                  |
| Mouse trachea | 7193         | 18388        | 6                  |

## Download

All datasets can be downloaded from https://drive.google.com/drive/folders/1JX589QWH7N895cAMAdJR9KDaFEigy1r3 or
downloaded from the following links.

| Dataset        | Website                                                      |
| -------------- | ------------------------------------------------------------ |
| Human pancreas | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230758 |
| Mouse ES       | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525  |
| Mouse bladder  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097 |
| Mouse kidney   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333  |
| Human PBMC     | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583  |
| Human kidney   | https://github.com/xuebaliang/scziDesk/tree/master/dataset/Young |
| Mouse retina   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63472  |
| Human embryo   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36552  |
| Mouse embryo   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45719  |
| Human liver    | https://github.com/quon-titative-biology/siVAE/blob/master/sample_fetal_liver_atlas_dataset.h5ad |
| Mouse trachea  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103354 |

In cell clustering task, we need to sample real scRNA-seq dataset. If you decide to perform sampling, please find the common folder and run the `sample` function in `read.py` to generate the sampled datasets for reproduciblity.

