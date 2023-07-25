## Version of  comparison methods

- Seurat: 4.3.0
- PCA: 1.6.1
- ICA: 0.5
- DCA: 0.3.4
- scVI: 0.17.3
- MAGIC: 3.0.0
- scImpute: 0.0.9

#### ZIFA

ZIFA is cloned from https://github.com/epierson9/ZIFA.git

#### VASC

VASC is cloned from https://github.com/wang-research/VASC.git

#### scGAE

scGAE is cloned from https://github.com/ZixiangLuo1161/scGAE.git

#### scGNN

scGNN is cloned from https://github.com/juexinwang/scGNN.git

#### ALRA

ALRA is cloned from https://github.com/KlugerLab/ALRA.git



## Downstream analysis tasks and comparison methods

### Expression recovery

In this task, we used comparison methods such as MAGIC, scImpute, ALRA, DCA, scVI.

### Cell clustering , visualization,time and memory

In this task, we used comparison methods such as Seurat, PCA, ICA,  ZIFA, VASC, scGAE, scGNN,DCA, scVI.

### Cell trajectory inference

In this task, we used comparison methods such as PCA, ICA,  ZIFA, VASC, scGAE, scGNN,DCA, scVI. To ensure fairness, we use all methods to reduce dimensionality to obtain the cell embedding of dimension 64.

For VASC, it provided file ` config.py` for parameter assignment and default setting for ` latent` is 2. In cell trajectory inference task, we changed  parameter ` latent` to 64:

```python
res = vasc( expr, 
           epoch=300,
           var=False,
           latent=64, #latent=config['latent']
           annealing=False,
           batch_size=batch_size,
           prefix=PREFIX,
           label=label,
           scale=config['scale'],
           patience=config['patience'])
```

For scGAE，the model parameter ` latent_dim` is 15. In cell trajectory inference task, we changed this parameter to 64:

```python
model = SCGAE(count, adj, adj_n, hidden_dim = 120, latent_dim = 64, decA = "DBL", layer_enc = "GAT")
```

Before training scGAE model, we first create  k-nearest neighbor graph. The default number of neighbors for each sample ` k` is 160, considering that a small dataset YAN was used in cell trajectory inference task, we changed the parameter ` k` to 80:

```python
adj, adj_n = get_adj(count, k = 80)
```

For scGNN, we made some changes when using scGNN, referring to three parameters ` --Regu-epochs`, ` --EM-epochs`, ` --EM-iteration` set by the tutorial of the algorithm instead of using their default parameters.

For `--Regu-epochs`, we changed number of epochs to train in feature autoencoder initially to 50:

```python
parser.add_argument('--Regu-epochs', type=int, default=50, metavar='N',
     help='number of epochs to train in Feature Autoencoder initially (default: 500)')
```

For `--EM-epochs`, we changed number of epochs to train feature autoencoder in iteration EM to 20:

```python
parser.add_argument('--EM-epochs', type=int, default=20, metavar='N', 
     help='number of epochs to train Feature Autoencoder in iteration EM (default: 200)')
```

For `--EM-iteration`, we changed number of iteration in total EM iteration to 2:

```python
parser.add_argument('--EM-iteration', type=int, default=2, metavar='N', 
     help='number of iteration in total EM iteration (default: 10)')
```

