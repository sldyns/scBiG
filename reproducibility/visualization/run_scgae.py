import warnings

import numpy as np
import scanpy as sc
import tensorflow as tf
import sys
sys.path.append('../pkgs/scGAE/')
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# scGAE modules
from scgae import SCGAE
from preprocessing import *
from utils import *
from losses import *
from clustering import *

import h5py
from scbig.utils import read_data, louvain
import os
import random


def seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def preprocess(adata, scale=True):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)
    else:
        print('no scale!')
    return adata


# Load data
for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'scGAE'
    dir0 = '../'
    if dataset in ['Adam']:
        mat, obs, var, uns = read_data(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset)), sparsify=False,
                                       skip_exprs=False)
        X = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        Y = cell_label

    else:
        with h5py.File(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset)), 'r') as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_).squeeze()
    idents = Y
    adata = sc.AnnData(X.astype('float'))
    adata = preprocess(adata)
    print(adata)
    count = adata.X
    # Compute adjacency matrix and normalized adjacency matrix
    adj, adj_n = get_adj(count, k=160)

    model = SCGAE(count, adj, adj_n, hidden_dim=120, latent_dim=15, decA="DBL", layer_enc="GAT")
    model.train(epochs=80, W_a=0.4, W_x=1)

    # Genertate latent embedding by tSNE
    Y = model.embedding(count, adj_n)

    ####### Get initialized clustering center ######
    # Louvain clustering
    # centers, labels = get_centers_louvain(Y, adj)
    # Spectral clustering
    centers, labels = get_centers_spectral(Y, adj)
    model.clustering_train(centers=centers, W_a=0.4, W_x=1, W_c=1.5, epochs=40)

    # Genertate latent embedding and visualization after clustering-training
    Yc = model.embedding(count, adj_n)
    Y2c = doumap(Yc, dim=2)
    # myscatter(Y2c, idents)

    ##Evaluate clustering performance
    # labels = clustering(Yc, n_cluster=10, f='louvain')
    adata.obsm['feat'] = Yc
    adata = louvain(adata, resolution=1, use_rep='feat')
    labels = np.array(adata.obs['louvain'])
    NMI, RAND, HOMO, COMP = measure(idents, labels)

    np.savez(os.path.join(dir0, "results/visualization/{}/record_{}_{}.npz".format(dataset, dataset, method)),
             ari=RAND, nmi=NMI,
             umap=Y2c,
             true=idents.astype(int),
             louvain=labels.astype(int))
