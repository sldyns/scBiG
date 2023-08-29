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
from clustering import *
import h5py
from scbig.utils import louvain, calculate_metric
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

for dataset in ['DPT', 'YAN', 'Deng']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'scGAE'
    dir0 = '../'

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset)), 'r') as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()
    idents = Y
    adata = sc.AnnData(X.astype('float'))
    adata.obs['true'] = Y
    adata = preprocess(adata)
    print(adata)
    count = adata.X
    # Compute adjacency matrix and normalized adjacency matrix
    adj, adj_n = get_adj(count, k=80)

    model = SCGAE(count, adj, adj_n, hidden_dim=120, latent_dim=64, decA="DBL", layer_enc="GAT")  ##15
    model.train(epochs=80, W_a=0.4, W_x=1)

    # Genertate latent embedding by tSNE
    E = model.embedding(count, adj_n)

    # # Spectral clustering
    # centers, labels = get_centers_spectral(Y, adj)
    # model.clustering_train(centers=centers, W_a = 0.4, W_x = 1, W_c = 1.5, epochs = 80)

    adata.obsm['X_scGAE'] = E
    # louvain
    adata = louvain(adata, resolution=None, use_rep='X_scGAE')
    y_pred_l = np.asarray(adata.obs['louvain'], dtype=int)
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(adata.obs['true'], y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)

    # Reconstruct expression matrix
    XX = model.rec_X(count, adj_n)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
             true=adata.obs['true'],
             umap=adata.obsm['X_umap'],
             latent=adata.obsm['X_scGAE'],
             data=XX,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))
