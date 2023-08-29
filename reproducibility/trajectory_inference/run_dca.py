import os
import random

import h5py
import numpy as np
import scanpy as sc
import tensorflow as tf
from dca.api import dca

from scbig.utils import louvain, calculate_metric


def seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

for dataset in ['DPT', 'YAN', 'Deng']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'DCA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X.astype('float'))
    sc.pp.filter_genes(adata, min_cells=1)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))

    dca(adata, mode='latent', ae_type='zinb', threads=1)
    print(adata)
    adata1 = adata
    dca(adata1, mode='denoise', ae_type='zinb')
    print(adata1)

    # louvain
    adata = louvain(adata, resolution=None, use_rep='X_dca')
    y_pred_l = np.asarray(adata.obs['louvain'], dtype=int)
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
             true=Y,
             umap=adata.obsm['X_umap'],
             latent=adata.obsm['X_dca'],
             data=adata1.X,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))
