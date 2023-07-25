import numpy as np
from ZIFA import block_ZIFA
import h5py
import os
import scanpy as sc
from scgraphne.utils import louvain,calculate_metric
import random
import tensorflow as tf

def seed(SEED):
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

for dataset in ['DPT', 'YAN', 'Deng', 'Buettner']:
    print('---------------- data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'ZIFA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)
    with h5py.File(os.path.join(dir0,'datasets/trajectory/{}.h5'.format(dataset)), 'r') as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    X = np.log2(1 + X)
    Z, model_params = block_ZIFA.fitModel(X, 64, n_blocks=None)
    adata.obsm['X_zifa'] = Z
    # louvain
    adata = louvain(adata, resolution=None, use_rep='X_zifa')
    y_pred_l = np.asarray(adata.obs['louvain'], dtype=int)
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0,"results/trajectory_inference/{}/{}_{}.npz".format(dataset,dataset,method)),
             true=Y,
             umap=adata.obsm['X_umap'],
             latent=Z,
             data=adata.X,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))