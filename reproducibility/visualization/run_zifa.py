import numpy as np
from ZIFA import block_ZIFA
import h5py
import os
import scanpy as sc
from scgraphne.utils import read_data,louvain,calculate_metric

import random
import tensorflow as tf
def seed(SEED):
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

# This gives an example for how to read in a real data called input.table. 
# genes are columns, samples are rows, each number is separated by a space. 
# If you do not want to install pandas, you can also use np.loadtxt: https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'ZIFA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)
    if dataset in ['Adam']:
        mat, obs, var, uns = read_data(os.path.join(dir0, '{}.h5'.format(dataset)), sparsify=False,
                                       skip_exprs=False)
        X = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        Y = cell_label

    else:
        with h5py.File(os.path.join(dir0, '{}.h5'.format(dataset)), "r") as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_)

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    X = np.log2(1 + X)
    Z, model_params = block_ZIFA.fitModel(X, 64, n_blocks=None)
    adata.obsm['feat'] = Z
    # louvain
    adata = louvain(adata, resolution=1,use_rep='feat')
    y_pred = np.array(adata.obs['louvain'])
    nmi_l, ari_l = calculate_metric(Y, y_pred)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/visualization/{}/record_{}_{}.npz".format(dataset, dataset, method)),
             ari=ari_l, nmi=nmi_l,
             umap=adata.obsm['X_umap'],
             true=np.array(adata.obs['cl_type'].values.astype(int)),
             louvain=np.array(adata.obs['louvain'].values.astype(int)))

