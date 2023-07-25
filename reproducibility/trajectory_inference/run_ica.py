from ica import ica1
import scanpy as sc
import h5py
import numpy as np
import os
from scgraphne.utils import setup_seed,louvain,calculate_metric

def preprocess(adata,scale=True):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)
    else:
        print('no scale!')
    return adata

for dataset in ['DPT', 'YAN', 'Deng', 'Buettner']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'ICA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset)), 'r') as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata.obs['true'] = Y
    adata = preprocess(adata)
    print(adata)

    A, S, W = ica1(adata.X, ncomp=64)
    adata.obsm['X_ica'] = A
    # louvain
    adata = louvain(adata, resolution=None, use_rep='X_ica')
    y_pred_l = np.asarray(adata.obs['louvain'], dtype=int)
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(adata.obs['true'], y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
             true=adata.obs['true'],
             umap=adata.obsm['X_umap'],
             latent=adata.obsm['X_ica'],
             data=adata.X,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))


