import os
import warnings

import h5py
import numpy as np
import scanpy as sc

from scbig import run_scbig, preprocess, setup_seed

warnings.filterwarnings('ignore')

for dataset in ['DPT', 'YAN', 'Deng']:
    print('---------------- data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scBiG'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X.astype('float'))
    adata.obs['cl_type'] = Y
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    adata = preprocess(adata)
    print(adata)
    n_clusters = len(np.unique(Y))
    print("number of cell type:{}".format(n_clusters))

    ## training
    adata, record = run_scbig(adata, cl_type='cl_type', return_all=True)

    final_ari_l = record['ari_l'][-1]
    final_nmi_l = record['nmi_l'][-1]
    n_pred = len(np.unique(np.array(adata.obs['louvain'])))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
             umap=adata.obsm['X_umap'],
             true=adata.obs['cl_type'],
             latent=adata.obsm['feat'],
             data=adata.X,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))
