import numpy as np
import h5py
import scanpy as sc
import scvi
from scgraphne.utils import setup_seed,louvain,calculate_metric
import os
import torch

for dataset in ['DPT', 'YAN', 'Deng', 'Buettner']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scVI'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    print(adata)

    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print(adata)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))


    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata,n_hidden=64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to_device(device)
    model.train()
    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    denoised = model.get_normalized_expression(library_size=1e4)

    # louvain
    adata = louvain(adata, resolution=None, use_rep='X_scVI')
    y_pred_l = np.asarray(adata.obs['louvain'], dtype=int)
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % ( nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
         true=Y,
         umap=adata.obsm['X_umap'],
         latent=adata.obsm['X_scVI'],
         data=denoised,
         louvain=np.array(adata.obs['louvain'].values.astype(int)))