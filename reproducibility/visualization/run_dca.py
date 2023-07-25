import numpy as np
import scanpy as sc
from dca.api import dca
import h5py
from scgraphne.utils import read_data,setup_seed,louvain,calculate_metric
import os

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'DCA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

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

    adata = sc.AnnData(X)
    print(adata)
    sc.pp.filter_genes(adata, min_cells=1)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    print(adata)

    dca(adata, mode='latent',ae_type='zinb',threads=1)

    # louvain
    adata = louvain(adata, resolution=1,use_rep='X_dca')
    y_pred_l = np.array(adata.obs['louvain'])
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0,"results/visualization/{}/record_{}_{}.npz".format(dataset,dataset,method)),
             ari=ari_l, nmi=nmi_l,
             umap=adata.obsm['X_umap'],
             true=np.array(adata.obs['cl_type'].values.astype(int)),
             louvain=np.array(adata.obs['louvain'].values.astype(int)))

    print(nmi_l)
    print(ari_l)
    print(n_pred)
