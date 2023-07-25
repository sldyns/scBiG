from pca import pca
import scanpy as sc
import h5py
import numpy as np
import os
from scgraphne.utils import setup_seed,read_data,louvain,calculate_metric

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

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'PCA'
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
    adata = preprocess(adata)
    print(adata)

    model = pca(n_components=64)
    X_reduction = model.fit_transform(adata.X)
    adata.obsm['feat'] = X_reduction['PC'].values
    # louvain
    adata = louvain(adata, resolution=1,use_rep='feat')
    y_pred = np.array(adata.obs['louvain'])
    nmi_l, ari_l = calculate_metric(Y, y_pred)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)
    np.savez(os.path.join(dir0,"results/visualization/{}/record_{}_{}.npz".format(dataset,dataset,method)),
             ari=ari_l, nmi=nmi_l,
             umap=adata.obsm['X_umap'],
             true=Y,
             louvain=np.array(adata.obs['louvain'].values.astype(int)))