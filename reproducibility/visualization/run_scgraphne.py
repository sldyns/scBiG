import numpy as np
import scanpy as sc
import h5py
import os
from scgraphne import run_scgraphne, preprocess, read_data, setup_seed
import warnings
warnings.filterwarnings('ignore')

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scGraphNE'
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
        with h5py.File(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset))) as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    adata = preprocess(adata)
    print(adata)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    ###training
    adata, record = run_scgraphne(adata, cl_type='cl_type', return_all=True)
    print(adata)

    final_ari_l = record['ari_l'][-1]
    final_nmi_l = record['nmi_l'][-1]
    n_pred = len(np.unique(np.array(adata.obs['louvain'])))

    ## save results
    sc.tl.umap(adata)
    np.savez(os.path.join(dir0,"results/visualization/{}/record_{}_{}.npz".format(dataset,dataset,method)),
          ari=final_ari_l, nmi=final_nmi_l,
          umap=adata.obsm['X_umap'],
          true=np.array(adata.obs['cl_type'].values.astype(int)),
          louvain=np.array(adata.obs['louvain'].values.astype(int)))

    print(final_nmi_l)
    print(final_ari_l)
    print(n_pred)