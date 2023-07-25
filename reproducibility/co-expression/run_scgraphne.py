import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import os
from scgraphne import run_scgraphne, preprocess, setup_seed

import warnings
warnings.filterwarnings('ignore')

for dataset in ['fetal_liver_atlas','Trachea']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scGraphNE'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    if dataset == 'fetal_liver_atlas':
        adata0 = sc.read_h5ad(os.path.join(dir0, 'datasets/co-expression/{}/sample_fetal_liver_atlas_dataset.h5ad').format(dataset))
        Y = pd.read_csv(os.path.join(dir0, 'datasets/co-expression/{}/cell_type.csv').format(dataset), header=None)
        Y = np.array(Y).astype(np.int_).squeeze()
        print(adata0)
        count = np.array(adata0.raw.X.todense())
        adata = sc.AnnData(count)
        adata.var_names = adata0.raw.var_names
        adata.obs['Cell Type'] = np.array(adata0.obs['Cell.Labels'].values)
        adata.obs['cl_type'] = Y
        n_clusters = len(np.unique(Y))
        print("number of cell type:{}".format(n_clusters))
        adata.var['feature_types'] = adata.var_names
        print(adata)

    else:
        with h5py.File(os.path.join(dir0, 'datasets/co-expression/{}/{}.h5').format(dataset,dataset)) as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_).squeeze()
        adata = sc.AnnData(X)
        gene = pd.read_csv(os.path.join(dir0, 'datasets/co-expression/{}/{}_geneset.csv').format(dataset,dataset),
                           header=0).values.squeeze()
        adata.var['feature_types'] = gene
        celltype = pd.read_csv(os.path.join(dir0, 'datasets/co-expression/{}/{}_celltype.csv').format(dataset,dataset), header=None).values
        celltype = np.array(celltype).squeeze()
        Y = pd.read_csv(os.path.join(dir0, 'datasets/co-expression/{}/{}_label.csv').format(dataset,dataset), header=None).values
        Y = np.array(Y).squeeze()
        adata.var_names = gene
        adata.obs_names = celltype
        adata.obs['Cell Type'] = celltype
        adata.obs['cl_type'] = Y
        print(adata)

    adata = preprocess(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable]
    print(adata)

    ###training
    adata, record = run_scgraphne(adata, cl_type='cl_type', resolution=0.5, return_all=True)
    print(adata)

    final_ari_l = record['ari_l'][-1]
    final_nmi_l = record['nmi_l'][-1]
    n_pred = len(np.unique(np.array(adata.obs['louvain'])))

    ##save results
    sc.pp.neighbors(adata, use_rep="feat")
    sc.tl.umap(adata)

    print(final_nmi_l)
    print(final_ari_l)
    print(n_pred)

    np.savez(os.path.join(dir0,"results/co-expression/{}_scGraphNE.npz".format(dataset)),
         data = adata.X,
         umap = adata.obsm['X_umap'],
         E_sample = adata.obsm['feat'],
         E_feature = adata.varm['feat'],
         cell_type = adata.obs['Cell Type'],
         gene_name = adata.var_names,
         true = np.array(adata.obs['cl_type'].values.astype(int)),
         louvain = np.array(adata.obs['louvain'].values.astype(int)))