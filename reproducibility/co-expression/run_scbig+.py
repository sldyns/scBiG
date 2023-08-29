import os
import warnings
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from scbig import run_scbig, preprocess, setup_seed

warnings.filterwarnings('ignore')

for dataset in ['fetal_liver_atlas', 'Trachea']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scBiG'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    if dataset == 'fetal_liver_atlas':
        adata0 = sc.read_h5ad(os.path.join(dir0, 'datasets/co-expression/{}/sample_fetal_liver_atlas_dataset.h5ad').format(dataset))
        print(adata0)
        count = np.array(adata0.raw.X.todense())
        adata = sc.AnnData(count)
        adata.var_names = adata0.raw.var_names
        adata.obs['Cell Type'] = np.array(adata0.obs['Cell.Labels'].values)
        print("number of cell type: {}".format(len(np.unique(adata.obs['Cell Type']))))
        adata.var['feature_types'] = adata.var_names
        print(adata)

    else:
        with h5py.File(os.path.join(dir0, 'datasets/co-expression/{}/{}.h5').format(dataset, dataset)) as data_mat:
            X = np.array(data_mat['X'])
            X = np.ceil(X).astype(np.float)
        adata = sc.AnnData(X)
        gene = pd.read_csv(os.path.join(dir0, 'datasets/co-expression/{}/{}_geneset.csv').format(dataset, dataset),
                           header=0).values.squeeze()
        adata.var['feature_types'] = gene
        celltype = pd.read_csv(
            os.path.join(dir0, 'datasets/co-expression/{}/{}_celltype.csv').format(dataset, dataset),
            header=None).values
        celltype = np.array(celltype).squeeze()
        adata.var_names = gene
        adata.obs['Cell Type'] = celltype
        print("number of cell type: {}".format(len(np.unique(adata.obs['Cell Type']))))
        print(adata)

    adata = preprocess(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable]
    print(adata)

    ###training
    adata, record = run_scbig(adata, cl_type='Cell Type', gene_similarity=True,
                                  sample_rate=0.5, resolution=0.5, return_all=True)

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

    np.savez(os.path.join(dir0,"results/co-expression/{}_scBiG+.npz".format(dataset)),
         data=adata.X,
         E_feature=adata.varm['feat'],
         cell_type=adata.obs['Cell Type'],
         gene_name=adata.var_names)