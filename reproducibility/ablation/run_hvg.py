import os
import warnings

import h5py
import numpy as np
import scanpy as sc

from scbig import run_scbig, preprocess, read_data, setup_seed, sample

warnings.filterwarnings('ignore')

for dataset in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'human_kidney_counts', 'Adam',
                'Human_pancreatic_islets']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scBiG'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    if dataset in ['Adam']:
        mat, obs, var, uns = read_data(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset)), sparsify=False,
                                       skip_exprs=False)
        X0 = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        Y0 = cell_label

    else:
        with h5py.File(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset))) as data_mat:
            X0 = np.array(data_mat['X'])
            Y0 = np.array(data_mat['Y'])
            X0 = np.ceil(X0).astype(np.int_)
            Y0 = np.array(Y0).astype(np.int_).squeeze()

    for highly_variable_genes in [2000, 3000, 5000]:
        print('----------------use highly variable genes: {} ----------------- '.format(highly_variable_genes))

        Final_ari_l, Final_nmi_l = [], []
        N = []
        times = 10
        for t in range(times):
            print('----------------times: %d ----------------- ' % int(t + 1))

            adata = sc.AnnData(X0)
            print(adata)
            print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
            ##sample
            seed = 10 * t
            X, Y = sample(X0, Y0, seed)
            adata = sc.AnnData(X.astype('float'))
            adata.obs['cl_type'] = Y
            adata = preprocess(adata)

            sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=highly_variable_genes)
            adata = adata[:, adata.var.highly_variable]
            print(adata)
            print("Sparsity of after preprocessing: ",
                  np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
            n_clusters = len(np.unique(Y))

            ###training
            adata, record = run_scbig(adata, cl_type='cl_type', return_all=True)

            print(adata)

            final_ari_l = record['ari_l'][-1]
            final_nmi_l = record['nmi_l'][-1]
            n_pred = len(np.unique(np.array(adata.obs['louvain'])))
            Final_ari_l.append(final_ari_l), Final_nmi_l.append(final_nmi_l), N.append(n_pred)

        ## save results
        np.savez(os.path.join(dir0,
                              "results/ablation/{}/hvg{}_{}_{}.npz".format(dataset, highly_variable_genes, dataset,
                                                                           method)),
                 aril=Final_ari_l, nmil=Final_nmi_l)

        print(Final_nmi_l)
        print(Final_ari_l)
        print(N)
