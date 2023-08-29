import os

import h5py
import numpy as np
import scanpy as sc
from dca.api import dca

from scbig.utils import read_data, setup_seed, sample, louvain, calculate_metric

for dataset in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'human_kidney_counts', 'Adam',
                'Human_pancreatic_islets', 'Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'DCA'
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

    NMI_l, ARI_l, N = [], [], []
    times = 10
    for t in range(times):
        print('----------------times: %d ----------------- ' % int(t + 1))
        ##sample
        seed = 10 * t
        X, Y = sample(X0, Y0, seed)
        adata = sc.AnnData(X.astype('float'))
        print(adata)
        sc.pp.filter_genes(adata, min_cells=1)
        print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
        adata.obs['cl_type'] = Y
        n_clusters = len(np.unique(Y))
        print(adata)

        ##DCA training
        dca(adata, mode='latent', ae_type='zinb', threads=1)
        print(adata)

        # louvain
        adata = louvain(adata, resolution=1, use_rep='X_dca')
        y_pred_l = np.array(adata.obs['louvain'])
        n_pred = len(np.unique(y_pred_l))
        nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
        print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

        NMI_l.append(nmi_l), ARI_l.append(ari_l), N.append(n_pred)

    np.savez(os.path.join(dir0, "results/clustering/{}/result_{}_{}.npz".format(dataset, dataset, method)),
             aril=ARI_l, nmil=NMI_l)

    print(NMI_l)
    print(ARI_l)
    print(N)
