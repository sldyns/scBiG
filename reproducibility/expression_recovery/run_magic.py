import os
from math import sqrt

import h5py
import magic
import numpy as np
import pandas as pd
import scanpy as sc
import scprep
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error

from scbig import setup_seed

for sparity in ['0.8', '0.85', '0.9', '0.95']:
    print('----------------sparity level of data: {} ----------------- '.format(sparity))
    setup_seed(0)
    method = 'MAGIC'
    dir0 = '../'
    dir1 = '{}'.format(sparity)
    dir2 = 'data_{}.h5'.format(sparity)

    with h5py.File(os.path.join(dir0, 'datasets/sim', dir2), 'r') as data_mat:
        X = np.array(data_mat['X'])
        Xt = np.array(data_mat['X_true'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Xt = np.ceil(Xt).astype(np.int_)
        Y = np.array(Y).astype(np.int_)

    adata = sc.AnnData(X.astype('float'))
    adata.obs['cl_type'] = Y
    emt_data_norm, libsize = scprep.normalize.library_size_normalize(adata.X, return_library_size=True)
    emt_data_norm = scprep.transform.sqrt(emt_data_norm)

    magic_op = magic.MAGIC()
    emt_magic = magic_op.fit_transform(emt_data_norm, genes="all_genes")
    adata = sc.AnnData(emt_magic)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    print(adata)

    ##true data
    adata0 = sc.AnnData(Xt)
    data = pd.DataFrame(Xt.T, index=list(adata0.var.index), columns=list(adata0.obs.index))
    data = data[list(adata.obs.index)].T
    data = data[list(adata.var.index)]
    print(data.shape)
    ##obsvered data
    data0 = pd.DataFrame(X.T, index=list(adata0.var.index), columns=list(adata0.obs.index))
    data0 = data0[list(adata.obs.index)].T
    data0 = data0[list(adata.var.index)]
    print(data0.shape)
    X_obs = np.array(data0).reshape(-1)

    YPred = pow(emt_magic, 2) / 1e4
    YPred = np.dot(np.diag(libsize), YPred)
    X_Impute = np.array(YPred).reshape(-1)
    X_true = np.array(data.values).reshape(-1)

    index = np.arange(X_true.shape[0])
    dropout = index[X_true != X_obs]
    rmse = float('%.4f' % sqrt(mean_squared_error(X_Impute, X_true)))
    rmse_false = float('%.4f' % sqrt(mean_squared_error(X_Impute[dropout], X_true[dropout])))

    pcc = float('%.4f' % pearsonr(X_Impute, X_true)[0])
    pcc_false = float('%.4f' % pearsonr(X_Impute[dropout], X_true[dropout])[0])

    spear = float('%.4f' % spearmanr(X_Impute, X_true)[0])
    spear_false = float('%.4f' % spearmanr(X_Impute[dropout], X_true[dropout])[0])

    print('RMSE_false: {:.4f}, PCC_false: {:.4f},Spearman_false: {:.4f},RMSE: {:.4f},PCC: {:.4f}, Spearman: {:.4f}'.format(
            rmse_false, pcc_false, spear_false, rmse, pcc, spear))

    ##save results
    np.savez(os.path.join(dir0, "results/expression_recovery/{}/result_{}_{}.npz".format(sparity, sparity, method)),
             rmsed=rmse_false, pccd=pcc_false, speard=spear_false,
             rmse=rmse, pcc=pcc, spear=spear)

    print(rmse_false)
    print(pcc_false)
    print(spear_false)
    print(rmse)
    print(pcc)
    print(spear)

