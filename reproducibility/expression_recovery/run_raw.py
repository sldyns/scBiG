import os
from math import sqrt

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from scbig import setup_seed

for sparity in ['0.8', '0.85', '0.9', '0.95']:
    print('----------------sparity level of data: {} ----------------- '.format(sparity))
    setup_seed(0)
    method = 'Raw'
    dir0 = '../'
    dir1 = '{}'.format(sparity)
    dir2 = 'data_{}.h5'.format(sparity)
    data_mat = h5py.File(os.path.join(dir0, 'datasets/sim', dir2), "r")
    X = np.array(data_mat['X'])
    Xt = np.array(data_mat['X_true'])
    Y = np.array(data_mat['Y'])
    n_clusters = len(np.unique(Y))

    adata = sc.AnnData(X.astype('float'))
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
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

    X_Impute = np.array(adata.X).reshape(-1)
    X_true = np.array(data.values).reshape(-1)

    index = np.arange(X_true.shape[0])
    dropout = index[X_true != X_obs]

    rmse = float('%.4f' % sqrt(mean_squared_error(X_Impute, X_true)))
    rmse_false = float('%.4f' % sqrt(mean_squared_error(X_Impute[dropout], X_true[dropout])))

    pcc = float('%.4f' % pearsonr(X_Impute, X_true)[0])
    pcc_false = float('%.4f' % pearsonr(X_Impute[dropout], X_true[dropout])[0])
    print('RMSE_false: {:.4f}, PCC_false: {:.4f}, RMSE: {:.4f},PCC: {:.4f}'.format(rmse_false, pcc_false, rmse, pcc))

    ##save results
    np.savez(os.path.join(dir0, "results/expression_recovery/{}/result_{}_{}.npz".format(sparity, sparity, method)),
             rmsed=rmse_false, pccd=pcc_false,
             rmse=rmse, pcc=pcc)

    print(rmse_false)
    print(pcc_false)
    print(rmse)
    print(pcc)
