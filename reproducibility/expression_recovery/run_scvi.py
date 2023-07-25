import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import scvi
from scgraphne import setup_seed
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
import os

for sparity in ['0.8','0.85','0.9','0.95']:
    print('----------------sparity level of data: {} ----------------- '.format(sparity))
    setup_seed(0)
    method = 'scVI'
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

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    print(adata)

    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()
    ff = np.sum(adata.X, axis=1) /1e4
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print(adata)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))

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

    ###training
    scvi.data.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata)
    model.train()
    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=1e4)
    print(adata)

    YPred = adata.layers["scvi_normalized"]
    Pred = np.dot(np.diag(ff),YPred)
    X_Impute = np.array(Pred).reshape(-1)
    X_true = np.array(data.values).reshape(-1)

    index = np.arange(X_true.shape[0])
    dropout = index[X_true != X_obs]
    rmse = float('%.4f' % sqrt(mean_squared_error(X_Impute, X_true)))
    rmse_false = float('%.4f' % sqrt(mean_squared_error(X_Impute[dropout], X_true[dropout])))

    pcc = float('%.4f' % pearsonr(X_Impute, X_true)[0])
    pcc_false = float('%.4f' % pearsonr(X_Impute[dropout], X_true[dropout])[0])
    print('RMSE_false: {:.4f}, PCC_false: {:.4f}, RMSE: {:.4f},PCC: {:.4f}'.format(rmse_false, pcc_false, rmse, pcc))

    np.savez(os.path.join(dir0, "results/expression_recovery/{}/result_{}_{}.npz".format(sparity, sparity, method)),
             rmsed=rmse_false, pccd=pcc_false,
             rmse=rmse, pcc=pcc)


    print(rmse_false)
    print(pcc_false)
    print(rmse)
    print(pcc)


