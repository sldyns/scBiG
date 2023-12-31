import os
import warnings

import h5py
import numpy as np
import scanpy as sc

from scbig import run_scbig, preprocess, setup_seed

warnings.filterwarnings('ignore')
import time
from memory_profiler import profile


@profile
def run(adata):
    adata = run_scbig(adata)
    from memory_profiler import memory_usage
    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print(max(mem_used))
    return adata, max(mem_used)

for dataset in ['2000', '4000', '8000', '16000', '32000', '64000']:
    print('---------------- data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scBiG'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X.astype('float'))
    adata.obs['cl_type'] = Y
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    adata = preprocess(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    print(adata)
    print("Sparsity of after preprocessing: ",
          np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    n_clusters = len(np.unique(Y))
    print("number of cell type:{}".format(n_clusters))

    ## training
    start_time = time.time()
    ## train
    adata, memory_usage = run(adata)
    print(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}_hvg2k.npz".format(dataset, dataset, method)),
             time=total_time, memory_usage=memory_usage)