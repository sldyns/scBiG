import os
import time

import h5py
import numpy as np
import scanpy as sc
from dca.api import dca
from memory_profiler import profile

from scbig import setup_seed


@profile
def run_dca(adata):
    dca(adata, mode='latent', ae_type='zinb', threads=1)
    from memory_profiler import memory_usage
    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print(max(mem_used))
    return adata, max(mem_used)

for dataset in ['2000', '4000', '8000', '16000', '32000', '64000']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'DCA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X.astype('float'))
    print(adata)
    sc.pp.filter_genes(adata, min_cells=1)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    print(adata)

    start_time = time.time()
    # train
    adata, memory_usage = run_dca(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
             time=total_time, memory_usage=memory_usage)




