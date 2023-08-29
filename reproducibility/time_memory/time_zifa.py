import os
import random
import time

import h5py
import numpy as np
import scanpy as sc
import tensorflow as tf
import sys
sys.path.append('../pkgs/ZIFA/')
from ZIFA import block_ZIFA
from memory_profiler import profile


def seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


@profile
def run_zifa(adata):
    Z, model_params = block_ZIFA.fitModel(X, 64, n_blocks=None)
    adata.obsm['feat'] = Z
    from memory_profiler import memory_usage
    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print(max(mem_used))
    return adata, max(mem_used)

for dataset in ['2000', '4000', '8000', '16000', '32000', '64000']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'ZIFA'
    dir0 = '../'

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()
    adata = sc.AnnData(X.astype('float'))
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))
    X = np.log2(1 + X)

    start_time = time.time()
    # train
    adata, memory_usage = run_zifa(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
             time=total_time, memory_usage=memory_usage)
