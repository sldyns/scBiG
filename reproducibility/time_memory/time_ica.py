from ica import ica1
import numpy as np
import h5py
import scanpy as sc
from scgraphne.utils import setup_seed
import os
import time
from memory_profiler import profile

def preprocess(adata,scale=True):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)
    else:
        print('no scale!')
    return adata

@profile
def run_ica(adata):
    A, S, W = ica1(adata.X, ncomp=64)
    adata.obsm['latent'] = A
    return adata


for dataset in ['2000','4000','8000','16000','32000','64000']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'ICA'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata = preprocess(adata)
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))

    start_time = time.time()
    # train
    adata = run_ica(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
             time=total_time)