import numpy as np
import scanpy as sc
import h5py
import os
from scgraphne import run_scgraphne, preprocess, setup_seed
import warnings
warnings.filterwarnings('ignore')
import time
from memory_profiler import profile

@profile
def run(adata):
    adata = run_scgraphne(adata)
    return adata

for dataset in ['2000','4000','8000','16000','32000','64000']:
    print('---------------- data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'scGraphNE'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    print(adata)
    adata = preprocess(adata)
    print(adata)
    print("Sparsity of after preprocessing: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    n_clusters = len(np.unique(Y))
    print("number of cell type:{}".format(n_clusters))

    ## training
    start_time = time.time()
    ## train
    adata = run(adata)
    print(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
             time=total_time)