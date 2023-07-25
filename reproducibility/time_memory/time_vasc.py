# -*- coding: utf-8 -*-
import numpy as np
from vasc import vasc
from helpers import clustering,measure,print_2D
from config import config
import os
import scanpy as sc
from scgraphne.utils import setup_seed
import h5py
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
def run_vasc(adata):
    for i in range(1):
        print("Iteration:"+str(i))
        res = vasc( expr,
                    epoch=300,
                    var=False,
                    latent=config['latent'],
                    annealing=False,
                    batch_size=batch_size,
                    prefix=PREFIX,
                    label=label,
                    scale=config['scale'],
                    patience=config['patience']
                )

    print("============SUMMARY==============")
    k = len(np.unique(label))
    print("cluster of k:"+str(k))

    pred,si = clustering(res,k=k)
    NMI,ARI = measure(pred,label)
    print({'NMI':NMI,'ARI':ARI})
    return adata,NMI,ARI,res,pred


if __name__ == '__main__':
    for dataset in ['2000','4000','8000','16000','32000','64000']:
        print('----------------real data: {} ----------------- '.format(dataset))
        setup_seed(0)
        method = 'VASC'
        dir0 = '../'
        PREFIX = dataset
        with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_).squeeze()

        idents = Y
        adata = sc.AnnData(X)
        adata = preprocess(adata)
        print(adata)
        ####
        expr = adata.X
        label = Y

        n_cell,_ = expr.shape
        if n_cell > 150:
            batch_size=config['batch_size']
        else:
            batch_size=32

        start_time = time.time()
        # train
        adata,NMI,ARI,res,pred = run_vasc(adata)
        end_time = time.time()
        total_time = end_time - start_time
        print("Run Done. Total Running Time: %s seconds" % (total_time))

        np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
                 time=total_time)