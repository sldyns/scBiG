# -*- coding: utf-8 -*-
import numpy as np
from vasc import vasc
from helpers import clustering,measure,print_2D
from config import config
import os
import scanpy as sc
import h5py
import tensorflow as tf
import random
def seed(SEED):
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

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

if __name__ == '__main__':
    for dataset in ['DPT', 'YAN', 'Deng', 'Buettner']:
        print('----------------real data: {} ----------------- '.format(dataset))
        seed(0)
        method = 'VASC'
        dir0 = '../'
        PREFIX = dataset
        with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset)), 'r') as data_mat:
            X = np.array(data_mat['X'])
            Y = np.array(data_mat['Y'])
            X = np.ceil(X).astype(np.int_)
            Y = np.array(Y).astype(np.int_).squeeze()
        idents = Y
        adata = sc.AnnData(X)
        adata.obs['true'] = Y
        adata = preprocess(adata)
        print(adata)
        ####
        expr = adata.X
        label = adata.obs['true']

        n_cell,_ = expr.shape
        if n_cell > 150:
            batch_size=config['batch_size']
        else:
            batch_size=32

        for i in range(1):
            print("Iteration:"+str(i))
            res = vasc( expr,
                        epoch=100,
                        var=False,
                        latent=64,
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

        adata.obsm['X_vasc'] = res

        sc.pp.neighbors(adata, use_rep="X_vasc")
        sc.tl.umap(adata)

        np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
                 true=adata.obs['true'],
                 umap=adata.obsm['X_umap'],
                 latent=adata.obsm['X_vasc'],
                 data=adata.X,
                 louvain=pred)

