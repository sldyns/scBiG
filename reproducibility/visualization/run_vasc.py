# -*- coding: utf-8 -*-
import numpy as np
from vasc import vasc
from helpers import clustering,measure,print_2D
from config import config
import os
import scanpy as sc
from scgraphne.utils import setup_seed,read_data
import h5py

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
    for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
        print('----------------real data: {} ----------------- '.format(dataset))
        setup_seed(0)
        method = 'VASC'
        dir0 = '../'
        PREFIX = dataset
        if dataset in ['Adam']:
            mat, obs, var, uns = read_data(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset)), sparsify=False,
                                           skip_exprs=False)
            X = np.array(mat.toarray())
            cell_name = np.array(obs["cell_type1"])
            cell_type, cell_label = np.unique(cell_name, return_inverse=True)
            Y = cell_label

        else:
            with h5py.File(os.path.join(dir0, 'datasets/real/{}.h5'.format(dataset)), 'r') as data_mat:
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
        #expr = np.exp(expr) - 1
        #expr = expr / np.max(expr)

        #latent = 2
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

        np.savez(os.path.join(dir0,"results/visualization/{}/record_{}_{}.npz".format(dataset,dataset,method)),
                 ari=ARI, nmi=NMI,
                 umap=res,
                 true=label,
                 louvain=pred)
