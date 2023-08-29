import warnings

import numpy as np
import scanpy as sc
import tensorflow as tf
import sys
sys.path.append('../pkgs/scGAE/')

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# scGAE modules
from scgae import SCGAE
from preprocessing import *
from utils import *
from losses import *
from clustering import *
import h5py
import os
import random
import time
from memory_profiler import profile


def seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def preprocess(adata, scale=True):
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
def run_scgae(adata):
    # Compute adjacency matrix and normalized adjacency matrix
    adj, adj_n = get_adj(count, k=160)

    model = SCGAE(count, adj, adj_n, hidden_dim=120, latent_dim=15, decA="DBL", layer_enc="GAT")
    model.train(epochs=80, W_a=0.4, W_x=1)  ##80

    # Genertate latent embedding by tSNE
    Y = model.embedding(count, adj_n)

    ####### Get initialized clustering center ######
    # Louvain clustering
    centers, labels = get_centers_louvain(Y, adj)
    # Spectral clustering
    # centers, labels = get_centers_spectral(Y, adj)
    model.clustering_train(centers=centers, W_a=0.4, W_x=1, W_c=1.5, epochs=40)  # 40

    # Genertate latent embedding and visualization after clustering-training
    Yc = model.embedding(count, adj_n)
    Y2c = doumap(Yc, dim=2)
    adata.obsm['feat'] = Yc

    from memory_profiler import memory_usage
    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print(max(mem_used))

    return adata, Y2c, max(mem_used)


for dataset in ['2000', '4000', '8000', '16000', '32000', '64000']:
    print('----------------real data: {} ----------------- '.format(dataset))
    seed(0)
    method = 'scGAE'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/time/data_cell{}.h5'.format(dataset))) as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    idents = Y
    adata = sc.AnnData(X.astype('float'))
    adata = preprocess(adata)
    count = adata.X
    start_time = time.time()
    # train
    adata, Y2c, memory_usage = run_scgae(adata)
    end_time = time.time()
    total_time = end_time - start_time
    print("Run Done. Total Running Time: %s seconds" % (total_time))

    np.savez(os.path.join(dir0, "results/time_memory/{}/record_cell{}_{}.npz".format(dataset, dataset, method)),
             time=total_time, memory_usage=memory_usage)

