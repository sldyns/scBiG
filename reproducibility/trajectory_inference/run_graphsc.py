import numpy as np
import dgl
import torch
import torch.nn.functional as F
import h5py
import sys
sys.path.append('../pkgs/graph-sc/')
import train
import models
device = train.get_device()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from scbig.utils import setup_seed,louvain,calculate_metric
import os

for dataset in ['DPT', 'YAN', 'Deng']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'graph-sc'
    dir0 = '../'
    dir1 = '{}'.format(dataset)

    with h5py.File(os.path.join(dir0, 'datasets/trajectory/{}.h5'.format(dataset)), 'r') as data_mat:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        X = np.ceil(X).astype(np.int_)
        Y = np.array(Y).astype(np.int_).squeeze()

    adata = sc.AnnData(X)
    adata.obs['cl_type'] = Y
    print("Sparsity: ", np.where(X == 0)[0].shape[0] / (X.shape[0] * X.shape[1]))
    normalize_weights = "log_per_cell"

    n_layers = 1
    hidden_dim = 200
    hidden = [300]
    nb_genes = 3000
    activation = F.relu
    epochs = 10
    batch_size = 128
    pca_size = 50

    # remove less variable genes
    genes_idx, cells_idx = train.filter_data(X, highly_genes=nb_genes)
    X = X[cells_idx][:, genes_idx]
    Y = Y[cells_idx]
    n_clusters = len(np.unique(Y))

    # create graph
    graph = train.make_graph(
        X,
        Y,  # Pass None of Y is not available for validation
        dense_dim=pca_size,
        normalize_weights=normalize_weights,
    )
    labels = graph.ndata["label"]
    train_ids = np.where(labels != -1)[0]

    # create training data loader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        graph,
        train_ids,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    # create model
    model = models.GCNAE(
        in_feats=pca_size,
        n_hidden=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        dropout=0.1,
        hidden=hidden,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    # train model
    results = train.train(model,
                          optim,
                          epochs,
                          dataloader,
                          n_clusters,
                          plot=False,
                          save=True,
                          cluster=["KMeans", "Leiden"])

    print(results.keys())
    print(results["features"].shape)
    adata.obsm['feat'] = results["features"]

    # louvain
    adata = louvain(adata, resolution=1, use_rep='feat')
    y_pred_l = np.array(adata.obs['louvain'])
    n_pred = len(np.unique(y_pred_l))
    nmi_l, ari_l = np.round(calculate_metric(Y, y_pred_l), 4)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)
    print(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
                 true=adata.obs['cl_type'],
                 umap=adata.obsm['X_umap'],
                 latent=adata.obsm['feat'],
                 data=adata.X,
                 louvain=np.array(adata.obs['louvain'].values.astype(int)))


