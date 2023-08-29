import numpy as np
import scanpy as sc
from scbig.utils import setup_seed,calculate_metric
import os

for dataset in ['DPT', 'YAN', 'Deng']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    dir0 = '../'
    method = 'Seurat'
    r = np.load(os.path.join(dir0, 'results/trajectory_inference/{}/embedding_{}_{}.npz'.format(dataset, dataset, method)))
    Y = r['true'],
    latent = r['latent']
    data = r['data'],
    print(data[0].shape)

    adata = sc.AnnData(data[0])
    print(adata)
    adata.obsm['feat'] = latent
    Y = Y[0].squeeze()

    sc.pp.neighbors(adata, use_rep='feat')
    sc.tl.louvain(adata, resolution=1)
    y_pred = np.array(adata.obs['louvain'])
    nmi_l, ari_l = calculate_metric(Y, y_pred)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

    sc.tl.umap(adata)

    np.savez(os.path.join(dir0, "results/trajectory_inference/{}/{}_{}.npz".format(dataset, dataset, method)),
             true=Y,
             umap=adata.obsm['X_umap'],
             latent=latent,
             data=data[0],
             louvain=np.array(adata.obs['louvain'].values.astype(int)))