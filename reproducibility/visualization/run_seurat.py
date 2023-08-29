import numpy as np
import scanpy as sc
from scbig.utils import setup_seed,calculate_metric
import os

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    setup_seed(0)
    dir0 = '../'
    method = 'Seurat'
    print('----------------real data: {} ----------------- '.format(dataset))
    r = np.load(os.path.join(dir0,'results/visualization/{}/embedding_{}_{}.npz'.format(dataset,dataset,method)))
    Y = r['true']
    print(Y)
    print(Y.shape)
    latent = r['latent']
    print(latent)
    adata = sc.AnnData(latent)
    print(adata)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.louvain(adata)
    y_pred = np.array(adata.obs['louvain'])
    nmi_l, ari_l = calculate_metric(Y, y_pred)
    print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))
    sc.tl.umap(adata)

    np.savez(os.path.join(dir0,"results/visualization/{}/record_{}_{}.npz".format(dataset,dataset,method)),
             ari=ari_l, nmi=nmi_l,
             umap=adata.obsm['X_umap'],
             true=np.array(adata.obs['cl_type'].values.astype(int)),
             louvain=np.array(adata.obs['louvain'].values.astype(int)))



