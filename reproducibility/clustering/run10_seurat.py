import numpy as np
import scanpy as sc
from scbig.utils import setup_seed,calculate_metric
import os

for dataset in ['10X_PBMC','mouse_bladder_cell','mouse_ES_cell','human_kidney_counts','Adam','Human_pancreatic_islets','Macosko_mouse_retina']:
    print('----------------real data: {} ----------------- '.format(dataset))
    setup_seed(0)
    method = 'Seurat'
    dir0 = '../'

    NMI_l, ARI_l, N = [], [], []
    times = 10
    for t in range(times):
        print('----------------times: %d ----------------- ' % int(t + 1))
        r = np.load(os.path.join(dir0, 'results/clustering/{}/embedding_{}_{}_{}.npz'.format(dataset, t + 1, dataset, method)))
        latent = r['latent'],
        Y = r['true']

        print(latent[0].shape)
        print(Y)

        adata = sc.AnnData(latent[0])
        print(adata)
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.louvain(adata)
        y_pred = np.array(adata.obs['louvain'])
        n_pred = len(np.unique(y_pred))
        nmi_l, ari_l = calculate_metric(Y, y_pred)
        print('Clustering Louvain: NMI= %.4f, ARI= %.4f' % (nmi_l, ari_l))

        NMI_l.append(nmi_l), ARI_l.append(ari_l), N.append(n_pred)

    np.savez(os.path.join(dir0, "results/clustering/{}/result_{}_{}.npz".format(dataset, dataset, method)),
             aril=ARI_l, nmil=NMI_l)

    print(NMI_l)
    print(ARI_l)
    print(N)