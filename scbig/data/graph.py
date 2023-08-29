import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def construct_gene_graph(gex_features, corr_method='cosine', corr_threshold=0.9):
    """Generate nodes, edges and edge weights for dataset.

    Parameters
    ----------
    gex_features: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.
    """

    if corr_method == 'pearson':
        corr = np.abs(np.corrcoef(gex_features, rowvar=False))
    elif corr_method == 'cosine':
        corr = cosine_similarity(gex_features.T)

    row, col = np.diag_indices_from(corr)
    corr[row, col] = 0

    coexp_edges = np.where(abs(corr) > corr_threshold)
    uncoexp_edges = np.where(abs(corr) < 1 - corr_threshold)
    # neg_idx = np.random.choice(len(nuu), 10*len(uu))
    # nuu, nvv = nuu[neg_idx], nvv[neg_idx]

    return coexp_edges, uncoexp_edges


def add_degree(graph, edge_types):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))
        return x.unsqueeze(1)

    cell_ci, gene_ci = _calc_norm(graph['reverse-exp'].in_degrees()), _calc_norm(graph['exp'].in_degrees())
    cell_cj, gene_cj = _calc_norm(graph['exp'].out_degrees()), _calc_norm(graph['reverse-exp'].out_degrees())
    graph.nodes['cell'].data.update({'ci': cell_ci, 'cj': cell_cj})
    graph.nodes['gene'].data.update({'ci': gene_ci, 'cj': gene_cj})

    if 'co-exp' in edge_types:
        gene_cii, gene_cjj = _calc_norm(graph['co-exp'].in_degrees()), _calc_norm(graph['co-exp'].out_degrees())
        graph.nodes['gene'].data.update({'cii': gene_cii, 'cjj': gene_cjj})
