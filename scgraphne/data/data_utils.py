import scanpy as sc
import numpy as np
import pandas as pd
import dgl
import torch
from .graph import construct_gene_graph, add_degree

def preprocess(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=200)

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['cs_factor'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    gs_factor = np.max(adata.X, axis=0, keepdims=True)
    adata.var['gs_factor'] = gs_factor.reshape(-1)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def make_graph(adata, raw_exp=False, gene_similarity=False):
    X = adata.X
    num_cells, num_genes = X.shape

    # Make expressioin/train graph
    num_nodes_dict = {'cell': num_cells, 'gene': num_genes}
    exp_train_cell, exp_train_gene = np.where(X > 0)
    unexp_edges = np.where(X == 0)

    # expression edges
    exp_edge_dict = {
        ('cell', 'exp', 'gene'): (exp_train_cell, exp_train_gene),
        ('gene', 'reverse-exp', 'cell'): (exp_train_gene, exp_train_cell)
    }

    coexp_edges, uncoexp_edges = None, None
    if gene_similarity:
        coexp_edges, uncoexp_edges = construct_gene_graph(X)
        exp_edge_dict[('gene', 'co-exp', 'gene')] = coexp_edges
    
    # expression encoder/decoder graph
    enc_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)

    exp_edge_dict.pop(('gene', 'reverse-exp', 'cell'))
    dec_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)
    
    # add degree to cell/gene nodes
    add_degree(enc_graph, ['exp'] + (['co-exp'] if gene_similarity else []))

    # If use ZINB decoder, add size factor to cell/gene nodes
    if raw_exp:
        Raw = pd.DataFrame(adata.raw.X, index=list(adata.raw.obs_names), columns=list(adata.raw.var_names))
        X = Raw[list(adata.var_names)].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1,1)
        dec_graph.nodes['cell'].data['cs_factor'] = torch.Tensor(adata.obs['cs_factor']).reshape(-1, 1)
        dec_graph.nodes['gene'].data['gs_factor'] = torch.Tensor(adata.var['gs_factor']).reshape(-1, 1)

    else:
        ## Deflate the edge values of the bipartite graph to between 0 and 1
        X = X / adata.var['gs_factor'].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)

    return adata, exp_value, enc_graph, dec_graph, unexp_edges, coexp_edges, uncoexp_edges