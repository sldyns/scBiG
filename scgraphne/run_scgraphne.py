#!/usr/bin/env python
# coding: utf-8
import gc

import dgl
import numpy as np
import scanpy as sc
import torch
from torch import nn, optim

from .data import make_graph
from .model import scGraphNE
from .utils import calculate_metric, ZINBLoss, kmeans, louvain


def run_scgraphne(adata: sc.AnnData,
                  n_clusters=None,
                  cl_type=None,
                  gene_similarity: bool = False,
                  alpha=0.9,
                  n_layers: int = 2,
                  feats_dim: int = 64,
                  drop_out: float = 0.,
                  gamma: int = 1,
                  decoder='ZINB',
                  lr: float = 0.1,
                  iteration: int = 200,
                  log_interval: int = 5,
                  resolution: float = 1,
                  sample_rate: float = 0.1,
                  use_rep: str = 'feat',
                  verbose: bool = True,
                  return_all: bool = False,
                  impute: bool = False
                  ):
    """
        Train scGraphNE.
        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        n_clusters
            Number of true cell type. If provided, you can select KMeans for clustering.
        cl_type
            Cell type information. If provided, calculate ARI and NMI after clustering.
        gene_similarity
            If True,consider correlation between genes and change bipartite graph structure.
        alpha
            When considering correlation between genes, set the proportional coefficient of BCE loss.
        n_layers
            Number of graph convolution layers in encoder.
        feats_dim
            The embedding dimensions of cells or genes in latent space.
        drop_out
            The probability of randomly dropping node embedding values.
        gamma
            The proportional coefficient of ZINB loss.
        decoder
            Structure of decoder. Default is 'ZINB', optionally input 'Dot', 'GMF' or 'ZINB'.
        lr
            Learning rate for AdamOptimizer.
        iteration
            Number of total iterations in training.
        log_interval
            how many iterations to wait before logging training status.
        resolution
            The resolution parameter of sc.tl.louvain for clustering.Default is 1.
        sample_rate
            The edge sampling rate of bipartite graph in decoder.
        use_rep
            Use the indicated representation. 'X' or any key for .obsm is valid. Here we use the cell embedding in the hidden space and store it in adata.obsm['feat'].
        verbose
            If True, show details when running.
        return_all
            If True, no clustering is performed during model training to save runtime.
        impute
            Whether to output reconstructed gene expression matrix(optional). If True, return imputed expression value and store it in adata.obsm['imputed'].

        Returns
        -------
        adata
            AnnData object of scanpy package. Cell embedding and louvain clustering result will be stored
            in adata.obsm['feat'] and adata.obs['louvain']
        record
            The results recorded during each logging training, including loss of training, ARI of KMeans,
            NMI of KMeans, ARI of Louvain and NMI of Louvain.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert decoder in ['Dot', 'GMF', 'ZINB'], "Please choose decoder in ['Dot', 'GMF', 'ZINB']"
    assert sample_rate <= 1, "Please set 0<sample_rate<=1"

    ####################   Prepare data for training   ####################
    cell_type = adata.obs[cl_type].values if cl_type else None
    raw_exp = True if decoder == 'ZINB' else False
    n_cells, n_genes = adata.X.shape

    if cell_type is not None:
        n_clusters = len(np.unique(cell_type))

    adata, exp_value, enc_graph, exp_dec_graph_, unexp_edges, coexp_edges, uncoexp_edges = make_graph(adata, raw_exp,
                                                                                                      gene_similarity)

    n_pos_edges, n_neg_edges = int(sample_rate * len(exp_value)), int(sample_rate * len(exp_value))
    n_neg_genes = len(coexp_edges[0]) if gene_similarity else None
    enc_graph, exp_value = enc_graph.to(device), torch.tensor(exp_value, device=device)

    #######################   Prepare models   #######################
    model = scGraphNE(n_layers=n_layers,
                      n_cells=n_cells,
                      n_genes=n_genes,
                      drop_out=drop_out,
                      alpha=alpha,
                      gene_similarity=gene_similarity,
                      decoder=decoder,
                      feats_dim=feats_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.MSELoss() if decoder in ['Dot', 'GMF'] else ZINBLoss()
    gene_cls = nn.BCELoss() if gene_similarity else None
    #######################   Record values   #######################
    all_loss = []

    best_ari_k, best_ari_l = 0, 0
    best_nmi_k, best_nmi_l = 0, 0
    all_ari_k, all_ari_l = [], []
    all_nmi_k, all_nmi_l = [], []

    best_iter_k, best_iter_l = -1, -1
    count_loss_exp, count_loss_unexp, count_loss_gene, count_loss_ungene = 0, 0, 0, 0
    #######################   Start training model   #######################
    print(f"Start training on {device}...")

    all_exp_index, all_unexp_index = np.arange(len(exp_value)), np.arange(len(unexp_edges[0]))
    all_uncoexp_index = np.arange(len(uncoexp_edges[0])) if gene_similarity else None

    if sample_rate == 1:
        pos_graph, pos_value = exp_dec_graph_.to(device), exp_value

    ### Save time by not performing clustering if time would be recorded during training.
    for iter_idx in range(iteration):
        # Sample un-expressed / un-co-expressed edges, construct negative graph
        neg_edges = {}

        unexp_sample_index = np.random.choice(all_unexp_index, n_neg_edges)
        neg_edges[('cell', 'exp', 'gene')] = (unexp_edges[0][unexp_sample_index], unexp_edges[1][unexp_sample_index])
        if gene_similarity:
            uncoexp_sample_index = np.random.choice(all_uncoexp_index, n_neg_genes)
            neg_edges[('gene', 'co-exp', 'gene')] = (
            uncoexp_edges[0][uncoexp_sample_index], uncoexp_edges[1][uncoexp_sample_index])

        neg_graph = dgl.heterograph(neg_edges, num_nodes_dict={'cell': n_cells, 'gene': n_genes}).to(device)
        # Add cell/gene size factor to negative graph
        if decoder == 'ZINB':
            neg_graph.nodes['cell'].data['cs_factor'] = exp_dec_graph_.nodes['cell'].data['cs_factor'].to(device)
            neg_graph.nodes['gene'].data['gs_factor'] = exp_dec_graph_.nodes['gene'].data['gs_factor'].to(device)

        # Sample expressed, construct positive graph
        if sample_rate != 1:
            pos_edges = {}

            exp_sample_index = np.random.choice(all_exp_index, n_pos_edges)
            pos_value = exp_value[exp_sample_index]
            exp_dec_edges = exp_dec_graph_[('cell', 'exp', 'gene')].edges()
            pos_edges[('cell', 'exp', 'gene')] = (
            exp_dec_edges[0][exp_sample_index], exp_dec_edges[1][exp_sample_index])
            if gene_similarity: pos_edges[('gene', 'co-exp', 'gene')] = coexp_edges

            pos_graph = dgl.heterograph(pos_edges, num_nodes_dict={'cell': n_cells, 'gene': n_genes}).to(device)

            # Add cell/gene size factor to positive graph
            if decoder == 'ZINB':
                pos_graph.nodes['cell'].data['cs_factor'] = exp_dec_graph_.nodes['cell'].data['cs_factor'].to(device)
                pos_graph.nodes['gene'].data['gs_factor'] = exp_dec_graph_.nodes['gene'].data['gs_factor'].to(device)

        # Feed forward
        pos_pre, neg_pre = model(enc_graph, pos_graph, neg_graph)

        # Calculate loss for regularization
        reg_loss = (1 / 2) * (model.cell_feature.norm(2).pow(2) +
                              model.gene_feature.norm(2).pow(2)) / float(n_cells + n_genes)

        # Calculate loss for (un)expression
        if decoder in ['Dot', 'GMF']:
            loss_exp = criterion(pos_pre, pos_value)
            loss_unexp = criterion(neg_pre, torch.zeros_like(neg_pre))

        else:
            loss_exp = criterion(pos_pre[0], pos_pre[1], pos_pre[2], pos_value)
            loss_unexp = criterion(neg_pre[0], neg_pre[1], neg_pre[2])

            ridge = torch.square(pos_pre[2]).mean() + torch.square(neg_pre[2]).mean()
            reg_loss = reg_loss + 1e-3 * ridge

        loss = loss_exp + gamma * loss_unexp + 0.0001 * reg_loss

        # Calculate loss for (un)co-expressed gene
        if gene_similarity:
            loss_gene = gene_cls(pos_pre[3], torch.ones_like(pos_pre[3]))
            loss_ungene = gene_cls(neg_pre[3], torch.zeros_like(neg_pre[3]))

            loss = loss + loss_gene + loss_ungene

        if verbose:
            count_loss_exp += loss_exp.item()
            count_loss_unexp += loss_unexp.item()
            if gene_similarity:
                count_loss_gene += loss_gene.item()
                count_loss_ungene += loss_ungene.item()

        all_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (iter_idx + 1) % log_interval == 0:
            print(
                "[{}/{}-iter] | [train] exp loss : {:.4f}, unexp loss : {:.4f}"
                .format(iter_idx + 1, iteration, count_loss_exp / log_interval, count_loss_unexp / log_interval) +
                (", gene loss : {:.4f}, ungene loss : {:.4f}".
                 format(count_loss_gene / log_interval, count_loss_unexp / log_interval) if gene_similarity else "")
            )

            count_loss_exp, count_loss_unexp, count_loss_gene, count_loss_ungene = 0, 0, 0, 0

        if verbose and cell_type is not None and (iter_idx + 1) % (log_interval * 5) == 0:
            model.eval()
            with torch.no_grad():
                c_feat, g_feat, c_last, g_last = model.encode(enc_graph)
            model.train()

            # Cell embeddings
            adata.obsm['e0'] = model.cell_feature.data.cpu().numpy()  # Return initial cell embedding
            adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
            adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings

            # kmeans
            adata = kmeans(adata, n_clusters, use_rep=use_rep)
            y_pred_k = np.array(adata.obs['kmeans'])

            # louvain
            adata = louvain(adata, resolution=resolution, use_rep=use_rep)
            y_pred_l = np.array(adata.obs['louvain'])
            print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))

            nmi_k, ari_k = calculate_metric(cell_type, y_pred_k)
            print('Clustering Kmeans %d: NMI= %.4f, ARI= %.4f' % (iter_idx + 1, nmi_k, ari_k))

            nmi_l, ari_l = calculate_metric(cell_type, y_pred_l)
            print('Clustering Louvain %d: NMI= %.4f, ARI= %.4f' % (iter_idx + 1, nmi_l, ari_l))

            all_ari_k.append(ari_k)
            all_ari_l.append(ari_l)
            all_nmi_k.append(nmi_k)
            all_nmi_l.append(nmi_l)

            if ari_k > best_ari_k:
                best_ari_k = ari_k
                best_nmi_k = nmi_k
                best_iter_k = iter_idx + 1

            if ari_l > best_ari_l:
                best_ari_l = ari_l
                best_nmi_l = nmi_l
                best_iter_l = iter_idx + 1

    ## End of training
    model.eval()
    with torch.no_grad():
        c_feat, g_feat, c_last, g_last = model.encode(enc_graph)

    adata.obsm['e0'] = model.cell_feature.data.cpu().numpy()  # Return initial cell embedding
    adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
    adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings
    adata.varm['feat'] = g_last.cpu().numpy()  # Return the final layer's gene embeddings

    if verbose and cell_type is not None:
        print(
            f'[END] For Kmeans, Best Iter : {best_iter_k} Best ARI : {best_ari_k:.4f}, Best NMI : {best_nmi_k:.4f}')
        print(
            f'[END] For Louvain, Best Iter : {best_iter_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')

    record = None
    if return_all and cell_type is not None:
        record = {
            'all_loss': all_loss,
            'ari_k': all_ari_k,
            'ari_l': all_ari_l,
            'nmi_k': all_nmi_k,
            'nmi_l': all_nmi_l
        }

    #######################   Impute expression matrix (Optional) ########################
    if impute:
        all_exp_cell, all_exp_gene = np.meshgrid(np.arange(n_cells), np.arange(n_genes))
        all_exp_cell, all_exp_gene = all_exp_cell.reshape(-1), all_exp_gene.reshape(-1)

        all_dec_graph = dgl.heterograph({('cell', 'exp', 'gene'): (all_exp_cell, all_exp_gene)},
                                        num_nodes_dict={'cell': n_cells, 'gene': n_genes}).to(device)
        all_dec_graph.nodes['cell'].data['cs_factor'] = exp_dec_graph_.nodes['cell'].data['cs_factor'].to(device)
        all_dec_graph.nodes['gene'].data['gs_factor'] = exp_dec_graph_.nodes['gene'].data['gs_factor'].to(device)

        model.eval()
        with torch.no_grad():
            all_exp = model(enc_graph, all_dec_graph)

        if decoder == 'ZINB':
            all_exp = all_exp[0]

        all_exp = all_exp.data.cpu().numpy().reshape(n_genes, n_cells).T

        adata.obsm['imputed'] = all_exp
    del model
    # del all_exp
    gc.collect()
    torch.cuda.empty_cache()

    if return_all: return adata, record

    return adata
