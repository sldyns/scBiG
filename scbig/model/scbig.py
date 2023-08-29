import dgl
import torch
from dgl import function as fn
from torch import nn
from .decoder import DotDecoder, GMFDecoder, ZINBDecoder

class LightGraphConv(nn.Module):
    def __init__(self, drop_out=0.1, gene2gene=False):
        """Light Graph Convolution

        Paramters
        ---------
        drop_out : float
            dropout rate (neighborhood dropout)
        gene2gene : boolean
            whether to add gene massage to cell-gene bipartite graph
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
        self.gene2gene = gene2gene

    def forward(self, graph, feats):
        """Apply Light Graph Convoluiton to specific edge type {r}

        Paramters
        ---------
        graph : dgl.graph
        src_feats : torch.FloatTensor
            source node features

        ci : torch.LongTensor
            in-degree of sources ** (-1/2)
            shape : (n_sources, 1)
        cj : torch.LongTensor
            out-degree of destinations ** (-1/2)
            shape : (n_destinations, 1)

        Returns
        -------
        output : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
                where N_{i, r} ; number of neighbors_{i, r} ** (1/2)
        2. aggregation
            \sum_{j \in N(i), r} MP_{j -> i, r}
        """
        if isinstance(feats, tuple):
            src_feats, dst_feats = feats

        with graph.local_scope():
            if self.gene2gene:
                cj, ci = graph.srcdata['cjj'], graph.dstdata['cii']
            else:
                cj, ci = graph.srcdata['cj'], graph.dstdata['ci']

            cj_dropout = self.dropout(cj)
            weighted_feats = torch.mul(src_feats, cj_dropout)
            graph.srcdata['h'] = weighted_feats

            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))
            out = torch.mul(graph.dstdata['out'], ci)

        return out


class lightgraphconvlayer(nn.Module):
    def __init__(self, drop_out=0.1, alpha=None):
        super().__init__()
        """lightgraphconv layer

        drop_out : float
            dropout rate (feature dropout)
        alpha: float
            weight for gene massage
        """
        self.alpha = alpha
        conv = {}

        cell_to_gene_key = 'exp'
        gene_to_cell_key = 'reverse-exp'
        gene_to_gene_key = 'co-exp'

        # convolution on cell -> gene graph
        conv[cell_to_gene_key] = LightGraphConv(drop_out=drop_out)

        # convolution on gene -> cell graph
        conv[gene_to_cell_key] = LightGraphConv(drop_out=drop_out)

        # convolution on gene -> gene graph
        if self.alpha is not None:
            conv[gene_to_gene_key] = LightGraphConv(drop_out=drop_out, gene2gene=True)

        self.conv = dgl.nn.HeteroGraphConv(conv, aggregate='stack')
        self.feature_dropout = nn.Dropout(drop_out)

    def forward(self, graph, c_feat, g_feat, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.graph
        c_feat, g_feat : torch.FloatTensor
            node features
        ckey, gkey : str
            target node types

        Returns
        -------
        c_feat, g_feat : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ckey: c_feat,
            gkey: g_feat
        }

        out = self.conv(graph, feats)
        c_feat = out[ckey].squeeze()
        g_feat = self.alpha * out[gkey][:, 0] + (1 - self.alpha) * out[gkey][:, 1] if self.alpha is not None else out[
            gkey].squeeze()

        return c_feat, g_feat


class scBiG(nn.Module):
    def __init__(self,
                 n_layers,
                 n_cells,
                 n_genes,
                 drop_out,
                 feats_dim,
                 decoder='ZINB',
                 learnable_weight=False,
                 gene_similarity=False,
                 alpha=0.9):
        super().__init__()
        """scBiG: a novel scRNA-seq representation learning method based on graph node embedding

        n_layers : int
            number of GCN layers
        n_cells : int
            number of cells
        n_genes : int
            number of genes   
        drop_out : float
            dropout rate (neighbors)
        feats_dim : int
            node feature dimension
        decoder = 'ZINB'
            structure of decoder. default is 'ZINB', optionally input 'Dot', 'GMF' or 'ZINB'
        learnable_weight : boolean
            whether to learn weights for embedding aggregation, if False, use 1/n_layers
        gene_similarity : boolean
           whether to consider gene massage
        alpha: float
            weight for gene massage
        
        """
        self.gene_similarity = gene_similarity
        self.alpha = alpha if gene_similarity else None

        self.n_cells = n_cells
        self.n_genes = n_genes

        self.cell_feature = nn.Parameter(torch.Tensor(self.n_cells, feats_dim))
        self.gene_feature = nn.Parameter(torch.Tensor(self.n_genes, feats_dim))

        nn.init.xavier_uniform_(self.cell_feature)
        nn.init.xavier_uniform_(self.gene_feature)

        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(lightgraphconvlayer(drop_out=drop_out, alpha=self.alpha))

        if self.n_layers == 2:
            self.weights = torch.tensor([1., 1. / 2, 1. / 2])
        else:
            self.weights = torch.ones([self.n_layers + 1, 1]) / (self.n_layers + 1)

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)

        if decoder == 'Dot':
            self.decoder = DotDecoder()
        elif decoder == 'GMF':
            self.decoder = GMFDecoder(feats_dim=feats_dim)
        elif decoder == 'ZINB':
            self.decoder = ZINBDecoder(feats_dim=feats_dim, gene_similarity=self.gene_similarity)

        for p, q in self.decoder.named_parameters():
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)

    def encode(self, graph, ckey='cell', gkey='gene'):
        c_feat, g_feat = self.cell_feature, self.gene_feature
        c_hidden, g_hidden = self.weights[0] * c_feat, self.weights[0] * g_feat
        for w, encoder in zip(self.weights[1:], self.encoders):
            c_feat, g_feat = encoder(graph, c_feat, g_feat, ckey, gkey)
            c_hidden = c_hidden + w * c_feat
            g_hidden = g_hidden + w * g_feat

        return c_hidden, g_hidden, c_feat, g_feat

    def decode(self, pos_graph, neg_graph, c_feat, g_feat, g_last, ckey, gkey):
        pos_pre = self.decoder(pos_graph, c_feat, g_feat, g_last, ckey, gkey)
        neg_pre = self.decoder(neg_graph, c_feat, g_feat, g_last, ckey, gkey)
        return pos_pre, neg_pre

    def forward(self,
                enc_graph,
                pos_graph,
                neg_graph=None,
                ckey='cell',
                gkey='gene'):
        """
        Parameters
        ----------
        enc_graph, pos_graph, neg_graph: dgl.graph
            constructed graphs

        ckey, gkey : str
            target node types

        Returns
        -------
        pos_pre, neg_pre : torch.FloatTensor
            edge predictions of positive and negative graph in the decoder
        """

        c_feat, g_feat, c_last, g_last = self.encode(enc_graph, ckey, gkey)
        if neg_graph is not None:
            return self.decode(pos_graph, neg_graph, c_feat, g_feat, g_last, ckey, gkey)

        return self.decoder(pos_graph, c_feat, g_feat, g_last, ckey, gkey)
