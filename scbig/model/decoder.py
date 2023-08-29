import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.act = nn.Sigmoid()

    def forward(self, graph, c_feat, g_feat, g_last=None, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last layer
        ckey, gkey : str
            target node types

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_edges, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = c_feat
            graph.nodes[gkey].data['h'] = g_feat
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pred = self.act(graph.edata['score'])

        return pred


class GMFDecoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.out = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())

    def forward(self, graph, c_feat, g_feat, g_last=None, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last layer
        ckey, gkey : str
            target node types

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_edges, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = c_feat
            graph.nodes[gkey].data['h'] = g_feat
            graph.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            pred = self.out(graph.edata['score'])

        return pred


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x) - 1., min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ZINBDecoder(nn.Module):
    def __init__(self, feats_dim, gene_similarity=False):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()
        self.gene_similarity = gene_similarity

    def forward(self, graph, c_feat, g_feat, g_last=None, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last laye
        ckey, gkey : str
            target node types

        Returns
        -------
        mu : torch.FloatTensor
            the estimated mean of ZINB model shape : (n_edges, 1)
        disp : torch.FloatTensor
            the estimated dispersion of ZINB model shape : (n_edges, 1)
        pi : torch.FloatTensor
            the estimated dropout rate of ZINB model shape : (n_edges, 1)
        ge_score : torch.FloatTensor
            the predicted values of highly correlated gene edges when considering gene massage
        """
        ge_score = None

        with graph.local_scope():
            graph.nodes[ckey].data['h'], graph.nodes[gkey].data['h'] = c_feat, g_feat
            graph.nodes[ckey].data['one'] = torch.ones([c_feat.shape[0], 1], device=c_feat.device)
            graph.nodes[gkey].data['one'] = torch.ones([g_feat.shape[0], 1], device=g_feat.device)

            exp_graph = graph['cell', 'exp', 'gene'] if self.gene_similarity else graph

            exp_graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))
            exp_graph.apply_edges(fn.u_mul_v('one', 'gs_factor', 'gs_factor'))
            exp_graph.apply_edges(fn.u_mul_v('cs_factor', 'one', 'cs_factor'))

            h_d = exp_graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            disp_ = self.dec_disp(h_d)
            pi = self.dec_pi(h_d)

            disp = self.dec_disp_act(exp_graph.edata['gs_factor'] * disp_)
            mu_ = exp_graph.edata['gs_factor'] * mu_
            mu = exp_graph.edata['cs_factor'] * self.dec_mean_act(mu_)

            if self.gene_similarity:
                co_exp_graph = graph['gene', 'co-exp', 'gene']
                co_exp_graph.nodes[gkey].data['hh'] = g_last
                co_exp_graph.apply_edges(fn.u_dot_v('hh', 'hh', 'h_d'))
                h_d = co_exp_graph.edata['h_d']

                ge_score = F.sigmoid(h_d)

        return mu, disp, pi, ge_score
