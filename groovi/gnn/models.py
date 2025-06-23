# high level models
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl 
from dgl.nn import SumPooling
from groovi.gnn.modules import RelGraphConvLayer

class RelationalGraphConv(nn.Module):
    def __init__(self, n_in_dim, m_in_dim, h_dim, num_rels, normalize, num_hidden_layers, update="WConcat"):
        super().__init__()
        self.normalize = normalize
        self.initial_layer = RelGraphConvLayer(n_in_feat=n_in_dim, m_in_feat=m_in_dim, out_feat=h_dim, num_rels=num_rels, update=update)
        layers = [RelGraphConvLayer(h_dim, m_in_dim, h_dim, num_rels, update=update) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(layers)
       
    def forward(self, g, nfeat, mfeat, etypes):
        etypes = etypes.long() # expected to be of type long; 64 bit int
        h = self.initial_layer(g, nfeat, mfeat, etypes=etypes)
        h = F.leaky_relu(h)
        if self.normalize: h = F.normalize(h, dim=1) 
        for layer in self.hidden_layers:
            h = layer(g, h, mfeat, etypes) 
            h = F.leaky_relu(h)
            if self.normalize: h = F.normalize(h, dim=1)
        return h

class MotifMiningModel(nn.Module):
    def __init__(self, n_in_dim, m_in_dim, h_dim, num_rels, update, num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, m_in_dim=m_in_dim, h_dim=h_dim, num_rels=num_rels, num_hidden_layers = num_hidden_layers, update=update, normalize=normalize)
        self.sumpool = SumPooling()

    def forward(self, g):
        nfeat = g.ndata['feat']
        mfeat = g.edata['feat']
        etypes = g.edata['rel_type']
        h = self.conv(g=g, nfeat=nfeat, mfeat=mfeat, etypes=etypes)
        h_pool = self.sumpool(g, h)
        return h_pool
    
class EdgeClassificationModel(nn.Module):
    def __init__(self, n_in_dim, m_in_dim, h_dim, num_classes, num_rels, update, num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, m_in_dim=m_in_dim, h_dim=h_dim, num_rels=num_rels, num_hidden_layers = num_hidden_layers, update=update, normalize=normalize)
        self.linear = nn.Linear(h_dim * 2, num_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.linear(torch.cat([h_u, h_v], dim=1))
        return {'score': score}

    def forward(self, g):
        nfeat = g.ndata['feat']
        mfeat = g.edata['feat']
        etypes = g.edata['rel_type']
        h = self.conv(g=g, nfeat=nfeat, mfeat=mfeat, etypes=etypes)
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']