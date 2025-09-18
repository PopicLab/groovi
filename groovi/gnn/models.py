from groovi.gnn.modules import RelGraphConvLayer, RelationalGraphConv
from groovi.seq.constants import *

import dgl
from dgl.nn import SumPooling, AvgPooling, MaxPooling
import torch
import torch.nn as nn

class MotifMiningModel(nn.Module):
    def __init__(self, n_in_dim, n_h_dim, ht_h_dim, num_rels, update="WConcat", pooling='sum', num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, h_dim=n_h_dim, ht_h_dim=ht_h_dim, num_rels=num_rels, num_hidden_layers=num_hidden_layers, update=update, normalize=normalize)
        if pooling == 'sum':
            self.pool = SumPooling()
        elif pooling == 'avg':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

    def forward(self, g):
        h = self.conv(g=g, nfeat=g.ndata['feat'], htfeat=g.ndata['HT'], mfeat=g.edata['feat'], etypes=g.edata['rel_type'])
        h_pool = self.pool(g, h)
        return h_pool
    
class EdgeClassificationModel(nn.Module):
    def __init__(self, n_in_dim, n_h_dim, ht_h_dim, num_classes, num_rels, update, num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, h_dim=n_h_dim, ht_h_dim=ht_h_dim, num_rels=num_rels, num_hidden_layers=num_hidden_layers, update=update, normalize=normalize)
        self.linear = nn.Linear(n_h_dim * 2, num_classes)

    def forward(self, g, src, dst):
        h = self.conv(g=g, nfeat=g.ndata['feat'], htfeat=g.ndata['HT'], mfeat=g.edata['feat'], etypes=g.edata['rel_type'])
        score = self.linear(torch.cat([h[src], h[dst]], dim=-1))
        return score