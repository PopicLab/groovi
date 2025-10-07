from groovi.gnn.modules import RelGraphConvLayer, RelationalGraphConv, MLP, GlobalAttentionPooling
from groovi.seq.constants import *

import dgl
from dgl.nn import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from dgl.nn import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
import torch
import torch.nn as nn

class GraphClassificationModel(nn.Module):
    def __init__(self, n_in_dim, h_dim, num_classes, ht_emb_dim, num_rels, update="WConcat", pooling='attention', num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, h_dim=h_dim, ht_emb_dim=ht_emb_dim, num_rels=num_rels, num_hidden_layers=num_hidden_layers, update=update, normalize=normalize)
        if pooling == 'sum':
            self.pool = SumPooling()
        elif pooling == 'avg':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()
        elif pooling == 'attention':
            self.pool = GlobalAttentionPooling(nn.Linear(h_dim, 1))
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        
        self.mlp = MLP(input_dim=h_dim, hidden_dim=h_dim, output_dim=num_classes, num_layers=3, dropout=0.1)

    def forward(self, g):
        h = self.conv(g=g, nfeat=g.ndata['feat'], htfeat=g.ndata['HT'], mfeat=g.edata['feat'], etypes=g.edata['rel_type'])
        h_pool = self.pool(g, h)
        out = self.mlp(h_pool)
        return out
    
class EdgeClassificationModel(nn.Module):
    def __init__(self, n_in_dim, h_dim, ht_emb_dim, num_classes, num_rels, update, num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RelationalGraphConv(n_in_dim=n_in_dim, h_dim=h_dim, ht_emb_dim=ht_emb_dim, num_rels=num_rels, num_hidden_layers=num_hidden_layers, update=update, normalize=normalize)
        self.mlp = MLP(input_dim=h_dim*2, hidden_dim=h_dim, output_dim=num_classes, num_layers=3, dropout=0.1)

    def forward(self, g, src, dst):
        h = self.conv(g=g, nfeat=g.ndata['feat'], htfeat=g.ndata['HT'], mfeat=g.edata['feat'], etypes=g.edata['rel_type'])
        score = self.mlp(torch.cat([h[src], h[dst]], dim=-1))
        return score
    

    