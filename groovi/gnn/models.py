from groovi.gnn.modules import (
    RelGraphConvLayer,
    MLP,
    GlobalAttentionPooling,
)
from groovi.seq.constants import *

import dgl
from dgl.nn import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphClassificationModel(nn.Module):
    def __init__(
        self,
        # GNN
        n_in_dim,
        ht_emb_dim,
        hidden_dim,
        num_convolutions,
        num_rels,
        # MLP
        num_hidden_layers,
        num_classes,
        # optional
        update="WConcat",
        pooling="attention",
        normalize=True,
    ):
        super().__init__()
        # setup
        self.ht_embedding = nn.Embedding(len(NodeType), ht_emb_dim)
        self.projection = nn.Linear(n_in_dim + ht_emb_dim, hidden_dim)
        self.normalize = normalize

        # GNN
        self.convolutions = nn.ModuleList([
            RelGraphConvLayer(hidden_dim, hidden_dim, num_rels, update=update) for _ in range(num_convolutions)
        ])

        if pooling == "sum":
            self.pool = SumPooling()
        elif pooling == "avg":
            self.pool = AvgPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        elif pooling == "attention":
            self.pool = GlobalAttentionPooling(nn.Linear(hidden_dim, 1))
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        self.mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_hidden_layers=num_hidden_layers,
            dropout=0.1,
        )

    def forward(self, g):
        mfeat = g.edata["feat"]
        etypes = g.edata["rel_type"]
        nfeat = g.ndata["feat"]
        ht_feat = self.ht_embedding(g.ndata['HT'])
        h = torch.cat([nfeat, ht_feat], dim=-1)
        h = self.projection(h)

        # gnn
        for conv in self.convolutions:
            h = conv(g, h, mfeat, etypes)
            h = F.leaky_relu(h)
            if self.normalize:
                h = F.normalize(h)
                
        h_pool = self.pool(g, h)
        out = self.mlp(h_pool)
        return out


class EdgeClassificationModel(nn.Module):
    def __init__(
        self,
        # GNN Params
        n_in_dim,
        ht_emb_dim,
        hidden_dim,
        num_rels,
        num_convolutions,
        update,
        normalize,
        # MLP params
        num_hidden_layers,
        num_classes,
    ):
        super().__init__()
        # setup
        self.ht_embedding = nn.Embedding(len(NodeType), ht_emb_dim)
        self.projection = nn.Linear(n_in_dim + ht_emb_dim, hidden_dim)
        self.normalize = normalize

        # GNN
        self.convolutions = nn.ModuleList([
            RelGraphConvLayer(hidden_dim, hidden_dim, num_rels, update=update) for _ in range(num_convolutions)
        ])

        # MLP, vectors are twice as long due to concatenation of two nodes
        self.mlp = MLP(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim * 2,
            output_dim=num_classes,
            num_hidden_layers=num_hidden_layers,
            dropout=0.1,
        )

    def forward(self, g, src, dst):
        # prepare input to gnn
        mfeat = g.edata["feat"]
        etypes = g.edata["rel_type"]
        nfeat = g.ndata["feat"]
        ht_feat = self.ht_embedding(g.ndata['HT'])
        h = torch.cat([nfeat, ht_feat], dim=-1)
        h = self.projection(h)

        # gnn
        for conv in self.convolutions:
            h = conv(g, h, mfeat, etypes)
            h = F.leaky_relu(h)
            if self.normalize:
                h = F.normalize(h)
        
        # class predictions
        score = self.mlp(torch.cat([h[src], h[dst]], dim=-1))
        return score
