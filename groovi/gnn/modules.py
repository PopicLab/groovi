# nn modules
from groovi.seq.constants import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import sum_nodes, mean_nodes, softmax_nodes
import dgl.function as fn
from dgl.nn.pytorch.linear import TypedLinear

# Adapted from https://www.dgl.ai/dgl_docs/en/2.4.x/_modules/dgl/nn/pytorch/conv/relgraphconv.html#RelGraphConv
'''
Graph convolutions
'''
class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        n_in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        dropout=0.0,
        update="WConcat",
    ):
        super().__init__()
        # set parameters
        m_in_feat = 1  # edge feature dimension is 1 
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        in_feat = n_in_feat + m_in_feat #TODO: Change
        self.bias = bias
        self.update = update

        # Message function definitions
        self.linear_r = TypedLinear(
            in_feat, out_feat, num_rels, regularizer, num_bases
        )
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # update functions
        if self.update == "WConcat":
            self.linear = nn.Linear(out_feat + n_in_feat, out_feat)

        # misc
        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        # edges is an instance of EdgeBatch, containing src, dst, data fields
        in_feat = torch.concat([edges.src["h"], edges.data["efeat"]], dim=-1)
        m = self.linear_r(in_feat, edges.data["etype"])
        if "norm" in edges.data:
            m = m * edges.data["norm"]
        return {"m": m} # b, d_out

    def forward(self, g, nfeat, efeat, etypes, norm=None, *, presorted=False):
        self.presorted = presorted
        with g.local_scope():
            g: dgl.DGLGraph
            g.ndata["h"] = nfeat
            if norm is not None:
                g.edata["norm"] = norm
            g.edata["etype"] = etypes
            g.edata["efeat"] = efeat
            # apply message, reduce functions
            g.update_all(
                self.message,            # creates messages, writes to m
                fn.sum("m", "h_reduced") # aggregates m across edges (ie diff relationships), writes to h_reduced
            ) 
            h_reduced = g.ndata["h_reduced"]
            if self.bias: h_reduced = h_reduced + self.h_bias

            # apply update function
            if self.update == "WConcat":
                h_updated = torch.concat((h_reduced, nfeat), dim=-1)
                h_updated = self.linear(h_updated)
            # add additional update functions here
            else:
                raise ValueError(f"Unknown update function: {self.update}")
            
            h_updated = self.dropout(h_updated)
            return h_updated

class RelationalGraphConv(nn.Module):
    def __init__(self, n_in_dim, h_dim, ht_emb_dim, num_rels, normalize, num_hidden_layers, update="WConcat"):
        super().__init__()
        self.normalize = normalize
        self.ht_embedding = nn.Embedding(len(NodeType), ht_emb_dim) # 2 for head/tail
        self.initial_layer = RelGraphConvLayer(n_in_feat=n_in_dim + ht_emb_dim, out_feat=h_dim, num_rels=num_rels, update=update)
        layers = [RelGraphConvLayer(h_dim, h_dim, num_rels, update=update) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(layers)
       
    def forward(self, g, nfeat, htfeat, mfeat, etypes):
        ht_emb = self.ht_embedding(htfeat)
        nfeat = torch.cat([nfeat, ht_emb], dim=-1)
        h = self.initial_layer(g, nfeat, mfeat, etypes=etypes.long()) # etypes expected to be of type long; 64 bit int
        h = F.leaky_relu(h)
        if self.normalize: h = F.normalize(h, dim=1) 
        for layer in self.hidden_layers:
            h = layer(g, h, mfeat, etypes) 
            h = F.leaky_relu(h)
            if self.normalize: h = F.normalize(h, dim=1)
        return h
    
'''
Global Pooling modules
'''
# from https://www.dgl.ai/dgl_docs/_modules/dgl/nn/pytorch/glob.html#GlobalAttentionPooling
class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert (
                gate.shape[-1] == 1
            ), "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata["gate"] = gate
            gate = softmax_nodes(graph, "gate")
            graph.ndata.pop("gate")

            graph.ndata["r"] = feat * gate
            readout = sum_nodes(graph, "r")
            graph.ndata.pop("r")

            if get_attention:
                return readout, gate
            else:
                return readout

'''
General modules
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != num_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
