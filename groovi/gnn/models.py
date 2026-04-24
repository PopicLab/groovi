from groovi.gnn.modules import (
    RelGraphConvLayer,
    MLP,
    GlobalAttentionPooling,
)
from groovi.graph.constants import *

import dgl
from dgl.nn import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import RelGraphConv


class EdgeRepresentation(str, Enum):
    BASE = "BASE"
    SUPPORT = "SUPPORT"
    SUPPORT_TYPE = "SUPPORT_TYPE"
    SYMMETRIC = "SYMMETRIC"
    SYMMETRIC_WITH_SUPPORT = "SYMMETRIC_WITH_SUPPORT"
    NODEOPS = "NODEOPS"

class NodeAugmentation(str, Enum):
    RATIO = "RATIO"       # src / dst per edge type (mean over all neighbors)
    DIFF  = "DIFF"        # src - dst per edge type (mean over all neighbors)


class NodeFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.count_proj = None
        self.binary_proj = None
        self.binary_idx = [config.node_feat2id[f] for f in config.node_feature_set if f in BINARY_NODE_FEATURE_TYPES]
        self.count_idx = [config.node_feat2id[f] for f in config.node_feature_set if f not in BINARY_NODE_FEATURE_TYPES]
        if config.node_feature_proj_dim is not None:
            count_out = config.node_feature_proj_dim // 2 if self.binary_idx else config.node_feature_proj_dim
            self.count_proj = nn.Sequential(nn.Linear(len(self.count_idx), count_out),
                                            nn.LayerNorm(count_out),
                                            nn.PReLU())
        if self.binary_idx:
            self.binary_proj = nn.Sequential(nn.Linear(len(self.binary_idx), config.node_feature_proj_dim // 2),
                                             nn.LayerNorm(config.node_feature_proj_dim // 2),
                                             nn.PReLU())


    def forward(self, x):
        if self.count_proj is None: return x
        if not self.binary_idx: return self.count_proj(x[:, self.count_idx])
        return torch.cat([
            self.count_proj(x[:, self.count_idx]),
            self.binary_proj(x[:, self.binary_idx]),
        ], dim=-1)

class EdgeFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(1, config.edge_feature_proj_dim) if config.edge_feature_proj_dim is not None else None

    def forward(self, e):
        return self.proj(e) if self.proj is not None else e

class RGCNCustomLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_edge_types, dropout=0.0, layer_norm=False, custom_update=True):
        super().__init__()
        self.custom_update = custom_update
        self.linear = TypedLinear(in_dim, out_dim, n_edge_types)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        if self.custom_update:
            self.update = nn.Linear(out_dim + in_dim, out_dim)
        else:
            self.update = nn.Linear(in_dim, out_dim)

    def message(self, edges):
        m = self.linear(torch.cat([edges.src["h"], edges.data["efeat"]], dim=-1), edges.data[FEAT_TYPE])
        #m = m * edges.data["efeat"] # standard rgcn scaling
        return {"m": m}

    def forward(self, g, nfeat, etypes, efeat):
        with g.local_scope():
            g.ndata["h"] = nfeat
            g.edata["efeat"] = efeat
            g.edata[FEAT_TYPE] = etypes
            g.update_all(self.message, fn.sum("m", "h_out"))
            h = g.ndata["h_out"] + self.bias
            if self.norm: h = self.norm(h)
            if self.custom_update:
                h = self.update(torch.cat([h, nfeat], dim=-1))
            else:
                h = h + self.update(nfeat)
            return self.dropout(h)

class RGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_dim = config.n_node_features if config.node_feature_proj_dim is None else config.node_feature_proj_dim
        in_dim = in_dim * (1 + config.n_edge_types) if config.augment_nodes_rgcn else in_dim
        dims = [in_dim] + config.rgcn_hidden_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(len(dims) - 1):
            if config.rgcn_layer == "Custom":
                edge_dim = config.edge_feature_proj_dim if config.edge_feature_proj_dim is not None else 1
                self.layers.append(RGCNCustomLayer(dims[i] + edge_dim, dims[i + 1], config.n_edge_types,
                                                   dropout=config.rgcn_dropout, custom_update=config.rgcn_custom_update))
            else:
                self.layers.append(RelGraphConv(dims[i], dims[i+1], config.n_edge_types, dropout=config.rgcn_dropout))
            self.norms.append(nn.LayerNorm(dims[i+1]))
            self.acts.append(nn.PReLU())

    def _augment_node_features(self, g, x):
        with g.local_scope():
            g.ndata["h"] = x
            g.apply_edges(lambda edges:
                          {"ratio": edges.src["h"] / (edges.dst["h"] + 1e-6),
                           "diff": edges.src["h"] - edges.dst["h"],
                           })
            augments = []
            for t in range(self.config.n_edge_types):
                mask = (g.edata[FEAT_TYPE] == t).float().unsqueeze(1)
                if self.config.node_augmentation == NodeAugmentation.RATIO:
                    g.edata["r"] = g.edata["ratio"] * mask
                    g.update_all(dgl.function.copy_e("r", "m"), dgl.function.mean("m", "rt"))
                    augments.append(g.ndata["rt"])
                if self.config.node_augmentation == NodeAugmentation.DIFF:
                    g.edata["d"] = g.edata["diff"] * mask
                    g.update_all(dgl.function.copy_e("d", "m"), dgl.function.mean("m", "dt"))
                    augments.append(g.ndata["dt"])
            return torch.cat([x] + augments, dim=-1)

    def forward(self, g, h_n, h_e):
        e_type = g.edata[FEAT_TYPE]
        if self.config.augment_nodes_rgcn:
            h_n = self._augment_node_features(g, h_n)
        for layer, norm, act in zip(self.layers, self.norms, self.acts):
            h_n = act(norm(layer(g, h_n, e_type, h_e)))
        return h_n

class MLPPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.edge_representation = config.edge_representation
        node_dim = config.rgcn_hidden_dim[-1]
        in_dim = {
            EdgeRepresentation.BASE: node_dim * 2,
            EdgeRepresentation.SUPPORT: node_dim * 2 + 1,
            EdgeRepresentation.SUPPORT_TYPE: node_dim * 2 + 1 + config.n_edge_types,
            EdgeRepresentation.SYMMETRIC: node_dim * 4,
            EdgeRepresentation.SYMMETRIC_WITH_SUPPORT: node_dim * 4 + 1,
            EdgeRepresentation.NODEOPS: node_dim * 2 + config.n_node_features * 2 + 1,
        }[self.edge_representation]
        dims = [in_dim] + config.mlp_hidden_dim
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.PReLU(),
                nn.Dropout(config.mlp_dropout),
            ]
        layers.append(nn.Linear(dims[-1], config.n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_src, h_dst, src_feat, dst_feat, edge_support, edge_type):
        repr = [h_src, h_dst] # BASE
        if self.edge_representation == EdgeRepresentation.SUPPORT:
            repr = [h_src, h_dst, edge_support]
        elif self.edge_representation == EdgeRepresentation.SUPPORT_TYPE:
            repr = [h_src, h_dst, edge_support, edge_type]
        elif self.edge_representation == EdgeRepresentation.SYMMETRIC:
            repr = [h_src, h_dst, h_src * h_dst, torch.abs(h_src - h_dst)]
        elif self.edge_representation == EdgeRepresentation.SYMMETRIC_WITH_SUPPORT:
            repr = [h_src, h_dst, h_src * h_dst, torch.abs(h_src - h_dst), edge_support]
        elif self.edge_representation == EdgeRepresentation.NODEOPS:
            repr = [h_src, h_dst, torch.abs(src_feat - dst_feat), src_feat / (dst_feat + 1e-6), edge_support]
        return self.mlp(torch.cat(repr, dim=-1)).unsqueeze(0)


class EdgeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_encoder = NodeFeatureEncoder(config)
        self.edge_encoder = EdgeFeatureEncoder(config)
        self.rgcn = RGCN(config)
        self.mlp = MLPPredictor(config)

    def forward(self, g, src, dst):
        h_n = self.node_encoder(g.ndata[FEAT])
        h_e = self.edge_encoder(g.edata[FEAT])
        h = self.rgcn(g, h_n, h_e)
        n, e = g.ndata[FEAT], g.edata[FEAT]
        src_flat, dst_flat = src.flatten(), dst.flatten()
        src_feat, dst_feat = n[src_flat], n[dst_flat]
        h_src, h_dst = h[src_flat], h[dst_flat]
        edge_ids = g.edge_ids(src_flat, dst_flat)
        e_type = F.one_hot(g.edata[FEAT_TYPE][edge_ids], num_classes=self.config.n_edge_types).float()
        return self.mlp(h_src, h_dst, src_feat, dst_feat, e[edge_ids], e_type)
