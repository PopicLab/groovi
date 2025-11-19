from groovi.graph.constants import *
import torch
from dgl import transforms as T


class NormalizeNodes01(T.BaseTransform):
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, g):
        x = torch.clip(g.ndata[FEAT], self.vmin, self.vmax)
        if self.vmin < 0: x = 0.5 + 0.5 * x / self.vmax
        else: x = x / self.vmax
        g.ndata[FEAT] = x
        return g

class NormalizeEdges01(T.BaseTransform):
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, g):
        for e_type in NORM_EDGE_FEATURE_TYPES:
            e_type_id = EDGE_FEATURE_TO_ID[e_type]
            ids = (g.edata[FEAT_TYPE] == e_type_id).nonzero().squeeze()
            x = torch.clip(g.edata[FEAT][ids], self.vmin, self.vmax)
            if self.vmin < 0: x = 0.5 + 0.5 * x / self.vmax
            else: x = x / self.vmax
            g.edata[FEAT][ids] = x
        return g

class AddSuperNode(T.BaseTransform):
    def __init__(self):
        pass

    def __call__(self, g):
        g_new = g.clone()
        num_original_nodes = g_new.num_nodes()

        # Add a super node. Its feature dimension matches the original nodes.
        super_node_feat = torch.zeros((1, g.ndata[FEAT].size(1)), dtype=torch.float32)
        g_new.add_nodes(1, {FEAT: super_node_feat})
        super_node_id = g_new.num_nodes() - 1
        num_new_edges = num_original_nodes
        
        # Define source and destination nodes for new edges
        src = torch.full((num_new_edges,), super_node_id, dtype=torch.int64)
        dst = torch.arange(num_new_edges, dtype=torch.int64)
        efeat = torch.ones((num_new_edges, 1), dtype=torch.float32)
        rel_type = torch.full((num_new_edges,), EDGE_FEATURE_TO_ID[EdgeFeature.SUPER], dtype=torch.int64)

        g_new.add_edges(src, dst, data={FEAT: efeat, FEAT_TYPE: rel_type})
        g_new.add_edges(dst, src, data={FEAT: efeat, FEAT_TYPE: rel_type})
        
        return g_new

class NodeFeatureMask(T.BaseTransform):
    def __init__(self, features: list[NodeFeature]):
        self.features = features

    def __call__(self, g):
        _, d = g.ndata[FEAT].size()
        n_mask = torch.ones((d,), dtype=torch.float32)
        for feature in self.features:
            feature_idx = NODE_FEATURE_TO_ID[feature]
            n_mask[:, feature_idx] = 0
        g.ndata[FEAT] = g.ndata[FEAT] * n_mask
        return g


TRANSFORMS = lambda config: {
    'NORMALIZE_NODES_01': NormalizeNodes01(vmin=config.feature_norm_vmin, vmax=config.feature_norm_vmax),
    'NORMALIZE_EDGES_01': NormalizeEdges01(vmin=config.feature_norm_vmin, vmax=config.feature_norm_vmax),
    'NODE_FEAT_MASK_01': NodeFeatureMask(features=config.features_to_mask),
}