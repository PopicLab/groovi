from groovi.graph.graphs import GrooviGraph
from groovi.graph.constants import *

import dgl
import torch
from dgl import transforms as T


'''
Feature normalization transforms
'''
class NormalizeNodes01(T.BaseTransform):
    def __init__(self, n_features_max: float, n_features_min: float = 0):
        self.n_features_max = n_features_max
        self.n_features_min = n_features_min

    def __call__(self, g):
        g_new = g.clone()
        g_new.ndata[FEAT] = (g_new.ndata[FEAT]-self.n_features_min)/(self.n_features_max-self.n_features_min)
        return g_new

class NormalizeEdges01(T.BaseTransform):
    def __init__(self, m_features_max: float, edge_types: list[EdgeFeature]):
        self.edge_types = edge_types
        self.m_features_max = m_features_max

    def __call__(self, g):
        g_new: dgl.DGLGraph = g.clone()
        for e_type in self.edge_types:
            e_type_id = EDGE_FEATURE_TO_ID[e_type]
            ids = (g_new.edata[FEAT_TYPE] == e_type_id).nonzero().squeeze()
            g_new.edata[FEAT][ids] /= self.m_features_max
        return g_new

'''
Edge (addition/removal) transforms
'''

class RemoveEdge(T.BaseTransform):
    def __init__(self, edge_feat_type):
        self.edge_feat_type = edge_feat_type
        pass

    def __call__(self, g):
        edge_feat_type_id = EDGE_FEATURE_TO_ID[self.edge_feat_type]
        assert torch.any(g.edata[FEAT_TYPE] == edge_feat_type_id), "Edge type does not exist in the graph."
        g_new: dgl.DGLGraph = g.clone()
        edge_mask = g_new.edata[FEAT_TYPE] == edge_feat_type_id
        g_new.remove_edges( g_new.edges('eid')[edge_mask])
        return g_new


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

'''
Feature Masking Transforms
'''
class NodeFeatureMask(T.BaseTransform):
    def __init__(self, features: list[NodeFeature] = []):
        self.features = features

    def __call__(self, g):
        g_new = g.clone()
        _, d = g_new.ndata[FEAT].size()
        n_mask = torch.ones((d,), dtype=torch.float32)
        for feature in self.features:
            feature_idx = NODE_FEATURE_TO_ID[feature]
            n_mask[:, feature_idx] = 0
        g_new.ndata[FEAT] = g_new.ndata[FEAT] * n_mask
        return g_new

# global export for transformations mappings
# TODO should be initialized from the config
TRANSFORMS = {
    'NORMALIZE_NODES_01': NormalizeNodes01(n_features_max=100, n_features_min=0.0),
    'NORMALIZE_EDGES_01': NormalizeEdges01(m_features_max=100, edge_types=[EdgeFeature.LR, EdgeFeature.RL,
                                                                           EdgeFeature.LL, EdgeFeature.RR,
                                                                           EdgeFeature.RD, EdgeFeature.RD_POS,
                                                                           EdgeFeature.RD_NEG]),
    'REMOVE_EDGE': RemoveEdge(EdgeFeature.RD),
    'SUPER_NODE': AddSuperNode(),
    'NODE_MASK_CLIPPINGS': NodeFeatureMask([NodeFeature.RD_CLIPPED_L, NodeFeature.RD_CLIPPED_R]),
    'NODE_MASK_INSERTIONS': NodeFeatureMask([NodeFeature.INS]),
    # ... more transformations if needed
}
