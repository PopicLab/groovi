from groovi.graph.constants import *
import dgl
import torch
from dgl import transforms as T


class FeatureNormType(str, Enum):
    LOG1P = "LOG1P"
    CLAMP = "CLAMP"
    SCALE = "SCALE"
    RATIO = "RATIO"
    NEIGHBOR_RATIO = "NEIGHBOR_RATIO"
    NONE  = "NONE"

class NormalizeFeatures(T.BaseTransform):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vmax = self.config.feature_norm_vmax

    def _clamp_to_percentile(self, x, clamp_percentile=95.0):
        vmax = torch.quantile(x, clamp_percentile / 100.0).item()
        return x.clamp(0, vmax)

    def _apply_ratio(self, x):
        x = self._clamp_to_percentile(x)
        x /= (x.max(dim=0, keepdim=True).values + 1e-6)
        return x

    def _apply_norm(self, feature_norm, x):
        if feature_norm == FeatureNormType.LOG1P: return torch.log1p(x)
        elif feature_norm == FeatureNormType.RATIO: return self._apply_ratio(x)
        elif feature_norm == FeatureNormType.CLAMP: return x.clamp(0, self.vmax)
        elif feature_norm == FeatureNormType.SCALE: return x.clamp(0, self.vmax) / self.vmax
        return x

    def _normalize_nodes(self, g):
        norm_indices = [self.config.node_feat2id[f] for f in self.config.node_feature_set if f not in BINARY_NODE_FEATURE_TYPES]
        if self.config.node_feature_norm == FeatureNormType.NEIGHBOR_RATIO:
            with g.local_scope():
                g.ndata["h"] = g.ndata[FEAT][:, norm_indices]
                g.update_all(dgl.function.copy_u("h", "m"), dgl.function.mean("m", "neigh_mean"))
                g.ndata[FEAT][:, norm_indices] = g.ndata[FEAT][:, norm_indices] / (g.ndata["neigh_mean"] + 1e-6)
        else:
            g.ndata[FEAT][:, norm_indices] = self._apply_norm(self.config.node_feature_norm, g.ndata[FEAT][:, norm_indices])

    def _normalize_edges(self, g):
        norm_types = [self.config.edge_feat2id[f] for f in self.config.edge_feature_set if f not in BINARY_EDGE_FEATURE_TYPES]
        mask = torch.isin(g.edata[FEAT_TYPE], torch.tensor(norm_types, device=g.device))
        g.edata[FEAT][mask] = self._apply_norm(self.config.edge_feature_norm, g.edata[FEAT][mask])

    def __call__(self, g):
        self._normalize_nodes(g)
        self._normalize_edges(g)
        return g

class NodeFeatureMask(T.BaseTransform):
    def __init__(self, config):
        self.config = config
        self.features = self.config.features_to_mask

    def __call__(self, g):
        _, d = g.ndata[FEAT].size()
        n_mask = torch.ones((d,), dtype=torch.float32)
        for feature in self.features:
            feature_idx = self.config.node_feat2id[feature]
            n_mask[:, feature_idx] = 0
        g.ndata[FEAT] = g.ndata[FEAT] * n_mask
        return g


TRANSFORMS = lambda config: {
    'NODE_FEAT_MASK_01': NodeFeatureMask(config),
    'NORMALIZE_FEATURES': NormalizeFeatures(config),
}