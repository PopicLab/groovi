# nn modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.linear import TypedLinear # https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.TypedLinear.

# Adapted from https://www.dgl.ai/dgl_docs/en/2.4.x/_modules/dgl/nn/pytorch/conv/relgraphconv.html#RelGraphConv
class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        n_in_feat,
        m_in_feat,
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

# Usage Example
if __name__ == "__main__":
    torch.manual_seed(0)
    g = dgl.graph(([0, 1, 1, 2, 2], [1, 0, 2, 3, 0]))
    g.ndata["h"] = torch.randn(4, 5)
    g.edata["efeat"] = torch.randn(5, 5)
    g.edata["etype"] = torch.tensor([0, 1, 1, 2, 2])
    rgcn = RelGraphConvLayer(5, 5, 5, 3, num_bases=3)
    h = rgcn(g, g.ndata["h"], g.edata["efeat"], g.edata["etype"])
    print(h)