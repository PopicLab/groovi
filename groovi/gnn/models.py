# high level models
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl 
from dgl.nn import SumPooling
from groovi.gnn.modules import RelGraphConvLayer

class RGraphConv(nn.Module):
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
        self.conv = RGraphConv(n_in_dim=n_in_dim, m_in_dim=m_in_dim, h_dim=h_dim, num_rels=num_rels, num_hidden_layers = num_hidden_layers, update=update, normalize=normalize)
        self.sumpool = SumPooling()
        
    def forward(self, g, nfeat, mfeat, etypes):
        h = self.conv(g=g, nfeat=nfeat, mfeat=mfeat, etypes=etypes)
        h_pool = self.sumpool(g, h)
        return h_pool
    
class EdgeClassificationModel(nn.Module):
    def __init__(self, n_in_dim, m_in_dim, h_dim, num_classes, num_rels, update, num_hidden_layers=6, normalize=False):
        super().__init__()
        self.conv = RGraphConv(n_in_dim=n_in_dim, m_in_dim=m_in_dim, h_dim=h_dim, num_rels=num_rels, num_hidden_layers = num_hidden_layers, update=update, normalize=normalize)
        self.linear = nn.Linear(h_dim * 2, num_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.linear(torch.cat([h_u, h_v], dim=1))
        return {'score': score}

    def forward(self, g, nfeat, mfeat, etypes):
        h = self.conv(g=g, nfeat=nfeat, mfeat=mfeat, etypes=etypes)
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

if __name__ == "__main__":
    # Example usage
    n_in_dim = 5
    m_in_dim = 5
    h_dim = 20
    num_classes = 2
    num_rels = 4
    num_hidden_layers = 3
    normalize = True

    edge_model = EdgeClassificationModel(n_in_dim, m_in_dim, h_dim, num_classes, num_rels, update="WConcat", num_hidden_layers=num_hidden_layers, normalize=normalize)
    motif_model = MotifMiningModel(n_in_dim, m_in_dim, h_dim, num_rels, update="WConcat", num_hidden_layers=num_hidden_layers, normalize=normalize)
    torch.manual_seed(0)
    g = dgl.graph(([0, 1, 1, 2, 2], [1, 0, 2, 3, 0]))
    g.ndata["h"] = torch.randn(4, n_in_dim)
    g.edata["efeat"] = torch.randn(5, m_in_dim)
    g.edata["etype"] = torch.tensor([0, 1, 1, 2, 2])
    g_out = edge_model(g, g.ndata["h"], g.edata["efeat"], g.edata["etype"])
    e_out = motif_model(g, g.ndata["h"], g.edata["efeat"], g.edata["etype"])
    print(f'{edge_model=}, {motif_model=}')
    print(f'{g_out.shape=}, {e_out.shape=}')