import dgl
import numpy as np
import torch

class GrooviGraph:
    '''
    A Groovi Graph must have the following fields in dgl graph g.
    Node features:
        - 'feat': Tensor of shape (num_nodes, num_node_features)
        - 'position': Tensor of shape (num_nodes,) indicating the genomic position of each node
        - 'chr': Tensor of shape (num_nodes,) indicating the chromosome ID of each node
    Edge features:
        - 'feat': Tensor of shape (num_edges, num_edge_features)
        - 'rel_type': Tensor of shape (num_edges,) indicating the type of each edge
        - 'distance': Tensor of shape (num_edges,) indicating the genomic distance between connected nodes
    
    Additional attributes:
        - candidate_src: Tensor of shape (num_candidate_edges,) indicating source nodes of candidate edges
        - candidate_dst: Tensor of shape (num_candidate_edges,) indicating destination nodes of candidate edges
        - vcf_path: str, path to the VCF file associated with this graph
        - chr_ids_included: np.ndarray of uniquechromosome IDs included in the graph
    '''
    def __init__(self, graph: dgl.DGLGraph):
        required_node_keys = ['feat', 'position', 'chr']
        required_edge_keys = ['feat', 'rel_type', 'distance']
        assert all(key in graph.ndata for key in required_node_keys), f"Graph nodes must contain keys: {required_node_keys}"
        assert all(key in graph.edata for key in required_edge_keys), f"Graph edges must contain keys: {required_edge_keys}"

        self.graph = graph
        self.candidate_src: torch.LongTensor = None
        self.candidate_dst: torch.LongTensor = None
        # labeling params
        self.vcf_path: str = None
        # graph level annotations
        self.chr_ids_included: np.ndarray = None # X,Y chromosomes are encoded as 22, 23 respectively
    
    def __repr__(self):
        return f"GrooviGraph(num_nodes={self.graph.num_nodes()}, num_edges={self.graph.num_edges()})"
    
    @staticmethod
    def from_dgl_graph(graph: dgl.DGLGraph):
        """Create a GrooviGraph from a DGL graph."""
        return GrooviGraph(graph)
    
    def get_dgl_graph(self) -> dgl.DGLGraph:
        return self.graph
    
    def get_vcf_path(self):
        return self.vcf_path

    def set_vcf_path(self, vcf_path: str):
        self.vcf_path = vcf_path

    def get_candidate_edge_ids(self):
        assert self.candidate_src is not None and self.candidate_dst is not None, "Candidate edges are not set."
        return self.candidate_src, self.candidate_dst
    
    def set_candidate_edges(self, src: torch.LongTensor, dst: torch.LongTensor):
        assert src.shape == dst.shape, "Source and destination tensors must have the same shape."
        assert src.dtype == torch.int64 and dst.dtype == torch.int64, "Source and destination tensors must be of type torch.int64."
        self.candidate_src = src
        self.candidate_dst = dst
    
    def get_chr_ids_included(self, as_string: bool = False) -> np.ndarray:
        if as_string:
            return np.array2string(self.chr_ids_included, separator=',')
        return self.chr_ids_included
    
    def set_chr_ids_included(self, chr_ids: np.ndarray):
        assert isinstance(chr_ids, np.ndarray), "Chromosome IDs must be a numpy array."
        self.chr_ids_included = chr_ids
    
    def num_nodes(self):
        return self.graph.num_nodes()
    
    def num_edges(self):
        return self.graph.num_edges()

    def get_subgraph(self, node_ids: list[int]) -> 'GrooviGraph':
        """Get a subgraph from the BreakpointGraph."""
        dgl_subgraph = dgl.node_subgraph(self.graph, node_ids)
        groovi_subgraph = GrooviGraph.from_dgl_graph(dgl_subgraph)
        if self.candidate_src is not None:
            subgraph_src = self.candidate_src[torch.isin(self.candidate_src, node_ids)]
            subgraph_dst = self.candidate_dst[torch.isin(self.candidate_dst, node_ids)]
            remapped_src = self._remap_original_to_subgraph_ids(dgl_subgraph, subgraph_src)
            remapped_dst = self._remap_original_to_subgraph_ids(dgl_subgraph, subgraph_dst)
            groovi_subgraph.set_candidate_edges(remapped_src, remapped_dst)

        if self.vcf_path is not None:
            groovi_subgraph.set_vcf_path(self.vcf_path)
        
        if self.chr_ids_included is not None:
            unique_chr_ids = np.unique(self.graph.ndata['chr'][node_ids])
            groovi_subgraph.set_chr_ids_included(unique_chr_ids)

        return groovi_subgraph
    
    # Internal Helper Functions
    
    def _remap_original_to_subgraph_ids(
        self,
        subgraph: dgl.DGLGraph,
        original_node_ids_to_remap: torch.Tensor
    ) -> torch.Tensor:
        if dgl.NID not in subgraph.ndata:
            raise ValueError(
                "Subgraph must contain original node IDs (dgl.NID) in its ndata. "
                "Ensure it was created with `relabel_nodes=True`"
            )

        new_to_original_map = subgraph.ndata[dgl.NID]

        max_original_id_in_subgraph = new_to_original_map.max().item()
        original_to_subgraph_id_map = torch.full(
            (max_original_id_in_subgraph + 1,), 
            -1,                                  
            dtype=new_to_original_map.dtype,     
        )

        original_to_subgraph_id_map[new_to_original_map] = torch.arange(
            len(new_to_original_map),
            dtype=new_to_original_map.dtype,
            device=subgraph.device
        )

        remapped_ids = original_to_subgraph_id_map[original_node_ids_to_remap]

        assert not (remapped_ids == -1).any(), "Some original node IDs were not found in the subgraph."

        return remapped_ids