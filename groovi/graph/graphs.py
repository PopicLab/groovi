from groovi.cli.config import Config
from groovi.seq.constants import *
from groovi.seq.index import AdjIndex
from groovi.seq.signal_extraction import *

from collections import deque
import dgl
import dgl.transforms as T
import numpy as np
import torch
from typing import Any

class GrooviGraph:
    '''
    A Groovi Graph must have the following fields in dgl graph g.
    Node features (g.ndata):
        - 'feat': Tensor of shape (num_nodes, num_node_features)
        - 'position': Tensor of shape (num_nodes,) indicating the genomic position of each node
        - 'chr': Tensor of shape (num_nodes,) indicating the chromosome ID of each node
        - 'HT': Tensor of shape (num_nodes,) indicating HEAD/TAIL (0/1) of each node
    Edge features (g.edata):
        - 'feat': Tensor of shape (num_edges, num_edge_features)
        - 'rel_type': Tensor of shape (num_edges,) indicating the type of each edge
        - 'distance': Tensor of shape (num_edges,) indicating the genomic distance between connected nodes
    
    Additional attributes:
        - candidate_src: Tensor of shape (num_candidate_edges,) indicating source nodes of candidate edges
        - candidate_dst: Tensor of shape (num_candidate_edges,) indicating destination nodes of candidate edges
        - vcf_path: str, path to the VCF file associated with this graph
        - chr_ids_included: np.ndarray of unique chromosome IDs included in the graph
    '''
    def __init__(self, graph: dgl.DGLGraph):
        required_node_keys = ['feat', 'position', 'chr', 'HT']
        required_edge_keys = ['feat', 'rel_type', 'distance']
        assert all(key in graph.ndata for key in required_node_keys), f"Graph nodes must contain keys: {required_node_keys}"
        assert all(key in graph.edata for key in required_edge_keys), f"Graph edges must contain keys: {required_edge_keys}"

        self.graph: dgl.DGLGraph = graph
        self.ndata = graph.ndata
        self.edata = graph.edata
        self.candidate_src: torch.LongTensor = None
        self.candidate_dst: torch.LongTensor = None
        # labeling params
        self.vcf_path: str = None

        # graph level annotations
        self.chr_ids_included: np.ndarray = None

        # additional mappings
        self.chr_int_to_str: dict[int, str] = None  # mapping from chromosome integer IDs (relative to reference used at construction) to string names
    
    def __repr__(self):
        return f"GrooviGraph(num_nodes={self.graph.num_nodes()}, num_edges={self.graph.num_edges()})"
    
    @staticmethod
    def get_node_types(adj: Edge) -> tuple[NodeType, NodeType]:
        if adj.orientation == Laterality.LL:
            return NodeType.TAIL, NodeType.TAIL
        elif adj.orientation == Laterality.RR:
            return NodeType.HEAD, NodeType.HEAD
        elif adj.orientation == Laterality.LR:
            return NodeType.TAIL, NodeType.HEAD
        elif adj.orientation == Laterality.RL:
            return NodeType.HEAD, NodeType.TAIL
        else:
            raise ValueError(f"Unknown orientation: {adj.orientation}")
        
    @staticmethod
    def from_config(config: Config) -> list['GrooviGraph']:
        signal_extractor = SignalExtractor(config)
        adj_index = AdjIndex(config, edges = signal_extractor.get_edges())
        adj_index.cluster_splitreads()

        # begin graph construction
        seg_rel_id = EDGE_FEATURE_TO_ID[EdgeFeature.SEG]
        ref_rel_id = EDGE_FEATURE_TO_ID[EdgeFeature.REF]
        segments = adj_index.get_segments()
        g = dgl.graph(([], []))
        
        # create nodes
        node_iter = 0
        node_id_map = {}
        for segment in segments:
            chrom, start_pos, end_pos = segment.get_chrom(), segment.get_start_pos(), segment.get_end_pos()
            g.add_nodes(2, data={
                'position': torch.tensor([start_pos, end_pos], dtype=torch.int64),
                'chr': torch.tensor([chrom, chrom], dtype=torch.int64),
                'HT': torch.tensor([0, 1], dtype=torch.int64)  # 0 for HEAD, 1 for TAIL
            })
            node_id_map[(chrom, start_pos, NodeType.HEAD)] = node_iter
            node_id_map[(chrom, end_pos, NodeType.TAIL)] = node_iter + 1
            node_iter += 2

        # add segment edges
        for segment in segments:
            chrom, start_pos, end_pos = segment.get_chrom(), segment.get_start_pos(), segment.get_end_pos()
            head_id = node_id_map[(chrom, start_pos, NodeType.HEAD)]
            tail_id = node_id_map[(chrom, end_pos, NodeType.TAIL)]
            g.add_edges(head_id, tail_id, data={
                'rel_type': torch.tensor([seg_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[1]], dtype=torch.float32),
            })
            g.add_edges(tail_id, head_id, data={
                'rel_type': torch.tensor([seg_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[1]], dtype=torch.float32),
            })

        # add reference edges
        for i in range(len(segments)-1):
            seg1, seg2 = segments[i], segments[i+1]
            if seg1.get_chrom() != seg2.get_chrom():
                continue
            chrom = seg1.get_chrom()
            end_pos, start_pos = seg1.get_end_pos(), seg2.get_start_pos()
            tail_id = node_id_map[(chrom, end_pos, NodeType.TAIL)]
            head_id = node_id_map[(chrom, start_pos, NodeType.HEAD)]
            g.add_edges(tail_id, head_id, data={
                'rel_type': torch.tensor([ref_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[1]], dtype=torch.float32),
            })
            g.add_edges(head_id, tail_id, data={
                'rel_type': torch.tensor([ref_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[1]], dtype=torch.float32),
            })

        # TODO: add node features
        g.ndata['feat'] = torch.randn((g.num_nodes(), 5))

        # create adjacency edges
        candidate_src, candidate_dst = [], []
        for adj in adj_index.get_adjacencies():
            node_type_src, node_type_dst = GrooviGraph.get_node_types(adj)
            src_node_id = node_id_map[(adj.chrom1, adj.pos1, node_type_src)]
            dst_node_id = node_id_map[(adj.chrom2, adj.pos2, node_type_dst)]
            edge_rel_id = EDGE_FEATURE_TO_ID[adj.get_edge_type()]
            g.add_edges(src_node_id, dst_node_id, data={
                'rel_type': torch.tensor([edge_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[adj.evidence]], dtype=torch.float32),
            })
            g.add_edges(dst_node_id, src_node_id, data={
                'rel_type': torch.tensor([edge_rel_id], dtype=torch.int64),
                'feat': torch.tensor([[adj.evidence]], dtype=torch.float32),
            })
            candidate_src.append(src_node_id)
            candidate_dst.append(dst_node_id)
        candidate_src, candidate_dst = torch.tensor(candidate_src, dtype=torch.int64), torch.tensor(candidate_dst, dtype=torch.int64)

        # add pairwise depth
        u, v = g.edges()
        edge_mask = g.edata['rel_type'] == ref_rel_id
        u, v = u[edge_mask], v[edge_mask]
        m = len(u)
        rd_index = NODE_FEATURE_TO_ID[NodeFeature.RD]
        pd_edge_id = EDGE_FEATURE_TO_ID[EdgeFeature.PD]
        pairwise_depth = (g.ndata['feat'][v,rd_index] - g.ndata['feat'][u,rd_index]).unsqueeze(-1)
        rel_type = pd_edge_id*torch.ones((m,), dtype=torch.int64)
        g.add_edges(u, v, data={'feat' : pairwise_depth, 'rel_type': rel_type})

        # add distance feature
        u, v, _ = g.edges(form='all')
        distances = torch.abs(g.ndata['position'][u] - g.ndata['position'][v])
        g.edata['distance'] = distances

        # object construction and annotations
        groovi_graph = GrooviGraph(g)
        groovi_graph._set_chr_ids_included(np.unique(g.ndata['chr'].numpy())) #  record chromosomes included in connected components
        groovi_graph._set_candidate_edges(candidate_src, candidate_dst)
        groovi_graph._set_vcf_path(config.vcf if hasattr(config, 'vcf') else None)
        groovi_graph._set_chr_int_to_str_conversion(signal_extractor.fai_index.get_tid_to_str_conversion())

        return [groovi_graph]

    @staticmethod
    def _split_by_components(graph: 'GrooviGraph') -> list['GrooviGraph']:
        g = graph.get_dgl_graph()
        components: list[GrooviGraph] = []
        n = g.num_nodes()
        visited = torch.zeros(n, dtype=torch.bool)

        for node_id in range(n):
            if visited[node_id]:
                continue
            
            # BFS to find all nodes in the component
            queue = deque([node_id])
            component_node_ids = []
            while queue:
                current_node = queue.popleft()
                if visited[current_node]:
                    continue
                visited[current_node] = True
                component_node_ids.append(current_node)
                
                neighbors = g.successors(current_node).tolist()
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)

            # Create a subgraph for the component
            component_node_ids = torch.tensor(sorted(component_node_ids), dtype=torch.int64)
            breakpoint_subgraph: GrooviGraph = graph.get_subgraph(component_node_ids)
            components.append(breakpoint_subgraph)

        return components
    
    def get_dgl_graph(self) -> dgl.DGLGraph:
        return self.graph
    
    def get_vcf_path(self):
        return self.vcf_path

    def _set_vcf_path(self, vcf_path: str):
        self.vcf_path = vcf_path

    def _set_chr_int_to_str_conversion(self, chr_int_to_str: dict[int, str]):
        self.chr_int_to_str = chr_int_to_str

    def get_candidate_edge_ids(self):
        assert self.candidate_src is not None and self.candidate_dst is not None, "Candidate edges are not set."
        return self.candidate_src, self.candidate_dst
    
    def _set_candidate_edges(self, src: torch.LongTensor, dst: torch.LongTensor):
        assert src.shape == dst.shape, "Source and destination tensors must have the same shape."
        assert src.dtype == torch.int64 and dst.dtype == torch.int64, "Source and destination tensors must be of type torch.int64."
        self.candidate_src = src
        self.candidate_dst = dst
    
    def get_chr_ids_included(self, as_string: bool = False) -> np.ndarray:
        if as_string:
            return np.array([self.chr_int_to_str[chr_id] for chr_id in self.chr_ids_included])
        return self.chr_ids_included
    
    def _set_chr_ids_included(self, chr_ids: np.ndarray):
        assert isinstance(chr_ids, np.ndarray), "Chromosome IDs must be a numpy array."
        self.chr_ids_included = chr_ids
    
    def nodes(self):
        return self.graph.nodes()

    def edges(self, form: str = 'all'):
        return self.graph.edges(form=form)

    def num_nodes(self):
        return self.graph.num_nodes()
    
    def num_edges(self):
        return self.graph.num_edges()
    
    def out_edges(self, u: Any, form: str = 'all'):
        return self.graph.out_edges(u, form=form)

    def get_chr_names_by_node_ids(self, node_ids: list[int]) -> list[str]:
        chr_ids = self.graph.ndata['chr'][node_ids].numpy()
        chr_names = [self.chr_int_to_str[chr_id] for chr_id in chr_ids]
        return chr_names

    def get_subgraph(self, node_ids: list[int]) -> 'GrooviGraph':
        """Get a subgraph from the GrooviGraph."""
        dgl_subgraph = dgl.node_subgraph(self.graph, node_ids)
        groovi_subgraph = GrooviGraph(dgl_subgraph)
        if self.candidate_src is not None:
            subgraph_src = self.candidate_src[torch.isin(self.candidate_src, node_ids)]
            subgraph_dst = self.candidate_dst[torch.isin(self.candidate_dst, node_ids)]
            remapped_src = self._remap_original_to_subgraph_ids(dgl_subgraph, subgraph_src)
            remapped_dst = self._remap_original_to_subgraph_ids(dgl_subgraph, subgraph_dst)
            groovi_subgraph._set_candidate_edges(remapped_src, remapped_dst)

        if self.vcf_path is not None:
            groovi_subgraph._set_vcf_path(self.vcf_path)
        
        if self.chr_ids_included is not None:
            unique_chr_ids = np.unique(self.graph.ndata['chr'][node_ids])
            groovi_subgraph._set_chr_ids_included(unique_chr_ids)

        groovi_subgraph._set_chr_int_to_str_conversion(self.chr_int_to_str)
        return groovi_subgraph
    
    def apply_transforms(self, transform: T.BaseTransform) -> 'GrooviGraph':
        """Apply dgl transforms to the GrooviGraph in-place, returns self for chaining."""
        transformed_graph = transform(self.graph)
        self.graph = transformed_graph
        self.ndata = transformed_graph.ndata
        self.edata = transformed_graph.edata
        return self
    
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
    
    