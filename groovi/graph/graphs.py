from groovi.cli.config import Config
from groovi.seq.constants import *
from groovi.seq.signal_processing import SVSignalProcessor
from groovi.graph.constants import *
from dataclasses import dataclass

from collections import deque
import dgl
import dgl.transforms as T
import numpy as np
import torch
from typing import Any


@dataclass(frozen=True)
class GrooviNode:
    """
    Node of the Groovi graph represented by the genomic locus and H/T assignment
    """
    chr_name: str
    ref_pos: int
    type: NodeType
    @classmethod
    def from_segment(cls, ref_segment, node_type):
        return cls(
            chr_name=ref_segment.chr,
            ref_pos=ref_segment.start if node_type == NodeType.HEAD else ref_segment.end,
            type=node_type
        )

class GrooviGraph:
    """
    A Groovi Graph is a feature-augmented breakpoint graph
    It is internally represented using a DGL graph with the following properties:
    Node features (graph.ndata):
        - 'HT': Tensor of shape (num_nodes,) indicating if this is a HEAD or TAIL (0/1) node
        - FEAT: Tensor of shape (num_nodes, num_node_features)
    Edge features (graph.edata):
        - FEAT_TYPE: Tensor of shape (num_edges,) indicating the type of each edge
        - FEAT: Tensor of shape (num_edges, num_edge_features) with feature values

    Additional attributes:
        - 'segment': Tensor of shape (num_nodes,) indicating the genomic segment represented by each node
        - candidate_src: Tensor of shape (num_candidate_edges,) indicating source nodes of candidate edges
        - candidate_dst: Tensor of shape (num_candidate_edges,) indicating destination nodes of candidate edges
        - vcf_path: str, path to the VCF file associated with this graph, optional
        - info_path: str, path to the info file associated with this graph, intended for graph classification tasks, optional
    """

    def __init__(self):
        self.graph: dgl.DGLGraph = dgl.graph(([], []))
        self.candidate_src: torch.LongTensor = []
        self.candidate_dst: torch.LongTensor = []
        self.node_to_id = {}
        self.id_to_node = {}
        # labeling params
        self.vcf_path: str = None
        self.info_path: str = None

    def __repr__(self):
        return f"GrooviGraph(num_nodes={self.graph.num_nodes()}, num_edges={self.graph.num_edges()})"

    def verify(self):
        required_node_keys = [FEAT, "HT"]
        required_edge_keys = [FEAT, FEAT_TYPE]
        assert all(key in self.graph.ndata for key in required_node_keys), f"Graph nodes must contain keys: {required_node_keys}"
        assert all(key in self.graph.edata for key in required_edge_keys), f"Graph edges must contain keys: {required_edge_keys}"

    def get_stats(self):
        return {'num_nodes': self.graph.num_nodes(),
                'num_edges':  self.graph.num_edges(),
                'vcf':  self.vcf_path,
                'candidate_src':  self.candidate_src,
                'candidate_dst':  self.candidate_dst}

    def add_nodes(self, ref_segments):
        node_idx = 0
        for segment in ref_segments:
            self._add_node(ref_segment=segment, node_type=NodeType.HEAD, idx=node_idx)
            self._add_node(ref_segment=segment, node_type=NodeType.TAIL, idx=node_idx + 1)
            self.graph.add_nodes(2, data={"HT": torch.tensor([0, 1], dtype=torch.int64)}) # 0 for HEAD, 1 for TAIL
            node_idx += 2

    def add_segment_edges(self, ref_segments):
        for segment in ref_segments:
            head_id = self.node_to_id[GrooviNode.from_segment(segment, NodeType.HEAD)]
            tail_id = self.node_to_id[GrooviNode.from_segment(segment, NodeType.TAIL)]
            self._add_edges(head_id, tail_id, data={
                FEAT_TYPE: torch.tensor([EDGE_FEATURE_TO_ID[EdgeFeature.SEG]], dtype=torch.int64),
                FEAT: torch.tensor([[1]], dtype=torch.float32)})

    def add_reference_edges(self, ref_segments):
        for i in range(len(ref_segments) - 1):
            seg1, seg2 = ref_segments[i], ref_segments[i + 1]
            if seg1.chr != seg2.chr: continue
            tail_id = self.node_to_id[GrooviNode.from_segment(seg1, NodeType.TAIL)]
            head_id = self.node_to_id[GrooviNode.from_segment(seg2, NodeType.HEAD)]
            self._add_edges(tail_id, head_id, data={
                FEAT_TYPE: torch.tensor([EDGE_FEATURE_TO_ID[EdgeFeature.REF]], dtype=torch.int64),
                FEAT: torch.tensor([[1]], dtype=torch.float32)})

    def add_novel_junction_edges(self, junctions):
        for j in junctions:
            node_type_src, node_type_dst = ORIENTATION_TO_NODE_TYPES[j.orientation]
            src_node_id = self.node_to_id[GrooviNode(j.chr1, j.pos1, node_type_src)]
            dst_node_id = self.node_to_id[GrooviNode(j.chr2, j.pos2, node_type_dst)]
            edge_type_feat_id = EDGE_FEATURE_TO_ID[ORIENTATION_TO_EDGE_TYPE[j.orientation]]
            self._add_edges(src_node_id, dst_node_id, data={
                FEAT_TYPE: torch.tensor([edge_type_feat_id], dtype=torch.int64),
                FEAT: torch.tensor([[j.support]], dtype=torch.float32)})
            self.candidate_src.append(src_node_id)
            self.candidate_dst.append(dst_node_id)
        self.candidate_src, self.candidate_dst = (torch.tensor(self.candidate_src, dtype=torch.int64),
                                                  torch.tensor(self.candidate_dst, dtype=torch.int64))

    def add_node_features(self):
        self.graph.ndata[FEAT] = torch.randn((self.graph.num_nodes(), 5))

    def add_edge_features(self):
        # edge feature: change in depth
        edge_mask = self.graph.edata[FEAT_TYPE] != EDGE_FEATURE_TO_ID[EdgeFeature.SEG]
        u, v = self.graph.edges()
        u, v = u[edge_mask], v[edge_mask]
        pairwise_depth = (self.graph.ndata[FEAT][v, NODE_FEATURE_TO_ID[NodeFeature.RD]] -
                          self.graph.ndata[FEAT][u, NODE_FEATURE_TO_ID[NodeFeature.RD]]).unsqueeze(-1)
        self._add_edges(u, v, data={FEAT_TYPE: torch.full((len(u),), EDGE_FEATURE_TO_ID[EdgeFeature.PD]),
                                   FEAT: pairwise_depth})

    @staticmethod
    def build(config: Config) -> "GrooviGraph":
        signal_processing = SVSignalProcessor(config)
        junctions = signal_processing.extract_junctions(cluster=True)
        segments = signal_processing.extract_segments(junctions)
        groovi_graph = GrooviGraph()
        groovi_graph.add_nodes(segments)
        groovi_graph.add_segment_edges(segments)
        groovi_graph.add_reference_edges(segments)
        groovi_graph.add_node_features()
        groovi_graph.add_novel_junction_edges(junctions)
        groovi_graph.add_edge_features()
        groovi_graph.vcf_path = config.vcf if hasattr(config, "vcf") else None
        groovi_graph.info_path = config.info if hasattr(config, "info") else None
        groovi_graph.verify()
        return groovi_graph

    def _add_edges(self, src, dst, data):
        self.graph.add_edges(src, dst, data=data)
        self.graph.add_edges(dst, src, data=data)

    def _add_node(self, ref_segment, node_type, idx):
        self.id_to_node[idx] = GrooviNode.from_segment(ref_segment, node_type)
        self.node_to_id[self.id_to_node[idx]] = idx

    def apply_transforms(self, transform: T.BaseTransform) -> "GrooviGraph":
        """Apply dgl transforms to the GrooviGraph in-place, returns self for chaining."""
        transformed_graph = transform(self.graph)
        self.graph = transformed_graph
        # TODO: src/dst?
        return self
