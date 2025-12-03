from distutils.command.config import config

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
from groovi.seq.sv import *


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
    A Groovi graph is a feature-augmented breakpoint graph
    It is internally represented using a DGL graph with the following properties:
    Node features (graph.ndata):
        - 'HT': Tensor of shape (num_nodes,) indicating if this is a HEAD or TAIL (0/1) node
        - FEAT: Tensor of shape (num_nodes, num_node_features)
    Edge features (graph.edata):
        - FEAT_TYPE: Tensor of shape (num_edges,) indicating the type of each edge
        - FEAT: Tensor of shape (num_edges, num_edge_features) with feature values

    Additional attributes:
        - candidate_src: Tensor of shape (num_candidate_edges,) indicating source nodes of candidate edges
        - candidate_dst: Tensor of shape (num_candidate_edges,) indicating destination nodes of candidate edges
        - vcf_path: str, path to the VCF file associated with this graph, optional
        - info_path: str, path to the info file associated with this graph, intended for graph classification tasks, optional
    """

    def __init__(self, config):
        self.config = config
        self.dgl_graph: dgl.DGLGraph = dgl.graph(([], []))
        self.candidate_src: torch.LongTensor = []
        self.candidate_dst: torch.LongTensor = []
        self.candidate_support: torch.LongTensor = []
        self.edges2support = defaultdict(int)
        self.edges2source = defaultdict(list)
        self.node_to_id = {}
        self.id_to_node = {}
        self.vcf_path: str = None
        self.info_path: str = None

    def __repr__(self):
        return f"GrooviGraph(num_nodes={self.dgl_graph.num_nodes()}, num_edges={self.dgl_graph.num_edges()})"

    def get_stats(self):
        return {'num_nodes': self.dgl_graph.num_nodes(),
                'num_edges':  self.dgl_graph.num_edges(),
                'vcf':  self.vcf_path,
                'candidate_src':  self.candidate_src,
                'candidate_dst':  self.candidate_dst}

    def add_nodes(self, ref_segments, node_features):
        head_segment_stats, tail_segment_stats = node_features
        node_idx = 0
        for seq_idx, segment in enumerate(ref_segments):
            self._add_node(ref_segment=segment, node_type=NodeType.HEAD, idx=node_idx)
            self._add_node(ref_segment=segment, node_type=NodeType.TAIL, idx=node_idx + 1)

            head_feat = torch.ones(self.config.n_node_features, dtype=torch.float32)
            tail_feat = torch.ones(self.config.n_node_features, dtype=torch.float32)
            for aln_stat in head_segment_stats[seq_idx]:
                feature_idx = NODE_FEATURE_TO_ID[ALNSTAT_TO_NODE_FEAT[aln_stat]]
                head_feat[feature_idx] = head_segment_stats[seq_idx][aln_stat]
                tail_feat[feature_idx] = tail_segment_stats[seq_idx][aln_stat]
            self.dgl_graph.add_nodes(2, data={"HT": torch.tensor([0, 1], dtype=torch.int64), # 0 for HEAD, 1 for TAIL
                                              FEAT: torch.stack([head_feat, tail_feat], dim=0)})
            node_idx += 2

    def add_segment_edges(self, ref_segments):
        for segment in ref_segments:
            head_id = self.node_to_id[GrooviNode.from_segment(segment, NodeType.HEAD)]
            tail_id = self.node_to_id[GrooviNode.from_segment(segment, NodeType.TAIL)]
            self._add_edges(head_id, tail_id, data={
                FEAT_TYPE: torch.tensor([EDGE_FEATURE_TO_ID[EdgeFeature.SEG]], dtype=torch.int64),
                FEAT: torch.tensor([[1]], dtype=torch.float32)})
            if abs(segment.start - segment.end) <= self.config.tiny_segment_threshold:
                self._add_edges(head_id, tail_id, data={
                    FEAT_TYPE: torch.tensor([EDGE_FEATURE_TO_ID[EdgeFeature.IS_TINY_SEG]], dtype=torch.int64),
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
            self.candidate_support.append(j.support)
            self.edges2support[(src_node_id, dst_node_id)] = j.support
            self.edges2source[(src_node_id, dst_node_id)] = [j.source] # TODO: update after clustering change
        self.candidate_src = torch.tensor(self.candidate_src, dtype=torch.int64)
        self.candidate_dst = torch.tensor(self.candidate_dst, dtype=torch.int64)
        self.candidate_support = torch.tensor(self.candidate_support, dtype=torch.float64)


    def add_edge_features(self, split_by_sign=True):
        # edge feature: change in depth
        edge_mask = ((self.dgl_graph.edata[FEAT_TYPE] != EDGE_FEATURE_TO_ID[EdgeFeature.SEG]) &
                     (self.dgl_graph.edata[FEAT_TYPE] != EDGE_FEATURE_TO_ID[EdgeFeature.IS_TINY_SEG]))
        u, v = self.dgl_graph.edges()
        edge_mask = edge_mask & (u < v)
        u, v = u[edge_mask], v[edge_mask]
        rd_min = self.dgl_graph.ndata[FEAT][u, NODE_FEATURE_TO_ID[NodeFeature.RD]]
        rd_max = self.dgl_graph.ndata[FEAT][v, NODE_FEATURE_TO_ID[NodeFeature.RD]]
        rd_delta = (rd_min - rd_max).unsqueeze(-1)

        if not split_by_sign:
            self._add_edges(u, v, data={FEAT_TYPE: torch.full((len(u),), EDGE_FEATURE_TO_ID[EdgeFeature.RD]), FEAT: rd_delta})
        else:
            pos_mask = (rd_delta >= 0).squeeze()
            neg_mask = ~pos_mask
            if pos_mask.any():
                self._add_edges(u[pos_mask], v[pos_mask],
                                data={FEAT_TYPE: torch.full((pos_mask.sum(),), EDGE_FEATURE_TO_ID[EdgeFeature.RD_POS]),
                                      FEAT: rd_delta[pos_mask]})
            if neg_mask.any():
                self._add_edges(u[neg_mask], v[neg_mask],
                                data={FEAT_TYPE: torch.full((neg_mask.sum(),), EDGE_FEATURE_TO_ID[EdgeFeature.RD_NEG]),
                                      FEAT: rd_delta[neg_mask].abs()})

    def get_edge_support(self, src_node_id, dst_node_id):
        _, _, edge_ids = self.dgl_graph.edge_ids(src_node_id, dst_node_id, return_uv=True)
        if len(edge_ids) == 0: return 0
        edge_type_ids = self.dgl_graph.edata[FEAT_TYPE][edge_ids]
        total_support = 0
        edge_types = set()
        for i, e_type_id in enumerate(edge_type_ids):
            if ID_TO_EDGE_FEATURE[e_type_id.item()] in CANDIDATE_EDGE_FEATURE_TYPES:
                total_support += int(self.dgl_graph.edata[FEAT][edge_ids][i].item())
                edge_types.add(ID_TO_EDGE_FEATURE[e_type_id.item()].name)
        return total_support, edge_types

    def add_hint(self):
        for src_node_id, dst_node_id in zip(self.candidate_src, self.candidate_dst):
            degree = (self.dgl_graph.out_degrees(self.dgl_graph.predecessors(src_node_id.item()) ).max() +
                      self.dgl_graph.out_degrees( self.dgl_graph.successors(src_node_id.item())).max())
            edge_type_feat_id = EDGE_FEATURE_TO_ID[EdgeFeature.HINT]
            self._add_edges(src_node_id, dst_node_id, data={
                    FEAT_TYPE: torch.tensor([edge_type_feat_id], dtype=torch.int64),
                    FEAT: torch.tensor([[degree]], dtype=torch.float32)})

    @staticmethod
    def build(config: Config) -> "GrooviGraph":
        logging.info("Building Groovi graph...")
        signal_processing = SVSignalProcessor(config)
        junctions = signal_processing.extract_junctions(cluster=config.cluster)
        segments = signal_processing.extract_segments(junctions)
        groovi_graph = GrooviGraph(config)
        groovi_graph.add_nodes(segments, node_features=signal_processing.generate_stats(segments))
        groovi_graph.add_segment_edges(segments)
        groovi_graph.add_reference_edges(segments)
        groovi_graph.add_novel_junction_edges(junctions)
        groovi_graph.add_edge_features()
        groovi_graph.vcf_path = config.vcf if hasattr(config, "vcf") else None
        groovi_graph.info_path = config.info if hasattr(config, "info") else None
        logging.info("Finished graph construction")
        return groovi_graph

    def _add_edges(self, src, dst, data, directed=False):
        self.dgl_graph.add_edges(src, dst, data=data)
        if not directed:
            self.dgl_graph.add_edges(dst, src, data=data)

    def _add_node(self, ref_segment, node_type, idx):
        self.id_to_node[idx] = GrooviNode.from_segment(ref_segment, node_type)
        self.node_to_id[self.id_to_node[idx]] = idx

    def apply_transforms(self, transform: T.BaseTransform):
        self.dgl_graph = transform(self.dgl_graph)
        # TODO: src/dst?

    def filter_candidate_edges(self, min_candidate_support=1):
        high_evidence_mask = self.candidate_support >= min_candidate_support
        self.candidate_src = self.candidate_src[high_evidence_mask]
        self.candidate_dst = self.candidate_dst[high_evidence_mask]
        self.candidate_support = self.candidate_support[high_evidence_mask]