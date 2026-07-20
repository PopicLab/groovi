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
        - candidate_support: Tensor of shape (num_candidate_edges,) indicating read support of candidate edges
    """

    def __init__(self, config):
        self.config = config
        self.dgl_graph: dgl.DGLGraph = dgl.graph(([], []))
        self.candidate_src: torch.LongTensor = []
        self.candidate_dst: torch.LongTensor = []
        self.candidate_support: torch.LongTensor = []
        self.edges2source = defaultdict(list)
        self.node_to_id = {}
        self.id_to_node = {}

    def __repr__(self):
        return f"GrooviGraph(num_nodes={self.dgl_graph.num_nodes()}, num_edges={self.dgl_graph.num_edges()})"

    def get_stats(self):
        return {'num_nodes': self.dgl_graph.num_nodes(),
                'num_edges':  self.dgl_graph.num_edges(),
                'candidate_src':  self.candidate_src,
                'candidate_dst':  self.candidate_dst,
                'candidate_support':  self.candidate_support}

    def add_nodes(self, ref_segments, node_stats):
        head_segment_stats, tail_segment_stats = node_stats
        n = len(ref_segments)

        for idx, segment in enumerate(ref_segments):
            head_node_idx, tail_node_idx = idx*2, idx*2+1
            self._add_node(ref_segment=segment, node_type=NodeType.HEAD, idx=head_node_idx)
            self._add_node(ref_segment=segment, node_type=NodeType.TAIL, idx=tail_node_idx)
        
        head_feats = torch.ones(n, self.config.n_node_features, dtype=torch.float32)
        tail_feats = torch.ones(n, self.config.n_node_features, dtype=torch.float32)

        for i, node_feature in enumerate(self.config.node_feature_set):
            if node_feature in AUX_NODE_FEATURES:
                continue
            aln_stat = NODE_FEAT_TO_ALNSTAT[node_feature]
            head_feats[:, i] = torch.tensor([s[aln_stat] for s in head_segment_stats], dtype=torch.float32)
            tail_feats[:, i] = torch.tensor([s[aln_stat] for s in tail_segment_stats], dtype=torch.float32)

        if NodeFeature.HT in self.config.node_feature_set:
            ht_idx = self.config.node_feat2id[NodeFeature.HT]
            head_feats[:, ht_idx] = 0
            tail_feats[:, ht_idx] = 1

        all_feats = torch.stack([head_feats, tail_feats], dim=1).view(2 * n, self.config.n_node_features)
        ht_labels = torch.tensor([0, 1] * n, dtype=torch.int64) # TODO: this is redundant with the new node feature set(s) setup

        self.dgl_graph.add_nodes(2 * n, data={HT: ht_labels, FEAT: all_feats})

    def add_segment_edges(self, ref_segments):
        head_ids = torch.tensor([self.node_to_id[GrooviNode.from_segment(s, NodeType.HEAD)] for s in ref_segments], dtype=torch.int64)
        tail_ids = torch.tensor([self.node_to_id[GrooviNode.from_segment(s, NodeType.TAIL)] for s in ref_segments], dtype=torch.int64)
        n = len(ref_segments)

        self._add_edges(head_ids, tail_ids, data={
            FEAT_TYPE: torch.full((n,), self.config.edge_feat2id[EdgeFeature.SEG], dtype=torch.int64),
            FEAT: torch.ones((n, 1), dtype=torch.float32)
        })

        tiny_mask = torch.tensor([abs(s.start - s.end) <= self.config.tiny_segment_threshold for s in ref_segments])
        if tiny_mask.any():
            num_tiny_edges = tiny_mask.sum().item()
            self._add_edges(head_ids[tiny_mask], tail_ids[tiny_mask], data={
                FEAT_TYPE: torch.full((num_tiny_edges,), self.config.edge_feat2id[EdgeFeature.IS_TINY_SEG], dtype=torch.int64),
                FEAT: torch.ones((num_tiny_edges, 1), dtype=torch.float32)
            })

    def add_reference_edges(self, ref_segments):
        same_chr = [i for i in range(len(ref_segments) - 1) if ref_segments[i].chr == ref_segments[i + 1].chr]
        if not same_chr:
            return
        tail_ids = torch.tensor([self.node_to_id[GrooviNode.from_segment(ref_segments[i], NodeType.TAIL)] for i in same_chr], dtype=torch.int64)
        head_ids = torch.tensor([self.node_to_id[GrooviNode.from_segment(ref_segments[i + 1], NodeType.HEAD)] for i in same_chr], dtype=torch.int64)

        n = len(same_chr)
        self._add_edges(tail_ids, head_ids, data={
            FEAT_TYPE: torch.full((n,), self.config.edge_feat2id[EdgeFeature.REF], dtype=torch.int64),
            FEAT: torch.ones((n, 1), dtype=torch.float32)
        })

    def add_novel_junction_edges(self, junctions):
        src_ids, dst_ids, supports, feat_ids = [], [], [], []

        for j in junctions:
            node_type_src, node_type_dst = ORIENTATION_TO_NODE_TYPES[j.orientation]
            src_node_id = self.node_to_id[GrooviNode(j.chr1, j.pos1, node_type_src)]
            dst_node_id = self.node_to_id[GrooviNode(j.chr2, j.pos2, node_type_dst)]
            src_ids.append(src_node_id)
            dst_ids.append(dst_node_id)
            supports.append(j.support)
            feat_ids.append(self.config.edge_feat2id[ORIENTATION_TO_EDGE_TYPE[j.orientation]])
            self.edges2source[(src_node_id, dst_node_id)] = [j.source]

        self.candidate_src = torch.tensor(src_ids, dtype=torch.int64)
        self.candidate_dst = torch.tensor(dst_ids, dtype=torch.int64)
        self.candidate_support = torch.tensor(supports, dtype=torch.int64) # TODO: check necessity of this int casting?

        if src_ids:
            self._add_edges(self.candidate_src, self.candidate_dst, data={
                FEAT_TYPE: torch.tensor(feat_ids, dtype=torch.int64),
                FEAT: self.candidate_support.float().unsqueeze(-1)
            })

    def add_edge_features(self, split_by_sign=True):
        # edge feature: change in depth
        edge_mask = ((self.dgl_graph.edata[FEAT_TYPE] != self.config.edge_feat2id[EdgeFeature.SEG]) &
                     (self.dgl_graph.edata[FEAT_TYPE] != self.config.edge_feat2id[EdgeFeature.IS_TINY_SEG]))
        u, v = self.dgl_graph.edges()
        edge_mask = edge_mask & (u < v)
        u, v = u[edge_mask], v[edge_mask]
        rd_min = self.dgl_graph.ndata[FEAT][u, self.config.node_feat2id[NodeFeature.RD]]
        rd_max = self.dgl_graph.ndata[FEAT][v, self.config.node_feat2id[NodeFeature.RD]]
        rd_delta = (rd_min - rd_max).unsqueeze(-1)

        if not split_by_sign:
            self._add_edges(u, v, data={FEAT_TYPE: torch.full((len(u),), self.config.edge_feat2id[EdgeFeature.RD]),
                                        FEAT: rd_delta})
        else:
            pos_mask = (rd_delta >= 0).squeeze()
            neg_mask = ~pos_mask
            if pos_mask.any():
                self._add_edges(u[pos_mask], v[pos_mask],
                                data={FEAT_TYPE: torch.full((pos_mask.sum(),), self.config.edge_feat2id[EdgeFeature.RD_POS]),
                                      FEAT: rd_delta[pos_mask]})
            if neg_mask.any():
                self._add_edges(u[neg_mask], v[neg_mask],
                                data={FEAT_TYPE: torch.full((neg_mask.sum(),), self.config.edge_feat2id[EdgeFeature.RD_NEG]),
                                      FEAT: rd_delta[neg_mask].abs()})

    def get_edge_types(self, src_node_id, dst_node_id):
        _, _, edge_ids = self.dgl_graph.edge_ids(src_node_id, dst_node_id, return_uv=True)
        if len(edge_ids) == 0: return 0
        edge_type_ids = self.dgl_graph.edata[FEAT_TYPE][edge_ids]
        edge_types = set()
        for i, e_type_id in enumerate(edge_type_ids):
            edge_feat = self.config.id2edge_feat[e_type_id.item()]
            if edge_feat in CANDIDATE_EDGE_FEATURE_TYPES:
                edge_types.add(edge_feat.name)
        return edge_types

    @staticmethod
    def build(config: Config) -> "GrooviGraph":
        signal_processing = SVSignalProcessor(config)
        junctions = signal_processing.extract_junctions(cluster=config.cluster)
        segments = signal_processing.extract_segments(junctions)

        groovi_graph = GrooviGraph(config)
        logging.info("Collecting stats")
        node_stats = signal_processing.generate_stats(segments)

        logging.info("Adding nodes and node features")
        groovi_graph.add_nodes(segments, node_stats=node_stats)

        logging.info("Adding segment edges")
        groovi_graph.add_segment_edges(segments)

        logging.info("Adding reference edges")
        groovi_graph.add_reference_edges(segments)

        logging.info("Adding novel adjacency edges")
        groovi_graph.add_novel_junction_edges(junctions)

        logging.info("Filtering novel adjacency edges")
        groovi_graph.filter_candidate_edges(config.candidate_support_threshold)

        logging.info("Graph construction complete")
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

    def filter_candidate_edges(self, min_candidate_support=1):
        high_evidence_mask = self.candidate_support >= min_candidate_support
        self.candidate_src = self.candidate_src[high_evidence_mask]
        self.candidate_dst = self.candidate_dst[high_evidence_mask]
        self.candidate_support = self.candidate_support[high_evidence_mask]
