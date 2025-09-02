from groovi.seq.constants import *
from groovi.seq.io import ChrFAIndex
from groovi.seq.signal_extraction import Edge

import numpy as np
from sklearn.cluster import DBSCAN


class Segment:
    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = chrom
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __repr__(self):
        return f"Segment(chrom={self.chrom}, start_pos={self.start_pos}, end_pos={self.end_pos})"
    
    def get_chrom(self):
        return self.chrom
    
    def get_start_pos(self):
        return self.start_pos
    
    def get_end_pos(self):
        return self.end_pos
    
class Adjacency:
    def __init__(self, chrom1, chrom2, pos1, pos2, orientation, evidence, edge_type):
        self.chrom1 = chrom1
        self.chrom2 = chrom2
        self.pos1 = pos1
        self.pos2 = pos2
        self.orientation = orientation
        self.evidence = evidence
        self.edge_type = edge_type

    def get_chrom1(self):
        return self.chrom1

    def get_chrom2(self):
        return self.chrom2

    def get_pos1(self):
        return self.pos1

    def get_pos2(self):
        return self.pos2

    def get_orientation(self):
        return self.orientation
    
    def get_evidence(self):
        return self.evidence

    def get_node_types(self):
        if self.orientation == 'LL':
            return (NodeType.TAIL, NodeType.TAIL)
        elif self.orientation == 'LR':
            return (NodeType.TAIL, NodeType.HEAD)
        elif self.orientation == 'RL':
            return (NodeType.HEAD, NodeType.TAIL)
        elif self.orientation == 'RR':
            return (NodeType.HEAD, NodeType.HEAD)
        else:
            raise ValueError(f"Invalid orientation: {self.orientation}")
    
    def get_edge_type(self):
        return self.edge_type

class AdjIndex:
    def __init__(self, config, edges: list[Edge]):
        self.config = config
        self.fai_index = ChrFAIndex.load(config.fai)
        self.chr_ids = self.fai_index.get_chr_ids()
        self.chr_lengths = self.fai_index.get_chr_lengths()

        self.adj_index = {
            'ref_positions': [], # (n, 2)
            'edge_types': [], # (n,)
            'orientations': [], # (n,)
            'chroms': [], # (n, 2)
            'evidence': [] # (n,)
        }

        for edge in edges:
            self.adj_index['ref_positions'].append((edge.pos1, edge.pos2))
            self.adj_index['edge_types'].append(edge.edge_type)
            self.adj_index['orientations'].append(edge.orientation.value) # cannot make np array of enum
            self.adj_index['chroms'].append((edge.chrom1, edge.chrom2))
            self.adj_index['evidence'].append(edge.evidence)

        for k, v in self.adj_index.items(): self.adj_index[k] = np.array(v)

    def cluster_splitreads(self, voting_metric = lambda positions, reads: -np.sum(reads)) -> None:
        '''
        Mutates AdjIndex, clusters split reads based on their genomic coordinates. For co-clustered signals, use signal of minimum voting_metric scoring fn.
        '''
        # initialize functions for clustering
        eps_edge = self.config.cluster_radius
        min_samples = self.config.cluster_samples
        edge_density_clustering_module = DBSCAN(eps=eps_edge, min_samples=min_samples, metric='manhattan')
        orientation_diffs = {
            'LL': np.array([0,0]),
            'LR': np.array([0,1]),
            'RL': np.array([1,0]),
            'RR': np.array([1,1]),
        }
        prefix_sums = self.fai_index.get_chr_prefix_sums()
        # convert to global coordinates
        global_ref_positions = self._local_to_global(self.adj_index['ref_positions'], self.adj_index['chroms'], prefix_sums)
        # TODO: temporary, sorting of coordinates
        global_ref_positions = np.sort(global_ref_positions, axis=1)

        # Edge Clustering
        edge_centroid_refs, edge_clustered_num_reads, edge_clustered_edge_types, edge_clustered_orientation, chr_names = [], [], [], [], []
        evidence, orientations, edge_types, chroms = self.adj_index['evidence'], self.adj_index['orientations'], self.adj_index['edge_types'], self.adj_index['chroms']
        edge_clustering_info = edge_density_clustering_module.fit(global_ref_positions)
        for c_index in np.unique(edge_clustering_info.labels_):
            assert c_index > -1, f"Unexpected cluster index: {c_index}"
            cluster_mask = edge_clustering_info.labels_ == c_index
            centroid_reads = np.sum(evidence[cluster_mask])
            if centroid_reads <= self.config.evidence_threshold:
                continue
            num_unique_orientations = len(np.unique(orientations[cluster_mask]))
            if num_unique_orientations == 1:
                centroid = np.sum(global_ref_positions[cluster_mask] * evidence[cluster_mask].reshape(-1, 1), axis=0) / centroid_reads # reshape for broadcasting
                centroid = np.round(centroid).astype(int)

                edge_centroid_refs.append(centroid)
                edge_clustered_num_reads.append(centroid_reads)
                edge_clustered_edge_types.append(edge_types[cluster_mask][0])
                chr_names.append(chroms[cluster_mask][0])
                edge_clustered_orientation.append(orientations[cluster_mask][0])
            else:
                # vote on centroid position...
                unique_orientations = np.unique(orientations[cluster_mask])
                orientation_indices = [np.flatnonzero((orientations == orientation) & cluster_mask) for orientation in unique_orientations]
                orientation_reads = [np.sum(evidence[orientation_index]) for orientation_index in orientation_indices]
                orientation_metrics = [voting_metric(global_ref_positions[orientation_index], evidence[orientation_index]) for orientation_index in orientation_indices]
                max_i = np.argmin(orientation_metrics)
                max_orientation = unique_orientations[max_i]
                max_centroid = np.round(np.sum(global_ref_positions[orientation_indices[max_i]] * evidence[orientation_indices[max_i]].reshape(-1, 1), axis=0) / orientation_reads[max_i])
                for i in range(len(unique_orientations)):
                    current_orientation = unique_orientations[i]
                    diff = orientation_diffs[current_orientation] - orientation_diffs[max_orientation]
                    new_centroid = max_centroid + diff

                    edge_centroid_refs.append(new_centroid)
                    edge_clustered_num_reads.append(orientation_reads[i])
                    edge_clustered_edge_types.append(edge_types[orientation_indices[i]][0])
                    chr_names.append(chroms[orientation_indices[i]][0])
                    edge_clustered_orientation.append(current_orientation)

        edge_centroid_refs = np.array(edge_centroid_refs)
        edge_clustered_num_reads = np.array(edge_clustered_num_reads)
        edge_clustered_edge_types = np.array(edge_clustered_edge_types)
        edge_clustered_orientation = np.array(edge_clustered_orientation)
        chr_names = np.array(chr_names)

        edge_centroid_refs = self._global_to_local(edge_centroid_refs, chr_names, prefix_sums)

        self._update_adj_index({
            'ref_positions': edge_centroid_refs,
            'edge_types': edge_clustered_edge_types,
            'orientations': edge_clustered_orientation,
            'chroms': chr_names,
            'evidence': edge_clustered_num_reads,
        })

    def get_segments(self) -> list[Segment]:
        chr_index_refs = {chr_name: [] for chr_name in self.chr_ids}
        chr_index_orientations = {chr_name: [] for chr_name in self.chr_ids}
        for (ref1, ref2), (orientation1, orientation2), (chrom1, chrom2) in zip(self.adj_index['ref_positions'], self.adj_index['orientations'], self.adj_index['chroms']):
            chr_index_refs[chrom1].append(ref1)
            chr_index_refs[chrom2].append(ref2)
            chr_index_orientations[chrom1].append(orientation1)
            chr_index_orientations[chrom2].append(orientation2)

        segments: list[Segment] = []
        for chr_id in self.chr_ids:
            refs = chr_index_refs[chr_id]
            orientations = chr_index_orientations[chr_id]
            if len(refs) == 0: continue

            # collect breakpoints. L defines tail nodes, R defines head nodes
            tail_node_refs = set()
            head_node_refs = set()
            for ref, orientation in zip(refs, orientations):
                if orientation == 'L':
                    tail_node_refs.add(ref)
                elif orientation == 'R':
                    head_node_refs.add(ref)

            # for each break, create the other end of the edge
            for L_ref in tail_node_refs: head_node_refs.add(L_ref + 1)
            for R_ref in head_node_refs: tail_node_refs.add(R_ref - 1)

            # create head and tail for start and end of chromosome
            head_node_refs.add(1)
            tail_node_refs.add(self.chr_lengths[chr_id])

            # sort and create segments
            head_refs = sorted(list(head_node_refs))
            tail_refs = sorted(list(tail_node_refs))

            for head_ref, tail_ref in zip(head_refs, tail_refs):
                segments.append(Segment(chr_id, head_ref, tail_ref))
        
        return segments
        
    def get_adjacencies(self) -> list[Adjacency]:
        adjacencies: list[Adjacency] = []
        for (ref1, ref2), orientation, (chrom1, chrom2), evidence, edge_type in zip(self.adj_index['ref_positions'], self.adj_index['orientations'], self.adj_index['chroms'], self.adj_index['evidence'], self.adj_index['edge_types']):
            adjacencies.append(Adjacency(chrom1, chrom2, ref1, ref2, orientation, evidence, edge_type))
        return adjacencies

    def _local_to_global(self, local_refs: np.ndarray, chr_names: np.ndarray, prefix_sum: dict[int, int]) -> np.ndarray:
        global_refs = []
        for (local_ref_start, local_ref_end), (chr_start, chr_end) in zip(local_refs, chr_names):
            global_ref_start = prefix_sum[chr_start] + local_ref_start
            global_ref_end = prefix_sum[chr_end] + local_ref_end
            global_refs.append((global_ref_start, global_ref_end))
        return np.array(global_refs)

    def _global_to_local(self, global_refs: np.ndarray, chr_names: np.ndarray, prefix_sum: dict[int, int]) -> np.ndarray:
        local_refs = []
        for (global_ref_start, global_ref_end), (chr_start, chr_end) in zip(global_refs, chr_names):
            local_ref_start = global_ref_start - prefix_sum[chr_start]
            local_ref_end = global_ref_end - prefix_sum[chr_end]
            local_refs.append((local_ref_start, local_ref_end))
        return np.array(local_refs)

    def _update_adj_index(self, dict: dict) -> None:
        for k, v in dict.items():
            assert k in self.adj_index, f"Key {k} not in adjacency index."
            self.adj_index[k] = v
    
