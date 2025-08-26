from groovi.graph.graphs import GrooviGraph
from groovi.graph.transforms import *
from groovi.seq.signal_extraction import SignalExtractor, Edge
from groovi.seq.constants import *

from collections import OrderedDict, deque
import dgl
import numpy as np
from sklearn.cluster import DBSCAN
import torch

class GraphConstructor:
    '''
    Given an alignment index, the generate method constructs a breakpoint graph:
    1. Clustering and filtering of split reads, creating aggregated breakpoint centroids
    2. Construct head/tail nodes with segment and reference edges
    3. Populate Node Features
    4. Construct adjacency edges
    5. Populate Edge Features

    This class is intended to provide a base data representation
    Any additional transformations should be done by calling transformations (Normalization, PD Edges, etc)
    '''
    def __init__(self, config):
        self.signal_extractor = SignalExtractor(config)
        self.config = config
        self.rel_map = EDGE_RELATION_MAP # map SVSignal to int, used for edge types
        self.chr_names = self.signal_extractor.get_chr_names()
        self.chr_lengths = self.signal_extractor.get_chr_lengths()
        self.max_bp = sum([L for _, L in self.chr_lengths.items()])
        self.chr_prefix_sums = self._get_chr_prefix_sums() 

    def generate_groovi_graphs(self) -> list[GrooviGraph]:
        # primary construction
        adj_index = self._get_adjacencies()
        references_clustered, num_reads_clustered, spo_clustered, orientation_clustered, chr_names = self._cluster_splitreads(adj_index)
        g, candidate_src, candidate_dst = self._generate_base_breakpoint_graph(references_clustered, orientation_clustered, chr_names)
        g = self._aggregate_node_features(g)
        g = self._generate_adjacency_edges_and_features(g, candidate_src, candidate_dst, num_reads_clustered, spo_clustered)
        # graph validation
        self._validate_graph(g)

        # convert to GrooviGraph(s)
        groovi_graph = GrooviGraph.from_dgl_graph(g)
        groovi_graph.set_candidate_edges(candidate_src, candidate_dst)
        groovi_graph.set_vcf_path(self.config.vcf)
        groovi_graph.set_chr_ids_included(np.unique(chr_names).astype(str))
        groovi_graph_components = self._split_by_components(groovi_graph)

        return groovi_graph_components

    '''
    Helper functions
    '''
    def _split_by_components(self, graph: GrooviGraph) -> list[GrooviGraph]:
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
    
    def _get_chr_prefix_sums(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping chromosome names to their starting positions in the genome.
        '''
        chr_names = list(self.chr_lengths.keys())
        print(f'Chromosomes included in graph: {chr_names}')
        chr_prefix_sums = OrderedDict({chr_names[0]: 0})  # start with first chromosome at position 0
        for i, chr_name in enumerate(chr_names):
            if i == 0: continue
            chr_prefix_sums[chr_name] = chr_prefix_sums[chr_names[i-1]] + self.chr_lengths[chr_names[i-1]]
        return chr_prefix_sums
    
    def _get_adjacencies(self) -> dict[str, np.ndarray]:
        edges: list[Edge] = self.signal_extractor.get_adj()
        adj_index = {
            'ref_positions': [], # (n, 2)
            'edge_types': [], # (n,)
            'orientations': [], # (n,)
            'chroms': [], # (n, 2)
            'evidence': [] # (n,)
        }

        for edge in edges:
            adj_index['ref_positions'].append((edge.pos1, edge.pos2))
            adj_index['edge_types'].append(edge.edge_type)
            adj_index['orientations'].append(edge.orientation.value) # cannot make np array of enum
            adj_index['chroms'].append((edge.chrom1, edge.chrom2))
            adj_index['evidence'].append(edge.evidence)       
        
        for k, v in adj_index.items(): adj_index[k] = np.array(v)

        return adj_index

    def _local_to_global(self, local_refs: np.ndarray, chr_names: np.ndarray) -> np.ndarray:
        global_refs = []
        for (local_ref_start, local_ref_end), (chr_start, chr_end) in zip(local_refs, chr_names):
            global_ref_start = self.chr_prefix_sums[chr_start] + local_ref_start
            global_ref_end = self.chr_prefix_sums[chr_end] + local_ref_end
            global_refs.append((global_ref_start, global_ref_end))
        return np.array(global_refs)
    
    def _global_to_local(self, global_refs: np.ndarray, chr_names: np.ndarray) -> np.ndarray:
        local_refs = []
        for (global_ref_start, global_ref_end), (chr_start, chr_end) in zip(global_refs, chr_names):
            local_ref_start = global_ref_start - self.chr_prefix_sums[chr_start]
            local_ref_end = global_ref_end - self.chr_prefix_sums[chr_end]
            local_refs.append((local_ref_start, local_ref_end))
        return np.array(local_refs)        

    def _cluster_splitreads(self, adj_index: dict[str, np.array], voting_metric = lambda positions, reads: -np.sum(reads)):
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
        # convert to global coordinates
        global_ref_positions = self._local_to_global(adj_index['ref_positions'], adj_index['chroms'])
        # TODO: temporary, sorting of coordinates
        global_ref_positions = np.sort(global_ref_positions, axis=1)

        # Edge Clustering
        edge_centroid_refs, edge_clustered_num_reads, edge_clustered_spo, edge_clustered_orientation, chr_names = [], [], [], [], []
        evidence, orientations, edge_types, chroms = adj_index['evidence'], adj_index['orientations'], adj_index['edge_types'], adj_index['chroms']
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
                edge_clustered_spo.append(edge_types[cluster_mask][0])
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
                    edge_clustered_spo.append(edge_types[orientation_indices[i]][0])
                    chr_names.append(chroms[orientation_indices[i]][0])
                    edge_clustered_orientation.append(current_orientation)

        edge_centroid_refs = np.array(edge_centroid_refs)
        edge_clustered_num_reads = np.array(edge_clustered_num_reads)
        edge_clustered_spo = np.array(edge_clustered_spo)
        edge_clustered_orientation = np.array(edge_clustered_orientation)
        chr_names = np.array(chr_names)

        edge_centroid_refs = self._global_to_local(edge_centroid_refs, chr_names)
        return edge_centroid_refs, edge_clustered_num_reads, edge_clustered_spo, edge_clustered_orientation, chr_names

    def _generate_base_breakpoint_graph(self, paired_refs: np.ndarray, paired_orientations: np.ndarray, paired_chr_names: np.ndarray) -> dgl.DGLGraph:
        # build index by chromosome
        chr_index_refs = {chr_name: [] for chr_name in self.chr_names}
        chr_index_orientations = {chr_name: [] for chr_name in self.chr_names}
        for (ref1, ref2), (orientation1, orientation2), (chrom1, chrom2) in zip(paired_refs, paired_orientations, paired_chr_names):
            chr_index_refs[chrom1].append(ref1)
            chr_index_refs[chrom2].append(ref2)
            chr_index_orientations[chrom1].append(orientation1)
            chr_index_orientations[chrom2].append(orientation2)

        node_iter_start = 0
        node_iter = 0
        head_ref_to_node_id = {} # map (chr_name, ref) to node id
        tail_ref_to_node_id = {} # map (chr_name, ref) to node id
        seg_rel, ref_rel = self.rel_map[EdgeRelation.SEG], self.rel_map[EdgeRelation.REF]
        g: dgl.DGLGraph = dgl.graph(([], []))
        
        for chr_name in self.chr_names:
            refs = chr_index_refs[chr_name]
            orientations = chr_index_orientations[chr_name]
            if len(refs) == 0: continue

            # collect breakpoints
            L_set = set()
            R_set = set()
            for ref, orientation in zip(refs, orientations):
                if orientation == 'L':
                    L_set.add(ref)
                elif orientation == 'R':
                    R_set.add(ref)

            # for each break, create the other end of the edge
            for L_ref in L_set: R_set.add(L_ref + 1)
            for R_ref in R_set: L_set.add(R_ref - 1)

            # create head and tail for start and end of chromosome
            R_set.add(1)
            L_set.add(self.chr_lengths[chr_name])

            # convert to sorted list
            head_refs = sorted(list(R_set))
            tail_refs = sorted(list(L_set))

            # initialize node annotations 
            position_node_annotations: list[int] = []
            chr_name_node_annotations: list[str] = []

            # create nodes for head and tail, while populating annotations
            for head_ref, tail_ref in zip(head_refs, tail_refs):

                head_ref_to_node_id[(chr_name, head_ref)] = node_iter
                node_iter += 1

                tail_ref_to_node_id[(chr_name, tail_ref)] = node_iter
                node_iter += 1

                position_node_annotations.extend([head_ref, tail_ref])
                chr_name_node_annotations.extend([chr_name, chr_name])

            print(f'{len(position_node_annotations)} nodes created for chromosome {chr_name}.')
            print(f'{len(chr_name_node_annotations)} chr annotations created for chromosome {chr_name}.')
            g.add_nodes(node_iter - node_iter_start, data={
                'position': torch.tensor(position_node_annotations, dtype=torch.int64),
                'chr': torch.tensor(chr_name_node_annotations, dtype=torch.int64)
            })

            src = torch.arange(node_iter_start, node_iter-1, dtype=torch.int64)
            dst = torch.arange(node_iter_start+1, node_iter, dtype=torch.int64)
            m = len(src)
            feat = torch.ones((m,1), dtype=torch.float32)
            g.add_edges(src, dst, data={'rel_type':  torch.tensor([seg_rel, ref_rel]).repeat(m)[:m], 'feat': feat})
            g.add_edges(dst, src, data={'rel_type':  torch.tensor([seg_rel, ref_rel]).repeat(m)[:m], 'feat': feat})

            node_iter_start = node_iter

        # get candidate src and dst
        candidate_src: list[int] = []
        candidate_dst: list[int] = []
        for paired_refs, orientation, chr_name_pair in zip(paired_refs, paired_orientations, paired_chr_names):
            src_ref, dst_ref = paired_refs
            src_o, dst_o = orientation
            src_chrom, dst_chrom = chr_name_pair
            if src_o == 'L':
                candidate_src.append(tail_ref_to_node_id[(src_chrom,src_ref)])
            elif src_o == 'R':
                candidate_src.append(head_ref_to_node_id[(src_chrom,src_ref)])
            if dst_o == 'L':
                candidate_dst.append(tail_ref_to_node_id[(dst_chrom,dst_ref)])
            elif dst_o == 'R':
                candidate_dst.append(head_ref_to_node_id[(dst_chrom,dst_ref)])

        candidate_src, candidate_dst = torch.tensor(candidate_src, dtype=torch.int64), torch.tensor(candidate_dst, dtype=torch.int64)
        return g, candidate_src, candidate_dst

    def _aggregate_node_features(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        # TODO: replace with actual features
        g.ndata['feat'] = torch.randn((g.num_nodes(), 5))
        return g

    def _generate_adjacency_edges_and_features(self, g: dgl.DGLGraph, candidate_src: list[int], candidate_dst: list[int], candidate_num_reads: list[int], candidate_edge_types: list[EdgeRelation]) -> dgl.DGLGraph:
        # novel adjacency edges
        for candidate in zip(candidate_src, candidate_dst, candidate_num_reads, candidate_edge_types):
            src, dst, num_reads, edge_type = candidate
            rel_type: torch.LongTensor = torch.tensor([EDGE_RELATION_MAP[edge_type]])
            feat = torch.tensor([num_reads], dtype=torch.float32).unsqueeze(-1)
            g.add_edges(src, dst, data={'rel_type': rel_type, 'feat': feat})
            g.add_edges(dst, src, data={'rel_type': rel_type, 'feat': feat})

        # pairwise depth edges
        u, v = g.edges()
        seg_rel, ref_rel = self.rel_map[EdgeRelation.SEG], self.rel_map[EdgeRelation.REF]
        edge_mask = (g.edata['rel_type'] == seg_rel) | (g.edata['rel_type'] == ref_rel)
        u, v = u[edge_mask], v[edge_mask]
        m = len(u)
        rd_index = NODE_FEATURE_MAP[NodeFeature.RD]
        pd_edge_id = EDGE_RELATION_MAP[EdgeRelation.PD]
        pairwise_depth = (g.ndata['feat'][v,rd_index] - g.ndata['feat'][u,rd_index]).unsqueeze(-1)
        rel_type = pd_edge_id*torch.ones((m,), dtype=torch.int64)
        g.add_edges(u, v, data={'feat' : pairwise_depth, 'rel_type': rel_type})
        g.add_edges(v, u, data={'feat' : pairwise_depth, 'rel_type': rel_type})

        # add distances
        u, v, _ = g.edges(form='all')
        distances = torch.abs(g.ndata['position'][u] - g.ndata['position'][v])
        g.edata['distance'] = distances
        return g
    
    def _validate_graph(self, g: dgl.DGLGraph):
        node_keys = ['feat', 'position', 'chr']
        edge_keys = ['rel_type', 'feat', 'distance']
        for key in node_keys:
            assert key in g.ndata, f"Node data missing key: {key}"
        for key in edge_keys:
            assert key in g.edata, f"Edge data missing key: {key}"

        # Ensure Features are of correct shape
        assert len(g.ndata['feat'].shape) == 2, 'Node features must be 2D'
        assert len(g.ndata['feat'].shape) == 2, 'Node features must be 2D'

        assert len(g.edata['feat'].shape) == 2, 'Edge features must be 2D'
        assert len(g.edata['rel_type'].shape) == 1, 'Edge relation types must be 1D'

        # ensure no self loops for specific edge types
        self_loop_types_int = [EDGE_RELATION_MAP[EdgeRelation.SEG], EDGE_RELATION_MAP[EdgeRelation.REF]]
        self_loop_types_tensor = torch.tensor(self_loop_types_int, dtype=torch.int64, device=g.device)
        u_all, v_all, eid_all = g.edges(form='all')
        is_self_loop_mask = (u_all == v_all)
        actual_self_loops_eids = eid_all[is_self_loop_mask]
        self_loop_rel_types = g.edata['rel_type'][actual_self_loops_eids]
        target_self_loops_mask = torch.isin(self_loop_rel_types, self_loop_types_tensor)
        problematic_self_loop_eids = actual_self_loops_eids[target_self_loops_mask]
        assert len(problematic_self_loop_eids) == 0, \
            f"Graph contains self-loop edges of types {self_loop_types_int} with EIDs: {problematic_self_loop_eids.tolist()}"
        
        # ensure relationship types alternate between seg/ref. missing a reference edge is okay.
        next_invalid_edge_type = self.rel_map[EdgeRelation.REF]
        next_valid_edge_type = self.rel_map[EdgeRelation.SEG]
        for node_id_prev, node_id_next in zip(range(g.num_nodes() - 1), range(1, g.num_nodes())):
            _, _, eids = g.edge_ids(node_id_prev, node_id_next, return_uv=True)
            rel_types = g.edata['rel_type'][eids]
            assert (next_valid_edge_type in rel_types) or (next_valid_edge_type == self.rel_map[EdgeRelation.REF] and len(rel_types) == 0), \
                f"Graph edges between {node_id_prev} and {node_id_next} must include {next_valid_edge_type}, but found {rel_types.tolist()}"
            assert next_invalid_edge_type not in rel_types, \
                f"Graph edges between {node_id_prev} and {node_id_next} must not include {next_invalid_edge_type}, but found {rel_types.tolist()}"
            next_valid_edge_type, next_invalid_edge_type = next_invalid_edge_type, next_valid_edge_type  


