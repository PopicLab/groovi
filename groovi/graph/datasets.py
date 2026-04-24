from groovi.graph.constants import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.stats import DatasetStats
from groovi.seq.sv import *
from groovi.graph.transforms import TRANSFORMS
from groovi.graph.visualize import visualize_groovi_graph

import dgl.transforms as T
import numpy as np
import os
import pickle
import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import yaml

class StreamingGraphDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stats = DatasetStats(config)

    def save(self, graph) -> None:
        graph_file_path = os.path.join(self.config.graph_dir, f"graph_full.pkl")
        with open(graph_file_path, 'wb') as graph_file: pickle.dump(graph, graph_file)
        with open(self.config.stats_file_path, 'w') as stats_file:
            stats_file.write('\n'.join(f"{k}: {v}" for k, v in graph.get_stats().items()))

    def __iter__(self):
         if self.config.use_cached_graphs:
            graph_file_path = os.path.join(self.config.graph_dir, f"graph_full.pkl")
            with open(graph_file_path, 'rb') as graph_file: graph = pickle.load(graph_file)
            if self.config.save_graphs: self.save(graph)
        else:
            graph = GrooviGraph.build(self.config)
        graph.apply_transforms(T.Compose(self.config.transforms))
        self.stats.write_stats([graph], "streaming")
        if self.config.visualize_graphs:
            visualize_groovi_graph(self.config, graph, os.path.join(self.config.img_dir, 'graph.html'))
        yield graph


class StaticDataset(Dataset):
    def __init__(self, config, graph_paths: list[str], graphs_info: list[str], tag: str ):
        super().__init__()
        self.config = config
        self.stats = DatasetStats(config)

        self.graphs: list[GrooviGraph] = []
        self.graph_paths = graph_paths
        self.annotations: list[str] = graphs_info
        self.labels = []
        self.transforms = T.Compose(self.config.transforms)
        self.dataset_tag = tag

        logging.info(f"Graph loading/transform...")
        for path in tqdm(self.graph_paths):
            g = self.load(path)
            g.filter_candidate_edges(self.config.candidate_support_threshold)
            g.apply_transforms(self.transforms)
            self.graphs.append(g)

    @staticmethod
    def load(path: str) -> GrooviGraph:
        with open(path, 'rb') as f: return pickle.load(f)

    def __len__(self):
        return len(self.graphs)

class StaticEdgeDataset(StaticDataset):
    def __init__(self, config, graph_paths: list[str], graphs_info: list[str], tag: str):
        super().__init__(config, graph_paths, graphs_info, tag)
        # perform edge labeling
        logging.info(f"Graph labeling...")
        label_summaries = []
        all_sv_class_matches = []
        for i, graph_info in enumerate(zip(self.graphs, self.annotations)):
            graph, vcf_path = graph_info
            tree: SVDB = SVDB(vcf_path, config.classes)
            sv_class_matches = []
            label_summary = [f"Graph {i}: {vcf_path}"]
            for src, dst, support in zip(graph.candidate_src, graph.candidate_dst, graph.candidate_support):
                node_src = graph.id_to_node[src.item()]
                node_dst = graph.id_to_node[dst.item()]
                sv_class, distance = tree.find_best_label(query=np.array([node_src.ref_pos, node_dst.ref_pos]),
                                                          chr1=node_src.chr_name, chr2=node_dst.chr_name, k=5,
                                                          orientation=NODE_TYPES_TO_ORIENTATION[(node_src.type, node_dst.type)])
                sv_class_matches.append(sv_class)
                edge_types = graph.get_edge_types(src, dst)
                label_summary.append(
                    f"Graph {i}, Edge ({node_src.chr_name}:{node_src.ref_pos}-{node_dst.chr_name}:{node_dst.ref_pos}): "
                    f"Label {sv_class}, Distance {distance}, Support {support}, Types {edge_types}"
                )
            labels = [self.config.class2label[c] for c in sv_class_matches]

            self.labels.append(torch.tensor(labels, dtype=torch.int64))
            label_summaries.append(label_summary)
            all_sv_class_matches.append(sv_class_matches)

            if config.graph_visualization_interval > 0 and i % config.graph_visualization_interval == 0:
                visualize_groovi_graph(self.config, graph, os.path.join(config.img_dir, f'graph_{self.dataset_tag}_{i}.html'),
                                       graph.candidate_src, graph.candidate_dst, sv_class_matches)
        self.stats.write_stats(self.graphs, self.dataset_tag, all_sv_class_matches, label_summaries)

    def __getitem__(self, index):
        graph = self.graphs[index]
        return graph.dgl_graph, graph.candidate_src, graph.candidate_dst, self.labels[index]
