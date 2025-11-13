from groovi.graph.constants import *
from groovi.graph.graphs import GrooviGraph
from groovi.seq.sv import *
from groovi.graph.transforms import TRANSFORMS
from groovi.graph.visualize import visualize_groovi_graph

import dgl.transforms as T
import numpy as np
import os
import pickle
import torch
from torch.utils.data import IterableDataset, Dataset
import yaml

class StreamingGraphDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def save(self, graph) -> None:
        graph_file_path = os.path.join(self.config.graph_dir, f"graph_full.pkl")
        with open(graph_file_path, 'wb') as graph_file: pickle.dump(graph, graph_file)
        with open(self.config.stats_file_path, 'w') as stats_file:
            stats_file.write('\n'.join(f"{k}: {v}" for k, v in graph.get_stats().items()))

    def __iter__(self):
        graph = GrooviGraph.build(self.config)
        if self.config.visualize_graphs:
            visualize_groovi_graph(graph, os.path.join(self.config.img_dir, 'graph.html'))
        if self.config.save_graphs: self.save(graph)
        yield graph


class StaticDataset(Dataset):
    def __init__(self, config, is_validation=False):
        super().__init__()
        self.config = config
        self.graphs: list[GrooviGraph] = []
        self.labels = []
        for path in config.graphs if not is_validation else config.graphs_validation:
            self.graphs.append(self.load(path))
        self.transforms = T.Compose([TRANSFORMS[transform_name] for transform_name in config.transforms])
        for g in self.graphs:
            g.filter_candidate_edges(self.config.candidate_support_threshold)
            g.apply_transforms(self.transforms)

    def load(self, path: str) -> GrooviGraph:
        with open(path, 'rb') as f: return pickle.load(f)

    def __len__(self):
        return len(self.graphs)


class StaticEdgeDataset(StaticDataset):
    def __init__(self, config, is_validation=False):
        super().__init__(config, is_validation)
        # perform edge labeling
        label_summary = []
        for i, graph in enumerate(self.graphs):
            tree: SVKDTree = SVKDTree(graph.vcf_path, config.classes)
            sv_class_matches = []
            label_summary.append(f"Graph {i}: {graph.vcf_path}")
            for src, dst in zip(graph.candidate_src, graph.candidate_dst):
                node_src = graph.id_to_node[src.item()]
                node_dst = graph.id_to_node[dst.item()]
                sv_class, distance = tree.find_best_label(query=np.array([node_src.ref_pos, node_dst.ref_pos]),
                                                         chr1=node_src.chr_name, chr2=node_dst.chr_name, k=5,
                                                         orientation=NODE_TYPES_TO_ORIENTATION[(node_src.type, node_dst.type)])
                sv_class_matches.append(sv_class)
                label_summary.append(
                    f"Graph {i}, Edge ({node_src.chr_name}:{node_src.ref_pos}-{node_dst.chr_name}:{node_dst.ref_pos}): "
                    f"Label {sv_class}, Distance {distance}"
                )

            if i % config.graph_visualization_interval == 0:
                visualize_groovi_graph(graph, os.path.join(config.img_dir, f'graph_{i}.html'), graph.candidate_src, graph.candidate_dst, sv_class_matches)

            labels = [self.config.class2label[c] for c in sv_class_matches]
            self.labels.append(torch.tensor(labels, dtype=torch.int64))
        with open(config.stats_file_path, "w") as f:
            f.write("\n".join(label_summary))
    def __getitem__(self, index):
        graph = self.graphs[index]
        return graph.dgl_graph, graph.candidate_src, graph.candidate_dst, self.labels[index]


class StaticGraphDataset(StaticDataset):
    def __init__(self, config):
        super().__init__(config)
        labels = []
        for graph in self.graphs:
            with open(graph.info_path, 'r') as file:
                annotation = yaml.safe_load(file) # graph-level labels
                assert 'class' in annotation
                #labels.append(SV_GRAPH_CLASS_TO_ID_FN[annotation['class']]) # todo
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __getitem__(self, index):
        return self.graphs[index].dgl_graph, self.labels[index]
