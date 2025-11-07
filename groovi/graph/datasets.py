from groovi.cli.config import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.handler import GraphHandler
from groovi.graph.transforms import TRANSFORMS
from groovi.graph.visualize import visualize_groovi_graph
from groovi.seq.labeling import VCF_KDTree
from groovi.seq.io import GenomeInfo
from groovi.graph.constants import *

import pickle
import dgl
import dgl.transforms as T
import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np


class StreamingGraphDataset(IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.handler = GraphHandler(self.config)

    def save(self, graph) -> None:
        graph_file_path = os.path.join(self.config.graph_dir, f"graph_full.pkl")
        with open(graph_file_path, 'wb') as graph_file: pickle.dump(graph, graph_file)
        with open(self.config.stats_file_path, 'w') as stats_file:
            stats_file.write('\n'.join(f"{k}: {v}" for k, v in graph.get_stats().items()))

    def __iter__(self):
        graph = GrooviGraph.build(self.config)
        if self.config.save_graphs: self.save(graph)
        yield graph


class StaticDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.handler = GraphHandler(config)
        self.graphs: list[GrooviGraph] = []
        self.labels = []
        for path in config.graph_paths:
            self.graphs.append(self.load(path))
        self.transforms = T.Compose([TRANSFORMS[transform_name] for transform_name in config.transforms])
        for graph in self.graphs:
            print(graph)
            print(graph.id_to_node)
        self.graphs = [graph.apply_transforms(self.transforms) for graph in self.graphs]

        # optionally, post transformation visualization
        # if config.graph_visualize_every > 0:
        #     for i, graph in enumerate(self.graphs):
        #         if i % config.graph_visualize_every == 0:
        #             save_path = os.path.join(config.img_dir, f"graph_vis_{i}.html")
        #             visualize_groovi_graph(
        #                 graph,
        #                 path=save_path,
        #                 node_feature_color_code=NodeFeature.RD,
        #                 stagger_edges=True,
        #             )

    def load(self, path: str) -> GrooviGraph:
        with open(path, 'rb') as f: return pickle.load(f)

    # def __getitem__(self, index):
    #     pass

    def __len__(self):
        return len(self.graphs)


class StaticEdgeDataset(StaticDataset):
    def __init__(self, config):
        super().__init__(config)
        # perform edge labeling
        label_summary = []
        for i, graph in enumerate(self.graphs):
            tree: VCF_KDTree = VCF_KDTree(graph.vcf_path)
            labels = []
            for src, dst in zip(graph.candidate_src, graph.candidate_dst):
                node_src = graph.id_to_node[src.item()]
                node_dst = graph.id_to_node[dst.item()]
                label, distance = tree.query_top_label(query=np.array([node_src.ref_pos, node_dst.ref_pos]),
                                                       start_chrom=node_src.chr_name,
                                                       end_chrom=node_dst.chr_name,
                                                       k=5, return_distance=True,
                                                       orientation=NODE_TYPES_TO_ORIENTATION[(node_src.type, node_dst.type)])
                labels.append(label)
                label_summary.append(
                    f"Graph {i}, Edge ({node_src.chr_name}:{node_src.ref_pos}-{node_dst.chr_name}:{node_dst.ref_pos}): "
                    f"Label {label}, Distance {distance}"
                )
            labels = [SV_CLASS_TO_ID_FN(label) for label in labels]
            labels = torch.tensor(labels, dtype=torch.long)
            self.labels.append(labels)
        with open(config.stats_file_path, "w") as f:
            f.write("\n".join(label_summary))
    def __getitem__(self, index):
        graph = self.graphs[index]
        return graph.dgl_graph, graph.candidate_src, graph.candidate_dst, self.labels[index]


class StaticGraphDataset(StaticDataset):
    def __init__(self, config):
        super().__init__(config)

        label_key = config.label_key
        label_to_int = {}
        labels = []
        self.num_classes = None

        for graph in self.graphs:
            genome_info = GenomeInfo(graph.get_info_path())
            label_value = genome_info.get_label(label_key)

            if self.num_classes is None:
                self.num_classes = genome_info.config_data[label_key]["num_classes"]
            else:
                assert (
                    self.num_classes
                    == genome_info.config_data[label_key]["num_classes"]
                ), f"Number of classes for label '{label_key}' inconsistent across graphs."

            if label_value not in label_to_int:
                label_to_int[label_value] = len(label_to_int)
            labels.append(label_to_int[label_value])

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        groovi_graph = self.graphs[index]
        dgl_graph = groovi_graph.get_dgl_graph()
        label = self.labels[index]
        return dgl_graph, label

    def get_num_classes(self):
        return self.num_classes

