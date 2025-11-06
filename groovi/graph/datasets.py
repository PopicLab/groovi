from groovi.cli.config import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.handler import GraphHandler
from groovi.graph.transforms import TRANSFORMS
from groovi.graph.visualize import visualize_groovi_graph
from groovi.seq.labeling import VCF_KDTree
from groovi.seq.io import GenomeInfo
from groovi.seq.constants import *

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
        self.graphs: list[GrooviGraph] = sum(
            [self.handler.load(path) for path in config.graph_paths], []
        )

        # apply transforms
        self.transforms = T.Compose(
            [TRANSFORMS[transform_name] for transform_name in config.transforms]
        )
        self.graphs = [
            graph.apply_transforms(self.transforms) or graph for graph in self.graphs
        ]

        # optionally, post transformation visualization
        if config.graph_visualize_every > 0:
            logging.info(
                f"Visualizing every {config.graph_visualize_every} graphs to {config.img_dir}"
            )
            for i, graph in enumerate(self.graphs):
                if i % config.graph_visualize_every == 0:
                    save_path = os.path.join(config.img_dir, f"graph_vis_{i}.html")
                    visualize_groovi_graph(
                        graph,
                        path=save_path,
                        node_feature_color_code=NodeFeature.RD,
                        stagger_edges=True,
                    )

    def __getitem__(self, index):
        raise NotImplementedError(
            "Please use StaticDatasetEdgeClassification or StaticDatasetGraphClassification."
        )

    def __len__(self):
        return len(self.graphs)


class StaticDatasetEdgeClassification(StaticDataset):
    def __init__(self, config):
        super().__init__(config)

        # edge labeling
        self.labels = []
        self.distances = []
        label_summary = ["Edge Labeling Summary:"]

        for i, graph in enumerate(self.graphs):
            tree: VCF_KDTree = VCF_KDTree(graph.get_vcf_path())
            labels = []
            distances = []
            for src, dst in zip(*graph.get_candidate_edges()):
                # obtain relevant info for querying: position, HT, chrom
                position1, position2 = (
                    graph.ndata["position"][src],
                    graph.ndata["position"][dst],
                )
                laterality = NODE_TYPES_TO_ORIENTATION(src.item(), dst.item())
                chrom1, chrom2 = graph.get_chr_names_by_node_ids(
                    [graph.ndata["chr"][src].item(), graph.ndata["chr"][dst].item()]
                )

                label, distance = tree.query_top_label(
                    query=np.array([position1, position2]),
                    start_chrom=chrom1,
                    end_chrom=chrom2,
                    k=5,
                    return_distance=True,
                    split_alignmnet_orientation=laterality,
                )
                labels.append(label)
                label_summary.append(
                    f"Graph {i}, Edge ({chrom1}:{position1}-{chrom2}:{position2}): Label {label}, Distance {distance}"
                )

            labels = [SV_CLASS_TO_ID_FN(label) for label in labels]
            labels = torch.tensor(labels, dtype=torch.long)
            self.labels.append(labels)
            self.distances.append(distances)

        with open(config.stats_path, "w") as f:
            f.write("\n".join(label_summary))

    def __getitem__(self, index):
        groovi_graph: GrooviGraph = self.graphs[index]
        dgl_graph = groovi_graph.get_dgl_graph()
        labels = self.labels[index]
        src, dst = groovi_graph.get_candidate_edges()
        return dgl_graph, src, dst, labels


class StaticDatasetGraphClassification(StaticDataset):
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

