from groovi.cli.config import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.handler import GraphHandler
from groovi.graph.transforms import TRANSFORMS
from groovi.graph.visualize import visualize_groovi_graph
from groovi.seq.labeling import VCF_KDTree
from groovi.seq.io import GenomeInfo
from groovi.seq.constants import *

import dgl
import dgl.transforms as T
import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Callable
import numpy as np

'''
Static Datasets
'''
class StaticDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        # initialize
        self.config = config
        self.handler = GraphHandler(config)
        self.graphs: list[GrooviGraph] = sum([self.handler.load(path) for path in config.graph_paths], [])

        # apply transforms
        self.transforms = T.Compose([TRANSFORMS[transform_name] for transform_name in config.transforms])
        self.graphs = [graph.apply_transforms(self.transforms) or graph for graph in self.graphs]

        # optionally, post transformation visualization
        if config.graph_visualize_every > 0:
            logging.info(f"Visualizing every {config.graph_visualize_every} graphs to {config.img_dir}")
            for i, graph in enumerate(self.graphs):
                if i % config.graph_visualize_every == 0:
                    save_path = os.path.join(config.img_dir, f'graph_vis_{i}.html')
                    visualize_groovi_graph(graph, path=save_path, node_feature_color_code=NodeFeature.RD, stagger_edges=True)

    def __getitem__(self, index):
        raise NotImplementedError("Please use StaticDatasetEdgeClassification or StaticDatasetGraphClassification.")
        
    def __len__(self):
        return len(self.graphs)

class StaticDatasetEdgeClassification(StaticDataset):
    def __init__(self, config):
        super().__init__(config)

        # edge labeling
        self.labels = []
        for graph in self.graphs:
            tree: VCF_KDTree = VCF_KDTree(graph.get_vcf_path())
            labels = []
            for src, dst in zip(*graph.get_candidate_edge_ids()):
                position1, position2 = graph.ndata['position'][src], graph.ndata['position'][dst]
                chrom1, chrom2 = graph.get_chr_names_by_node_ids([graph.ndata['chr'][src].item(), graph.ndata['chr'][dst].item()])
                label = tree.query_top_label(chrom=chrom1, chrom_target=chrom2, query=np.array([position1, position2]), k=5)
                labels.append(label)

            labels = torch.tensor(labels, dtype=torch.long)
            self.labels.append(labels)
    
    def __getitem__(self, index):
        groovi_graph = self.graphs[index]
        dgl_graph = groovi_graph.get_dgl_graph()
        labels = self.labels[index]
        src, dst = groovi_graph.get_candidate_edge_ids()
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
                self.num_classes = genome_info.config_data[label_key]['num_classes']
            else:
                assert self.num_classes == genome_info.config_data[label_key]['num_classes'], f"Number of classes for label '{label_key}' inconsistent across graphs."

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
    
'''
Streaming Dataset(s)
'''
class StreamingDataset(IterableDataset):
    def __init__(self, config: Config, extraction_fn: Callable = None):
        super().__init__()
        self.config = config
        self.extraction_fn = extraction_fn
        self.handler = GraphHandler(self.config)

    def __iter__(self):
        if isinstance(self.config, GenerateConfig):
            graphs: list[GrooviGraph] = GrooviGraph.from_config(self.config)
            if self.config.split_by_connected_components:
                graphs = sum([GrooviGraph._split_by_components(graph) for graph in graphs], [])
            yield from self._graph_generator(graphs)
            logging.info(f"Generated and saving {len(graphs)} graphs.")
            self.handler.add_graphs(graphs)
            self.handler.save()
                
        elif isinstance(self.config, CallConfig) or isinstance(self.config, InferenceMotifConfig):
            # if cache_graphs, save the generated graphs
            for graph_path in self.config.graph_paths:
                loaded_graphs = self.handler.load(graph_path, store=self.config.cache_graphs)
                yield from self._graph_generator(loaded_graphs)
            if self.config.cache_graphs: self.handler.save()

        # ... add support for other stremaing types types [URLs, multiple BAMs, etc.] ...

    def _graph_generator(self, graphs: list[GrooviGraph]):
        if self.extraction_fn:
            for graph in graphs:
                for subgraph in self.extraction_fn(graph):
                    yield subgraph
        else:
            for graph in graphs:
                yield graph 
