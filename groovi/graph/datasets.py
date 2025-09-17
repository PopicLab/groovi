from groovi.cli.config import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.handler import GraphHandler
from groovi.graph.transforms import TRANSFORMS
from groovi.seq.vcf import VCF_KDTree
from groovi.seq.constants import *

import dgl
import dgl.transforms as T
import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Callable
import numpy as np

class StaticDataset(Dataset):
    def __init__(self, config):
        self.handler = GraphHandler(config)
        self.transforms = T.Compose([TRANSFORMS[transform_name] for transform_name in config.transforms])
        self.graphs: list[GrooviGraph] = sum([self.handler.load(path) for path in config.graph_paths], [])
        self.dgl_graphs: list[dgl.DGLGraph] = [self.transforms(graph.get_dgl_graph()) for graph in self.graphs]
        self.labels = []

        # graph labeling
        for groovi_graph, dgl_graph in zip(self.graphs, self.dgl_graphs):
            tree: VCF_KDTree = VCF_KDTree(groovi_graph.get_vcf_path())
            labels = []
            for src, dst in zip(*groovi_graph.get_candidate_edge_ids()):
                position1 = dgl_graph.ndata['position'][src]
                position2 = dgl_graph.ndata['position'][dst]
                chrom1 = CHR_INT_TO_STR[dgl_graph.ndata['chr'][src].item()]
                chrom2 = CHR_INT_TO_STR[dgl_graph.ndata['chr'][dst].item()]

                label = tree.query_top_label(chrom=chrom1, chrom_target=chrom2, query=np.array([position1, position2]), k=5)
                labels.append(label)

            labels = torch.tensor(labels, dtype=torch.long)
            self.labels.append(labels)

    def __getitem__(self, index):
        groovi_graph = self.graphs[index]
        dgl_graph = self.dgl_graphs[index]
        labels = self.labels[index]
        # TODO: what if transformation gets at the nodes? transform it here
        src, dst = groovi_graph.get_candidate_edge_ids()
        return dgl_graph, src, dst, labels
    
    def __len__(self):
        return len(self.graphs)

class StreamingDataset(IterableDataset):
    def __init__(self, config: Config, extraction_fn: Callable = None):
        super().__init__()
        self.config = config
        self.extraction_fn = extraction_fn
        self.handler = GraphHandler(self.config)

    def __iter__(self):
        if isinstance(self.config, GenerateConfig):
            graphs: list[GrooviGraph] = GrooviGraph.from_config(self.config)
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
