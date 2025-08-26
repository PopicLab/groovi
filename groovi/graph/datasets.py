import torch
import dgl
from torch.utils.data import IterableDataset, Dataset
import dgl.transforms as T
from groovi.graph.graphs import GrooviGraph
from typing import Callable, Optional

from groovi.cli.config import *
from groovi.graph.construction import GraphConstructor
from groovi.graph.handler import GraphHandler

class StreamingDataset(IterableDataset):
    def __init__(self, config: Config, extraction_fn: Callable = None):
        super().__init__()
        self.config = config
        self.extraction_fn = extraction_fn

    def __iter__(self):
        if isinstance(self.config, GenerateConfig):
            graph_constructor = GraphConstructor(self.config)
            graphs: list[GrooviGraph] = graph_constructor.generate_groovi_graphs()
            yield from self._graph_generator(graphs)
                
        elif isinstance(self.config, CallConfig) or isinstance(self.config, InferenceMotifConfig):
            handler = GraphHandler(self.config)
            for graph_path in self.config.graph_paths:
                loaded_graphs = handler.load(graph_path, save=False)
                yield from self._graph_generator(loaded_graphs)
    
        # ... add support for other stremaing types types [URLs, multiple BAMs, etc.] ...

    def _graph_generator(self, graphs: list[GrooviGraph]):
        if self.extraction_fn:
            for graph in graphs:
                for subgraph in self.extraction_fn(graph):
                    yield subgraph
        else:
            for graph in graphs:
                yield graph 
