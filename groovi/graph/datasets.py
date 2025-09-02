from groovi.cli.config import *
from groovi.graph.graphs import GrooviGraph
from groovi.graph.handler import GraphHandler

from torch.utils.data import IterableDataset
from typing import Callable


class StreamingDataset(IterableDataset):
    def __init__(self, config: Config, extraction_fn: Callable = None):
        super().__init__()
        self.config = config
        self.extraction_fn = extraction_fn

    def __iter__(self):
        if isinstance(self.config, GenerateConfig):
            graphs: list[GrooviGraph] = GrooviGraph.from_config(self.config)
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
