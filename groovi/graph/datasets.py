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
