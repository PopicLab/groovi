from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingDataset
from groovi.graph.handler import GraphHandler

import logging


def main(config_yaml):
    logging.basicConfig(level=logging.INFO)
    # .... load config ....
    config = GenerateConfig(config_yaml)    
    # .... generate graphs ....
    handler = GraphHandler(config)
    dataset = StreamingDataset(config)
    graphs = [graph for graph in dataset]
    # .... save & generate stats for the entire dataset ....
    handler.add_graphs(graphs)
    handler.save()