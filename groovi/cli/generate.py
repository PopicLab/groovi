from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingDataset
from groovi.graph.handler import GraphHandler

import logging

def main(config_yaml):
    # .... load config ....
    config = GenerateConfig(config_yaml)    
    # .... generate graphs ....
    handler = GraphHandler(config)
    dataset = StreamingDataset(config)
    graphs = [graph for graph in dataset]
    # .... save & generate stats for the entire dataset ....
    logging.info(f"Saving {len(graphs)} graphs.")
    handler.add_graphs(graphs)
    handler.save()