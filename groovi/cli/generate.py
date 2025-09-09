from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingDataset
from groovi.graph.handler import GraphHandler

import logging

def generate(config):
    # .... load config ....
    # .... generate graphs ....
    handler = GraphHandler(config)
    dataset = StreamingDataset(config)
    graphs = [graph for graph in dataset]
    # .... save & generate stats for the entire dataset ....
    logging.info(f"Saving {len(graphs)} graphs.")
    handler.add_graphs(graphs)
    handler.save()


def main(config_yaml):
    config = GenerateConfig(config_yaml)
    generate(config)