from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingDataset
from groovi.graph.handler import GraphHandler

import logging


def generate(config):
    # .... generate graphs ....
    dataset = StreamingDataset(config)
    graphs = [graph for graph in dataset]  # graphs are saved internally after iteration
    # .... save & generate stats for the entire dataset ....


def main(config_yaml):
    config = GenerateConfig(config_yaml)
    logging.info(f"Using config from {config_yaml}, \n {config}")
    generate(config)
