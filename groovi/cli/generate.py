from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingGraphDataset
from groovi.graph.handler import GraphHandler

import logging


def generate(config):
    # .... generate graphs ....
    dataset = StreamingGraphDataset(config)
    graphs = [graph for graph in dataset]  # graphs are saved internally after iteration
    # .... save & generate stats for the entire dataset ....


def main(config_yaml):
    config = GenerateConfig(config_yaml)
    generate(config)
