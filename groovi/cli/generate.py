from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingGraphDataset
import logging
import torch

torch.set_num_threads(1)

def main(config_yaml):
    config = GenerateConfig(config_yaml)
    dataset = StreamingGraphDataset(config)
    for graph in dataset:
        pass
