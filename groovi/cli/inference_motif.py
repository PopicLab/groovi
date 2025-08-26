from groovi.cli.config import InferenceMotifConfig
from groovi.graph.datasets import StreamingDataset

import logging


def main(config_yaml):
    # ------ Initialization ------
    # load the training configs
    logging.basicConfig(level=logging.INFO)
    config = InferenceMotifConfig(config_yaml)
    streaming_dataset = StreamingDataset(config)
    # ---------Data setup--------
    for graph in streaming_dataset:
        logging.info(graph)
    
    # ---------Model--------
    # initialize/load pretrained model
    # ------ Training process ------
    pass