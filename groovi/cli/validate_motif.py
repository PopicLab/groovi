from groovi.cli.config import TrainMotifConfig

import logging



def main(config_yaml):
    logging.basicConfig(level=logging.INFO)
    # ------ Initialization ------
    # load the training configs
    config = TrainMotifConfig(config_yaml)
    # ---------Data setup--------
    # Label and Extract Subgraphs
    # Subsample Subgraphs
    # Initialize Dataset and DataLoaders
    # ---------Model--------
    # initialize/load pretrained model
    # ------ Training process ------
    pass