from groovi.cli.config import DiscoverConfig
from groovi.graph.datasets import StreamingGraphDataset
from groovi.gnn.models import EdgeClassificationModel
from groovi.gnn.engine import *
import groovi.seq.io as io
from groovi.seq.stitching import generate_merged_callset
import logging
import torch

def main(config_yaml):
    config = DiscoverConfig(config_yaml)

    # --------- 1. Model Setup--------
    model = EdgeClassificationModel(config)
    model.load_state_dict(torch.load(config.model, map_location=config.device))
    model = model.to(device=config.device)

    # ------ 2. Breakpoint classification ------
    logging.info("Running model prediction...")
    classified_sv_bkps = evaluate(config, model, next(iter(StreamingGraphDataset(config))))
    logging.info("Writing breakpoint classification results...")
    io.write_bkp_vcf(config.bkp_vcf_file_path, classified_sv_bkps, config.fai_index)

    # ------ 3. Breakpoint stitching/filtering ------
    generate_merged_callset(classified_sv_bkps, config)