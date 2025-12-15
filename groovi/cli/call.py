from groovi.cli.config import DiscoverConfig
from groovi.graph.datasets import StreamingGraphDataset
from groovi.graph.visualize import visualize_groovi_graph
from groovi.gnn.models import EdgeClassificationModel
from groovi.gnn.engine import *
import groovi.seq.io as io
from groovi.seq.stitching import generate_merged_callset
import logging
import torch

def main(config_yaml):
    config = DiscoverConfig(config_yaml)

    if config.use_cached_predictions:
        classified_sv_bkps = [sv for sv in io.vcf_iter(config.bkp_vcf_file_path)]
    else:
        # --------- 1. Model Setup--------
        model = EdgeClassificationModel(config)
        model.load_state_dict(torch.load(config.model, map_location=config.device))
        model = model.to(device=config.device)

        # ------ 2. Breakpoint classification ------
        logging.info("Running model prediction...")
        graph = next(iter(StreamingGraphDataset(config)))
        classified_sv_bkps = evaluate(config, model, graph)
        
        # ------ 3. Visualization ------
        visualize_groovi_graph(graph, os.path.join(config.img_dir, "graph.html"), graph.candidate_src, graph.candidate_dst, expected_labels=None, predicted_labels=[svbkp.type for svbkp in classified_sv_bkps])

        # ----- 4. Filter Neg, Write classified breakpoints to VCF ------
        classified_sv_bkps = [sv for sv in classified_sv_bkps if sv.type != SVBpType.NEG]
        logging.info("Writing breakpoint classification results...")
        io.write_bkp_vcf(config.bkp_vcf_file_path, classified_sv_bkps, config.fai_index)

    # ------ 3. Breakpoint stitching/filtering ------
    generate_merged_callset(classified_sv_bkps, config)