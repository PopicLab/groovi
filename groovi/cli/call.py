# Copyright (c) 2024-2026 Popic Lab
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import groovi
from groovi.cli.config import DiscoverConfig
from groovi.graph.datasets import StreamingGraphDataset
from groovi.gnn.models import EdgeClassifier
from groovi.gnn.engine import evaluate
import groovi.seq.io as io
from groovi.seq.stitch_rules import assemble_csvs
from groovi.seq.sv import JunctionType
from groovi.seq.filters import filter_csvs
from collections import Counter
import logging
import torch

def main(config_yaml):
    print("*********************************")
    print("* groovi (%s): CSV discovery *" % groovi.__version__)
    print("*********************************")

    config = DiscoverConfig(config_yaml)
    if config.use_cached_predictions:
        logging.info("Loading cached predictions...")
        sv_junctions = [sv for sv in io.vcf_iter(config.junction_classified_file)]
    else:
        # --------- 1. Model Setup--------
        model = EdgeClassifier(config)
        model.load_state_dict(torch.load(config.model, map_location=config.device))
        model = model.to(device=config.device)

        # ------ 2. Junction classification ------
        logging.info("Running model prediction...")
        graph = next(iter(StreamingGraphDataset(config)))
        classified_adj = evaluate(config, model, graph)

        # ----- 3. Output SV junctions ------
        sv_junctions = [sv for sv in classified_adj if sv.type != JunctionType.NEG]
        logging.info(f"Classified {len(sv_junctions)} SV junctions")
        io.write_junction_vcf(config.junction_classified_file, sv_junctions, config.fai_index)

    # ------ 4. Stitch CSVs ------
    csvs = assemble_csvs(sv_junctions, tolerance=config.breakpoint_threshold)
    logging.info(f'Assembled {len(csvs)} SV candidates')
    io.write_csv_vcf(csvs, config.csv_candidates_file, config.fai_index, split_records=config.split_csv_format)

    # ------ 5. Filter and report CSVs ------
    csvs = filter_csvs(csvs, config)
    logging.info(f"Final SV classes: {dict(sorted(Counter(sv.sv_type for sv in csvs).items()))}")
    io.write_csv_vcf(csvs, config.csv_final_vcf_file, config.fai_index, split_records=config.split_csv_format)
    logging.info(f'Wrote {len(csvs)} assembled SVs to: {config.csv_final_vcf_file}')