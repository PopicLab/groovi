from groovi.cli.config import CallConfig
from groovi.graph.datasets import StreamingDataset
from groovi.gnn.models import EdgeClassificationModel
from groovi.gnn.train_utils import predict_classes, inference_collate_fn, log_validation_metrics
from groovi.seq.constants import *
from groovi.utils.utils import seed_everything, limit_cores

from collections import namedtuple
from dgl.dataloading import GraphDataLoader
import logging
import torch

def main(config_yaml):
    # ------ Initialization ------
    config = CallConfig(config_yaml)
    seed_everything(config.seed)
    limit_cores(config.num_cores)

    logging.info(f"Using seed {config.seed}")
    logging.info(f"Logging to {config.log_file}")
    logging.info(f'Using config from {config_yaml}, \n {config}')

    # ---------Data setup--------
    dataset = StreamingDataset(config)
    data_loader = GraphDataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=inference_collate_fn)

    num_classes = len(SV_CLASSES)

    # ---------Model Setup--------
    model = EdgeClassificationModel(
        n_in_dim=len(NodeFeature),
        h_dim=config.hidden_dim,
        ht_emb_dim=config.ht_emb_dim,
        num_classes=num_classes,
        num_rels=len(EdgeFeature),
        update=config.update,
        num_hidden_layers=config.num_hidden_layers,
        normalize=config.normalize
    )
    
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device(config.device)))
    model = model.to(device=config.device)

    # ------ Get predicted classes ------
    predicted_class_ids = predict_classes(model, data_loader, device=config.device, top_k=None)  
    predicted_class_names = [SV_CLASSES[class_id] for class_id in predicted_class_ids]

    # ------ Flatten graph information, align with predictions ---------
    flattened_chr_starts = []
    flattened_chr_ends = []
    flattened_pos_starts = []
    flattened_pos_ends = []
    for graph in dataset:
        start_chrs, end_chrs, pos_starts, pos_ends = graph.decompose()
        flattened_chr_starts.extend(start_chrs)
        flattened_chr_ends.extend(end_chrs)
        flattened_pos_starts.extend(pos_starts)
        flattened_pos_ends.extend(pos_ends)
    
    out = [item for item in zip(flattened_chr_starts, flattened_chr_ends, flattened_pos_starts, flattened_pos_ends, predicted_class_names)]
    
    for line in out:
        print(line)

    # --- todo: stitching ---
        




        
        