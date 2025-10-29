from groovi.cli.config import EvaluateConfig
from groovi.graph.datasets import StaticDatasetEdgeClassification
from groovi.gnn.models import EdgeClassificationModel
from groovi.gnn.train_utils import (
    validate_one_epoch_edge_classification,
    edge_classification_collate_fn,
    log_validation_metrics,
)
from groovi.seq.constants import *
from groovi.utils.utils import seed_everything, limit_cores

from dgl.dataloading import GraphDataLoader
import logging
import torch


def main(config_yaml):
    # ------ Initialization ------
    config = EvaluateConfig(config_yaml)
    seed_everything(config.seed)
    limit_cores(config.num_cores)

    logging.info(f"Using seed {config.seed}")
    logging.info(f"Logging to {config.log_file}")
    logging.info(f"Using config from {config_yaml}, \n {config}")

    # ---------Data setup--------
    dataset = StaticDatasetEdgeClassification(config)
    data_loader = GraphDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=edge_classification_collate_fn,
    )

    num_classes = len(SV_CLASSES)
    unique_labels = list(range(num_classes))
    logging.info(f"Training on {len(dataset)}")

    # ---------Model Setup--------
    model = EdgeClassificationModel(
        n_in_dim=len(NodeFeature),
        h_dim=config.hidden_dim,
        ht_emb_dim=config.ht_emb_dim,
        num_classes=num_classes,
        num_rels=len(EdgeFeature),
        update=config.update,
        num_hidden_layers=config.num_hidden_layers,
        normalize=config.normalize,
    )

    model.load_state_dict(
        torch.load(config.model_path, map_location=torch.device(config.device))
    )
    model = model.to(device=config.device)

    # ---------Evaluation Setup--------
    loss_fn = torch.nn.CrossEntropyLoss()

    # ------ Evaluation process ------
    all_metrics, _ = validate_one_epoch_edge_classification(
        model, data_loader, loss_fn, device=config.device, unique_labels=unique_labels
    )  # sanity check before training

    logging.info("===Training complete: Summary Statistics===")
    log_validation_metrics("validation", None, all_metrics, None)
