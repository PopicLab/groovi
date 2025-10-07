from groovi.cli.config import TrainGraphConfig
from groovi.graph.datasets import StaticDatasetGraphClassification
from groovi.gnn.models import GraphClassificationModel
from groovi.gnn.train_utils import train_and_validate_graph_classification, log_validation_metrics
from groovi.seq.constants import *
from groovi.utils.utils import seed_everything, limit_cores

from dgl.dataloading import GraphDataLoader
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import random_split

def main(config_yaml):
    # ------ Initialization ------
    config = TrainGraphConfig(config_yaml)
    seed_everything(config.seed)
    limit_cores(config.num_cores)

    logging.info(f"Using seed {config.seed}")
    logging.info(f"Logging to {config.log_file}")
    logging.info(f'Using config from {config_yaml}, \n {config}')

    # ---------Data setup--------
    dataset = StaticDatasetGraphClassification(config)
    val_size = int(config.validation_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    num_classes = dataset.get_num_classes()
    unique_labels = list(range(num_classes))
    logging.info(f"Training on {len(train_dataset)} graphs, validating on {len(val_dataset)} graphs.")

    # ---------Model Setup--------
    model = GraphClassificationModel(
        n_in_dim=len(NodeFeature),
        h_dim=config.hidden_dim,
        num_classes=dataset.get_num_classes(),
        ht_emb_dim=config.ht_emb_dim,
        num_rels=len(EdgeFeature),
        update=config.update,
        pooling=config.pooling,
        num_hidden_layers=config.num_hidden_layers,
        normalize=config.normalize
    )

    # ---------Training Setup--------
    tensorboard_writer = SummaryWriter(log_dir=config.log_dir)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ------ Training process ------
    model, metrics = train_and_validate_graph_classification(model, train_dataloader, val_dataloader, optimizer, loss_fn, config.device, config.num_epochs, config.validation_interval, tensorboard_writer, config.model_save_dir, unique_labels=unique_labels)
    torch.save(model.state_dict(), os.path.join(config.model_save_dir, f'model_graph_classification_epoch_{config.num_epochs}.pth'))

    logging.info("===Training complete: Summary Statistics===")
    all_metrics, _ = metrics
    log_validation_metrics(config.num_epochs, None, all_metrics)