from groovi.cli.config import TrainEdgeConfig
from groovi.graph.datasets import StaticEdgeDataset
from groovi.gnn.models import *
from groovi.gnn.engine import *

from dgl.dataloading import GraphDataLoader
import logging
import torch
from torch.utils.data import random_split


def main(config_yaml):
    config = TrainEdgeConfig(config_yaml)

    # ---------Training dataset setup--------
    train_dataset = StaticEdgeDataset(config, graph_paths=config.graphs, graphs_info=config.graphs_info, tag="Train")
    val_datasets = [
        StaticEdgeDataset(
            config, 
            graph_paths=paths, 
            graphs_info=config.graphs_info_validation[key], 
            tag=key
        ) for key, paths in sorted(config.graphs_validation.items())
    ]

    # --------- Data loaders -----------------
    train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloaders = [GraphDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) for val_dataset in val_datasets]
    logging.info("Size of train set: %d; validation set: %d" % (len(train_dataloader), sum([len(val_dataloader) for val_dataloader in val_dataloaders])))

    # ---------Model Setup--------
    model = EdgeClassifier(config)
    if config.pretrained_model: model.load_state_dict(torch.load(config.pretrained_model, config.device))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.learning_rate_decay_interval,
                                                   gamma=config.learning_rate_decay_factor)
    model = model.to(device=config.device)

    # ------ Training process ------
    best_model = None
    best_loss = None
    best_epoch = None
    metric_tracker = MetricTracker(config)
    for epoch in range(config.num_epochs):
        train(epoch, config, model, train_dataloader, optimizer, metric_tracker)
        loss = validate(epoch, config, model, val_dataloaders, metric_tracker)
        lr_scheduler.step()
        if best_loss is None or loss < best_loss: 
            best_loss = loss
            best_model = model
            best_epoch = epoch
        if epoch % config.model_checkpoint_interval == 0:
            torch.save(model.state_dict(), "%s/model.epoch%d.pt" % (config.model_checkpoints_dir, epoch))
    if best_model:
        logging.info("Best model at epoch %d, loss = %f" % (best_epoch, best_loss))
        torch.save(best_model.state_dict(), config.model_path)


