from groovi.cli.config import TrainEdgeConfig
from groovi.graph.datasets import StaticEdgeDataset
from groovi.gnn.models import EdgeClassificationModel
from groovi.gnn.engine import *

from dgl.dataloading import GraphDataLoader
import logging
import torch
from torch.utils.data import random_split


def main(config_yaml):
    config = TrainEdgeConfig(config_yaml)

    # ---------Training dataset setup--------
    train_dataset = StaticEdgeDataset(config)
    val_dataset = StaticEdgeDataset(config, is_validation=True)
    #val_size = int(config.validation_ratio * len(dataset))
    #train_size = len(dataset) - val_size
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --------- Data loaders -----------------
    train_dataloader = GraphDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    logging.info("Size of train set: %d; validation set: %d" % (len(train_dataloader), len(val_dataloader)))

    # ---------Model Setup--------
    model = EdgeClassificationModel(config)
    if config.pretrained_model: model.load_state_dict(torch.load(config.pretrained_model, config.device))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.learning_rate_decay_interval,
                                                   gamma=config.learning_rate_decay_factor)
    model = model.to(device=config.device)

    # ------ Training process ------
    for epoch in range(config.num_epochs):
        train(epoch, config, model, train_dataloader, optimizer)
        # check model checkpoint interval
        torch.save(model.state_dict(), "%s/model.epoch%d.pt" % (config.model_checkpoints_dir, epoch))
        validate(epoch, config, model, val_dataloader)
        lr_scheduler.step()
    torch.save(model.state_dict(), config.model_path)

