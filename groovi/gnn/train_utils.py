# from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import *
import torch

###########################
# Training Utilities
###########################
class MetricTracker:
    def __init__(self, epoch, config):
        self.epoch = epoch
        #self.tensorboard_writer = SummaryWriter(log_dir=config.tensorboard_dir)

    def log_train_loss(self, training_loss):
        # self.tensorboard_writer.add_scalar("Train|AvgLoss", training_loss, self.epoch)
        logging.info(f'======= Train    | Epoch {self.epoch} | {training_loss}')

    def log_validation_metrics(self, metrics):
        # for key, value in metrics.items():
        #     self.tensorboard_writer.add_scalar(f"Validation|{key}", value, self.epoch)
        validation_loss = metrics["AvgLoss"]
        logging.info(f"******* Validate | Epoch {self.epoch} | {validation_loss}")
        for key, value in metrics.items():
            if key in ["ConfusionMatrix", "AvgLoss"]: continue
            logging.info(f"{key} = {value}")
        # TODO: plot confusion matrix

######################
# Edge Classification
######################

def train(epoch, config, model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for graph, src, dst, labels in dataloader:
        graph, src, dst, labels = (graph.to(config.device), src.to(config.device),
                                   dst.to(config.device), labels.to(config.device))
        optimizer.zero_grad()
        logits = model(graph, src, dst)
        loss = torch.nn.CrossEntropyLoss()(logits.permute(0, 2, 1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    MetricTracker(epoch, config).log_train_loss(total_loss / len(dataloader))

def validate(epoch, config, model, dataloader):
    model.eval()
    total_loss = 0
    labels, predicted = [], []
    with torch.no_grad():
        for graph, src, dst, label in dataloader:
            graph, src, dst, label = (graph.to(config.device), src.to(config.device),
                                      dst.to(config.device), label.to(config.device))
            logits = model(graph, src, dst)
            total_loss += torch.nn.CrossEntropyLoss()(logits.permute(0, 2, 1), label).item()
            labels.append(label)
            predicted.append(torch.argmax(logits, dim=-1))

    labels = torch.cat(labels).cpu()
    predicted = torch.cat(predicted).cpu()
    labels = labels.flatten()
    predicted = predicted.flatten()
    class_labels = list(range(config.n_classes))
    metrics = {
        "AvgLoss": total_loss / len(dataloader),
        "Accuracy": (predicted == labels).float().mean().item(),
        "Precision": precision_score(labels, predicted, average="micro", zero_division=0),
        "Recall": recall_score(labels, predicted, average="micro", zero_division=0),
        "F1": f1_score(labels, predicted, average="micro", zero_division=0),
        "ConfusionMatrix": confusion_matrix(labels, predicted, labels=class_labels),
        "Precision-by-class": precision_score(labels, predicted, average=None, zero_division=0, labels=class_labels),
        "Recall-by-class": recall_score(labels, predicted, average=None, zero_division=0, labels=class_labels),
        "F1-by-class": f1_score(labels, predicted, average=None, zero_division=0, labels=class_labels),
    }
    MetricTracker(epoch, config).log_validation_metrics(metrics)
    return predicted
