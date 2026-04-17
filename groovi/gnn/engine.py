from groovi.seq.sv import *
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *
import torch
from torch.utils.tensorboard import SummaryWriter

###########################
# Training Utilities
###########################
class MetricTracker:
    def __init__(self, config):
        self.config = config
        self.val_tags = sorted(config.graphs_validation.keys())

        self.train_writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, "Train"))
        self.val_writers = [SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, f'Validation_{tag}')) for tag in self.val_tags]

    def log_prf1_by_class(self, epoch, metrics, val_idx=None):
        logging.info(f'======= {epoch}  | Precision | Recall | F1 ')
        precision = metrics["Precision-by-class"]
        recall = metrics["Recall-by-class"]
        f1 = metrics["F1-by-class"]

        # if val_idx is none, write to training
        writer = self.val_writers[val_idx] if val_idx is not None else self.train_writer
        for i, class_name in enumerate(self.config.classes):
            logging.info(f'======= {class_name}  | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} ')
            writer.add_scalar(f"Validation | {class_name} | Precision", precision[i], epoch)
            writer.add_scalar(f"Validation | {class_name} | Recall", recall[i], epoch)
            writer.add_scalar(f"Validation | {class_name} | F1", f1[i], epoch)

    def log_train_loss(self, epoch, training_loss, metrics):
        logging.info(f'======= Train    | Epoch {epoch} | {training_loss:.4f}')
        self.train_writer.add_scalar("Train | AvgLoss", training_loss, epoch)
        self.log_prf1_by_class(epoch, metrics, val_idx=None)

    def log_validation_metrics(self, epoch, validation_loss, metrics, val_idx):
        logging.info(f"******* Validate | Epoch {epoch} | {validation_loss:.4f}")
        
        val_writer = self.val_writers[val_idx]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        f1 = metrics["F1"]

        val_writer.add_scalar(f"Validation | Precision", precision, epoch)
        val_writer.add_scalar(f"Validation | Recall", recall, epoch)
        val_writer.add_scalar(f"Validation | F1", f1, epoch)
        val_writer.add_scalar(f"Validation | AvgLoss", validation_loss, epoch)

        self.log_prf1_by_class(epoch, metrics, val_idx)

        cm = ConfusionMatrixDisplay(confusion_matrix=metrics["ConfusionMatrix"],
                                    display_labels=[c.name for c in self.config.classes])
        cm.plot(cmap='Blues', values_format='d')
        for text in cm.text_.ravel(): text.set_fontsize(6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, f'confusion_matrix_{self.val_tags[val_idx]}_epoch{epoch}.png'))
        plt.close()
        
######################
# Edge Classification
######################

def train(epoch, config, model, dataloader, optimizer, metric_tracker):
    model.train()
    total_loss = 0
    labels, predicted_classes = [], []
    for graph, src, dst, label in dataloader:
        graph, src, dst, label = (graph.to(config.device), src.to(config.device),
                                   dst.to(config.device), label.to(config.device))
        optimizer.zero_grad()
        logits = model(graph, src, dst).permute(0, 2, 1)
        loss = torch.nn.CrossEntropyLoss(weight=config.label_weights.to(config.device))(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted_classes.append(torch.argmax(logits, dim=1).flatten())
        labels.append(label.flatten())
    metrics = compute_metrics(config, labels, predicted_classes)
    metric_tracker.log_train_loss(epoch, total_loss / len(dataloader), metrics)

def validate(epoch, config, model, dataloaders, metric_tracker):
    model.eval()
    for idx, dataloader in enumerate(dataloaders):
        total_loss = 0
        labels, predicted_classes = [], []
        with torch.no_grad():
            for graph, src, dst, label in dataloader:
                graph, src, dst, label = (graph.to(config.device), src.to(config.device),
                                        dst.to(config.device), label.to(config.device))
                logits = model(graph, src, dst).permute(0, 2, 1)
                predicted_classes.append(torch.argmax(logits, dim=1).flatten())
                total_loss += torch.nn.CrossEntropyLoss(weight=config.label_weights.to(config.device))(logits, label).item()
                labels.append(label.flatten())
        metrics = compute_metrics(config, labels, predicted_classes)
        metric_tracker.log_validation_metrics(epoch, total_loss/ len(dataloader), metrics, idx)

def compute_metrics(config, labels, predicted_classes):
    labels = torch.cat(labels).cpu().flatten()
    predicted_classes = torch.cat(predicted_classes).cpu().flatten()
    class_labels = list(range(config.n_classes))
    metrics = {
        "Accuracy": (predicted_classes == labels).float().mean().item(),
        "Precision": precision_score(labels, predicted_classes, average="micro", zero_division=0),
        "Recall": recall_score(labels, predicted_classes, average="micro", zero_division=0),
        "F1": f1_score(labels, predicted_classes, average="micro", zero_division=0),
        "ConfusionMatrix": confusion_matrix(labels, predicted_classes, labels=class_labels),
        "Precision-by-class": precision_score(labels, predicted_classes, average=None, zero_division=0,
                                              labels=class_labels),
        "Recall-by-class": recall_score(labels, predicted_classes, average=None, zero_division=0, labels=class_labels),
        "F1-by-class": f1_score(labels, predicted_classes, average=None, zero_division=0, labels=class_labels),
    }
    return metrics


def evaluate(config, model, graph):
    model.eval()
    classified_edges = []
    with torch.no_grad():
        dgl_graph, src, dst, support = (graph.dgl_graph.to(config.device),
                                        graph.candidate_src.to(config.device),
                                        graph.candidate_dst.to(config.device),
                                        graph.candidate_support.to(config.device))
        logits = model(dgl_graph, src, dst)
        probs = torch.softmax(logits, dim=-1)
        max_probs, predicted_classes = torch.max(probs, dim=-1)
        for u, v, count, predicted_class, prob in zip(src, dst, support, predicted_classes, max_probs):
            node_src = graph.id_to_node[u.item()]
            node_dst = graph.id_to_node[v.item()]
            sv_type = config.classes[predicted_class.item()]
            if torch.isnan(prob):
                logging.warning(f": NaN at {node_src.chr_name}:{node_src.ref_pos}-{node_dst.chr_name}:{node_dst.ref_pos}")
                qual = 0
            else:
                qual = int(float(prob.item() * 100))
            classified_edges.append(SVBreakpoint(sv_type, node_src.chr_name, node_src.ref_pos,
                                                 node_dst.chr_name, node_dst.ref_pos, qual,(None, None), count.item()))
    return classified_edges
