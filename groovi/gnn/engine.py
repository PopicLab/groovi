from groovi.seq.sv import *
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *
import torch
# from torch.utils.tensorboard import SummaryWriter

###########################
# Training Utilities
###########################
class MetricTracker:
    def __init__(self, epoch, config):
        self.epoch = epoch
        self.config = config
        #self.tensorboard_writer = SummaryWriter(log_dir=config.tensorboard_dir)

    def log_prf1(self, metrics):
        precision = metrics["Precision-by-class"]
        recall = metrics["Recall-by-class"]
        f1 = metrics["F1-by-class"]
        for i, class_name in enumerate(self.config.classes):
            logging.info(f"{class_name}: {precision[i]:.4f}, {recall[i]:.4f}, {f1[i]:.4f}")

    def log_train_loss(self, training_loss, metrics):
        # self.tensorboard_writer.add_scalar("Train|AvgLoss", training_loss, self.epoch)
        logging.info(f'======= Train    | Epoch {self.epoch} | {training_loss:.4f}')
        self.log_prf1(metrics)

    def log_validation_metrics(self, metrics):
        # for key, value in metrics.items():
        #     self.tensorboard_writer.add_scalar(f"Validation|{key}", value, self.epoch)
        validation_loss = metrics["AvgLoss"]
        logging.info(f"******* Validate | Epoch {self.epoch} | {validation_loss:.4f}")
        self.log_prf1(metrics)

        cm = ConfusionMatrixDisplay(confusion_matrix=metrics["ConfusionMatrix"],
                                    display_labels=[c.name for c in self.config.classes])
        cm.plot(cmap='Blues', values_format='d')
        for text in cm.text_.ravel(): text.set_fontsize(6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, f'confusion_matrix_epoch{self.epoch}.png'))

######################
# Edge Classification
######################

def train(epoch, config, model, dataloader, optimizer):
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

    labels = torch.cat(labels).cpu()
    predicted_classes = torch.cat(predicted_classes).cpu()
    labels = labels.flatten()
    predicted_classes = predicted_classes.flatten()
    class_labels = list(range(config.n_classes))
    metrics = {
        "AvgLoss": total_loss / len(dataloader),
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
    MetricTracker(epoch, config).log_train_loss(total_loss / len(dataloader), metrics)

def validate(epoch, config, model, dataloader):
    model.eval()
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
    labels = torch.cat(labels).cpu()
    predicted_classes = torch.cat(predicted_classes).cpu()
    labels = labels.flatten()
    predicted_classes = predicted_classes.flatten()
    class_labels = list(range(config.n_classes))
    metrics = {
        "AvgLoss": total_loss / len(dataloader),
        "Accuracy": (predicted_classes == labels).float().mean().item(),
        "Precision": precision_score(labels, predicted_classes, average="micro", zero_division=0),
        "Recall": recall_score(labels, predicted_classes, average="micro", zero_division=0),
        "F1": f1_score(labels, predicted_classes, average="micro", zero_division=0),
        "ConfusionMatrix": confusion_matrix(labels, predicted_classes, labels=class_labels),
        "Precision-by-class": precision_score(labels, predicted_classes, average=None, zero_division=0, labels=class_labels),
        "Recall-by-class": recall_score(labels, predicted_classes, average=None, zero_division=0, labels=class_labels),
        "F1-by-class": f1_score(labels, predicted_classes, average=None, zero_division=0, labels=class_labels),
    }
    MetricTracker(epoch, config).log_validation_metrics(metrics)

def evaluate(config, model, graph):
    model.eval()
    classified_edges = []
    with torch.no_grad():
        dgl_graph, src, dst = (graph.dgl_graph.to(config.device),
                               graph.candidate_src.to(config.device),
                               graph.candidate_dst.to(config.device))
        logits = model(dgl_graph, src, dst)
        probs = torch.softmax(logits, dim=-1)
        max_probs, predicted_classes = torch.max(probs, dim=-1)
        for s, d, predicted_class, prob in zip(src, dst, predicted_classes, max_probs):
            node_src = graph.id_to_node[s.item()]
            node_dst = graph.id_to_node[d.item()]
            sv_type = config.classes[predicted_class.item()]
            if sv_type == SVBpType.NEG: continue
            support, _ = graph.get_edge_support(s.cpu(), d.cpu())
            qual = int(float(prob.item() * 100))
            classified_edges.append(SVBreakpoint(sv_type, node_src.chr_name, node_src.ref_pos,
                                                 node_dst.chr_name, node_dst.ref_pos, qual,(None, None), support))
    return classified_edges
