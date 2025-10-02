import dgl
import logging
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
from typing import Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

'''
Edge Classification Training Utilities
'''
def train_and_validate_edge_classification(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, num_epochs, validation_interval, tensorboard_writer, model_save_path, unique_labels) -> Tuple[torch.nn.Module, dict]:
    """
    Trains and validates a model with rich progress bars, including the current epoch number.
    """
    model = model.to(device)

    # Define the rich progress bar for the main training epochs
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "|",
        TimeRemainingColumn(),
        "|",
        TextColumn("[progress.description]{task.fields[loss]}"),
    ) as progress:
        # Create a task for the main epoch loop
        epoch_task = progress.add_task("[green]Training...", total=num_epochs, loss="N/A")

        train_loss, val_loss = 0.0, 0.0
        for epoch in range(num_epochs):
            # Validate
            if epoch % validation_interval == 0:
                all_metrics , tensorboard_metrics = validate_one_epoch_edge_classification(model, val_dataloader, loss_fn, device, unique_labels)

                log_validation_metrics(epoch, tensorboard_metrics, all_metrics, tensorboard_writer)

                torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))
                val_loss = all_metrics['AvgLoss']
                progress.update(epoch_task, loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")

            # Train
            train_loss = train_one_epoch_edge_classification(model, train_dataloader, optimizer, loss_fn, device)
            progress.update(epoch_task, 
                            advance=1, 
                            description=f"[green]Training Epoch {epoch + 1}/{num_epochs}[/green]",
                            loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")
            tensorboard_writer.add_scalar('Train/AvgLoss', train_loss, epoch)


    # ------ Final Steps ------
    all_metrics, tensorboard_metrics = validate_one_epoch_edge_classification(model, val_dataloader, loss_fn, device, unique_labels)
    for key, value in tensorboard_metrics.items():
        tensorboard_writer.add_scalar(f'Final/{key}', value, num_epochs)
    
    # TODO: use misc_metrics (e.g., confusion matrix) for visualization
    tensorboard_writer.close()
    return model, (all_metrics, tensorboard_metrics)

def train_one_epoch_edge_classification(model, dataloader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0
    for bg, bsrc, bdst, blabels in dataloader:
        bg, bsrc, bdst, blabels = bg.to(device), bsrc.to(device), bdst.to(device), blabels.to(device)
        optimizer.zero_grad()
        logits = model(bg, bsrc, bdst)
        loss = loss_fn(logits, blabels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch_edge_classification(model, dataloader, loss_fn, device, unique_labels) -> dict:
    model.eval()

    total_loss = 0

    labels, predicted = [], []
    with torch.no_grad():
        for bg, bsrc, bdst, blabels in dataloader:
            bg, bsrc, bdst, blabels = bg.to(device), bsrc.to(device), bdst.to(device), blabels.to(device)
            logits = model(bg, bsrc, bdst)
            loss = loss_fn(logits, blabels)
            total_loss += loss.item()
            _, bpredicted = torch.max(logits, dim=1)
            
            labels.append(blabels)
            predicted.append(bpredicted)

            

    # TODO: collect metrics
    labels = torch.cat(labels, dim=0).cpu()
    predicted = torch.cat(predicted, dim=0).cpu()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)

    logging_metrics = {
        'AvgLoss': total_loss / len(dataloader),
        'Accuracy': correct / total,
        # global metrics across all classes
        'Precision': precision_score(labels, predicted, average='micro', zero_division=0), 
        'Recall': recall_score(labels, predicted, average='micro', zero_division=0),
        'F1': f1_score(labels, predicted, average='micro', zero_division=0),
    }
    
    misc_metrics = {
        'ConfusionMatrix': confusion_matrix(labels, predicted, labels=unique_labels),
        # global metrics separated by class
        'Precision-by-class': precision_score(labels, predicted, average=None, zero_division=0, labels=unique_labels), 
        'Recall-by-class': recall_score(labels, predicted, average=None, zero_division=0, labels=unique_labels),
        'F1-by-class': f1_score(labels, predicted, average=None, zero_division=0, labels=unique_labels),
    }

    all_metrics = logging_metrics | misc_metrics
    return all_metrics, logging_metrics 

def edge_classification_collate_fn(batch_items):
    """
    Custom collate_fn for DGL graphs with source node IDs, destination node IDs, and labels.
    Args:
        batch_items (list): A list of tuples, where each tuple is
                            (graph, src_ids_local, dst_ids_local, labels).

    Returns:
        tuple: (batched_graph, batched_src_ids, batched_dst_ids, batched_labels)
            - batched_graph (dgl.DGLGraph): All graphs in the batch combined.
            - batched_src_ids (torch.LongTensor): source node IDs per graph.
            - batched_dst_ids (torch.LongTensor): destination node IDs per graph.
            - batched_labels (torch.LongTensor): Labels for all pairs.
    """
    # Unzip the batch items
    graphs, src_ids_list, dst_ids_list, labels_list = zip(*batch_items)

    # batch graphs
    batched_graph = dgl.batch(graphs)

    # batch edge ids
    batched_src_ids = []
    batched_dst_ids = []

    node_offset = 0

    for i, g in enumerate(graphs):
        batched_src_ids.append(src_ids_list[i] + node_offset)
        batched_dst_ids.append(dst_ids_list[i] + node_offset)
        node_offset += g.num_nodes()

    batched_src_ids = torch.cat(batched_src_ids, dim=0)
    batched_dst_ids = torch.cat(batched_dst_ids, dim=0)

    # batch labels
    batched_labels = torch.cat(labels_list, dim=0) 

    return batched_graph, batched_src_ids, batched_dst_ids, batched_labels

'''
Graph Classification Training Utilities
'''
def train_and_validate_graph_classification(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, num_epochs, validation_interval, tensorboard_writer, model_save_path, unique_labels) -> Tuple[torch.nn.Module, dict]:
    """
    Trains and validates a model with rich progress bars, including the current epoch number.
    """
    model = model.to(device)

    # Define the rich progress bar for the main training epochs
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "|",
        TimeRemainingColumn(),
        "|",
        TextColumn("[progress.description]{task.fields[loss]}"),
    ) as progress:
        # Create a task for the main epoch loop
        epoch_task = progress.add_task("[green]Training...", total=num_epochs, loss="N/A")

        train_loss, val_loss = 0.0, 0.0
        for epoch in range(num_epochs):
            # Validate
            if epoch % validation_interval == 0:
                all_metrics, tensorboard_metrics = validate_one_epoch_graph_classification(model, val_dataloader, loss_fn, device, unique_labels)

                log_validation_metrics(epoch, tensorboard_metrics, all_metrics, tensorboard_writer)

                torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))
                val_loss = all_metrics['AvgLoss']
                progress.update(epoch_task, loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")
                
            # Train
            train_loss = train_one_epoch_graph_classification(model, train_dataloader, optimizer, loss_fn, device)
            progress.update(epoch_task, 
                            advance=1, 
                            description=f"[green]Training Epoch {epoch + 1}/{num_epochs}[/green]",
                            loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")
            tensorboard_writer.add_scalar('Train/AvgLoss', train_loss, epoch)


    # ------ Final Steps ------
    all_metrics, tensorboard_metrics = validate_one_epoch_graph_classification(model, val_dataloader, loss_fn, device, unique_labels)
    for key, value in tensorboard_metrics.items():
        tensorboard_writer.add_scalar(f'Final/{key}', value, num_epochs)
    
    return model, (all_metrics, tensorboard_metrics)

def train_one_epoch_graph_classification(model, dataloader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0
    for bg, blabels in dataloader:
        bg, blabels = bg.to(device), blabels.to(device)
        optimizer.zero_grad()
        logits = model(bg)
        loss = loss_fn(logits, blabels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch_graph_classification(model, dataloader, loss_fn, device, unique_labels) -> dict:
    model.eval()

    total_loss = 0

    labels, predicted = [], []
    with torch.no_grad():
        for bg, blabels in dataloader:
            bg, blabels = bg.to(device), blabels.to(device)
            logits = model(bg)
            loss = loss_fn(logits, blabels)
            total_loss += loss.item()
            _, bpredicted = torch.max(logits, dim=1)
            
            labels.append(blabels)
            predicted.append(bpredicted)
    
    labels = torch.cat(labels, dim=0).cpu()
    predicted = torch.cat(predicted, dim=0).cpu()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)

    logging_metrics = {
        'AvgLoss': total_loss / len(dataloader),
        'Accuracy': correct / total,
        'Precision': precision_score(labels, predicted, average='micro', zero_division=0),
        'Recall': recall_score(labels, predicted, average='micro', zero_division=0),
        'F1': f1_score(labels, predicted, average='micro', zero_division=0),
        # TODO: ...add more metrics if necessary
    }

    misc_metrics = {
        # TODO: add more metrics if necessary, consider local logging
        'ConfusionMatrix': confusion_matrix(labels, predicted, labels=unique_labels),
        'Precision-by-class': precision_score(labels, predicted, average=None, zero_division=0, labels=unique_labels),
        'Recall-by-class': recall_score(labels, predicted, average=None, zero_division=0, labels=unique_labels),
        'F1-by-class': f1_score(labels, predicted, average=None, zero_division=0, labels=unique_labels),
    }

    all_metrics = logging_metrics | misc_metrics
    return all_metrics, logging_metrics

def log_validation_metrics(epoch, tensorboard_metrics=None, all_metrics=None, writer=None) -> None:
    if tensorboard_metrics is not None:
        for key, value in tensorboard_metrics.items():
            writer.add_scalar(f'Validation/{key}', value, epoch)

    if all_metrics is not None:
        logging.info(f"===Epoch {epoch} Metrics===")
        for key, value in all_metrics.items():
            logging.info(f"{key} = {value}")