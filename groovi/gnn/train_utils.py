import os
import torch
import dgl
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def train_and_validate(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, num_epochs, validation_interval, tensorboard_writer, model_save_path) -> torch.nn.Module:
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
                metrics = validate_one_epoch(model, val_dataloader, loss_fn, device)
                for key, value in metrics.items():
                    tensorboard_writer.add_scalar(f'Validation/{key}', value, epoch)
                torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))
                
                val_loss = metrics['AvgLoss']
                progress.update(epoch_task, loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")

            # Train
            train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
            progress.update(epoch_task, 
                            advance=1, 
                            description=f"[green]Training Epoch {epoch + 1}/{num_epochs}[/green]",
                            loss=f"[yellow]Train Loss: {train_loss:.4f}[/yellow] | [cyan]Val Acc: {val_loss:.4f}[/cyan]")
            tensorboard_writer.add_scalar('Train/AvgLoss', train_loss, epoch)


    # ------ Final Steps ------
    final_metrics = validate_one_epoch(model, val_dataloader, loss_fn, device)
    for key, value in final_metrics.items():
        tensorboard_writer.add_scalar(f'Final/{key}', value, num_epochs)
    tensorboard_writer.close()
    return model
def train_one_epoch(model, dataloader, optimizer, loss_fn, device) -> float:
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

def validate_one_epoch(model, dataloader, loss_fn, device) -> dict:
    model.eval()

    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for bg, bsrc, bdst, blabels in dataloader:
            bg, bsrc, bdst, blabels = bg.to(device), bsrc.to(device), bdst.to(device), blabels.to(device)
            logits = model(bg, bsrc, bdst)
            loss = loss_fn(logits, blabels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == blabels).sum().item()
            total += blabels.size(0)

    # TODO: collect metrics (accuracy, precision, recall, F1, etc)
    metrics = {
        'AvgLoss': total_loss / len(dataloader),
        'Accuracy': correct / total
    }
    return metrics

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

    # 1. Batch the DGL graphs using dgl.batch
    batched_graph = dgl.batch(graphs)

    # 2. Adjust src_ids and dst_ids for global indexing and concatenate labels
    adjusted_src_ids = []
    adjusted_dst_ids = []
    batched_labels = torch.cat(labels_list, dim=0) # Labels just need concatenation

    current_node_offset = 0

    # Iterate through the original graphs and their corresponding src/dst ID lists
    for i, g in enumerate(graphs):
        # Apply the current node offset to local src_ids and dst_ids
        adjusted_src_ids.append(src_ids_list[i] + current_node_offset)
        adjusted_dst_ids.append(dst_ids_list[i] + current_node_offset)

        # Update the offset for the next graph in the batch
        current_node_offset += g.num_nodes() # Or batched_graph.batch_num_nodes()[i]

    # Concatenate all adjusted source and destination IDs
    batched_src_ids = torch.cat(adjusted_src_ids, dim=0)
    batched_dst_ids = torch.cat(adjusted_dst_ids, dim=0)

    return batched_graph, batched_src_ids, batched_dst_ids, batched_labels