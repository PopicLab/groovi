import torch
import dgl
import logging

def limit_cores(num_cores: int):
    """
    Limit the number of CPU cores used by PyTorch and DGL.
    """
    logging.info(f"Limiting to {num_cores} CPU cores")
    torch.set_num_threads(num_cores)
    dgl.utils.set_num_threads(num_cores)
