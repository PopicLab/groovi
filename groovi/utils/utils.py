import torch
import numpy as np
import random
import dgl


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    # For CUDA operations (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For all GPUs

def limit_cores(num_cores: int):
    """
    Limit the number of CPU cores used by PyTorch and DGL.
    """
    torch.set_num_threads(num_cores)
    dgl.utils.set_num_threads(num_cores)