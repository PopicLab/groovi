import torch
import numpy as np
import random
import logging
import dgl

def seed_everything(seed: int):
    logging.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    # For CUDA operations (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For all GPUs