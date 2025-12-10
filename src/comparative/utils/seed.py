# src/comparative/utils/seed.py

import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For Lightning users (if needed)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed)
    except ImportError:
        pass
    # Some PyTorch settings for deterministic results (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
