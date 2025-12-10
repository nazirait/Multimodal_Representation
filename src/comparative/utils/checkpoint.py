# src/comparative/utils/checkpoint.py

import torch
from pathlib import Path

def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

def load_checkpoint(path, map_location=None):
    return torch.load(str(path), map_location=map_location or "cpu")
