# src/comparative/training/callbacks.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(PROJECT_ROOT))


from src.comparative.utils.paths import CHECKPOINTS

def basic_callbacks(cfg=None):
    callbacks = []

    model_name = (
        cfg.model.name
        if cfg is not None and hasattr(cfg, "model") and hasattr(cfg.model, "name")
        else "default"
    )

    checkpoint_dir = CHECKPOINTS / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath= str(checkpoint_dir),
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.3f}",
        save_last=True,
        auto_insert_metric_name=True,
    )
    callbacks.append(checkpoint_cb)

    # Early stopping to prevent overfitting )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        verbose=True
    )
    callbacks.append(early_stop_cb)

    return callbacks
