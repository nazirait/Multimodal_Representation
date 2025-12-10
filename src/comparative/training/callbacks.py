# src/comparative/training/callbacks.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def basic_callbacks(cfg=None):
    callbacks = []

    # Save best and last model (per metric, per run)
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"D:/COmparative_Study_of_Multimodal_Represenations/src/comparative/checkpoints/{cfg.model.name if cfg and hasattr(cfg.model, 'name') else 'default'}/",
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
