# src/comparative/models/classical/late_fusion.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchvision import models
from typing import Optional

class LateFusionClassifier(pl.LightningModule):
    def __init__(
        self,
        text_model_name: Optional[str] = None,
        image_model_name: Optional[str] = None,
        n_classes: int = 2,
        use_text: bool = True,
        use_image: bool = False,
        image_emb_dim: Optional[int] = None,
        use_graph: bool = False,
        graph_emb_dim: Optional[int] = None,
        use_tabular: bool = False,
        tabular_dim: Optional[int] = None,
        fusion_hidden: int = 256,
        lr: float = 2e-5,
        wd: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Text branch
        if use_text and text_model_name:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            self.text_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Identity()
        else:
            self.text_encoder = None
            self.text_dim = 0

        # Raw image (optional)
        if use_image and image_model_name:
            base = models.__dict__[image_model_name](pretrained=True)
            if hasattr(base, "fc") and hasattr(base.fc, "in_features"):
                img_dim = base.fc.in_features
                base.fc = nn.Identity()
            elif hasattr(base, "classifier") and hasattr(base.classifier, "in_features"):
                img_dim = base.classifier.in_features
                base.classifier = nn.Identity()
            else:
                raise ValueError("Unknown CNN arch.")
            self.image_encoder = base
            self.image_proj = nn.Linear(img_dim, 256)
            self.image_dim = 256
        else:
            self.image_encoder = None
            self.image_dim = 0

        # Precomputed image embeddings
        self.image_emb_dim = int(image_emb_dim) if image_emb_dim is not None else 0

        # Other modalities (dims provided by YAML)
        self.graph_dim = int(graph_emb_dim) if (use_graph and graph_emb_dim) else 0
        self.tabular_dim = int(tabular_dim) if (use_tabular and tabular_dim) else 0

        input_dim = self.text_dim + self.image_dim + self.image_emb_dim + self.graph_dim + self.tabular_dim
        if input_dim == 0:
            raise ValueError("No active modalities. Enable at least one.")

        self.dropout = nn.Dropout(0.2)
        self.fusion = nn.Linear(input_dim, fusion_hidden)
        self.head = nn.Linear(fusion_hidden, n_classes)
        self.criterion = nn.BCEWithLogitsLoss()  # multilabel

    def forward(self, batch):
        feats = []

        if self.text_encoder is not None and "input_ids" in batch and "attention_mask" in batch:
            out = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled = out.last_hidden_state[:, 0, :]
            feats.append(self.text_proj(pooled))

        if self.image_encoder is not None and "image" in batch:
            img_feat = self.image_encoder(batch["image"])
            feats.append(self.image_proj(img_feat))

        if self.image_emb_dim > 0 and "image_emb" in batch:
            feats.append(batch["image_emb"])

        if self.graph_dim > 0 and "graph" in batch:
            feats.append(batch["graph"])

        if self.tabular_dim > 0 and "tabular" in batch:
            feats.append(batch["tabular"])

        if not feats:
            raise ValueError("No input modalities found in batch!")

        x = torch.cat(feats, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fusion(x))
        return self.head(x)

    def _metrics(self, logits, y, thr=0.5):
        preds = (torch.sigmoid(logits) > thr).float()
        acc = (preds == y).float().mean()
        return preds, acc

    def training_step(self, batch, batch_idx):
        y = batch["label"].float()
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        _, acc = self._metrics(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["label"].float()
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        _, acc = self._metrics(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["wd"])
