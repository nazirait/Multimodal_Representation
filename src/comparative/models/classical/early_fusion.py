# src/comparative/models/classical/early_fusion.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Optional

class EarlyFusionClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_classes: int = 2,
        image_emb_dim: Optional[int] = None,
        graph_emb_dim: Optional[int] = None,
        tabular_dim: Optional[int] = None,
        fusion_hidden: int = 256,
        lr: float = 2e-5,
        wd: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(model_name)
        bert_dim = self.encoder.config.hidden_size

        input_dim = bert_dim
        if image_emb_dim is not None:
            input_dim += image_emb_dim
        if graph_emb_dim is not None:
            input_dim += graph_emb_dim
        if tabular_dim is not None:
            input_dim += tabular_dim

        self.dropout = nn.Dropout(0.2)
        self.fusion = nn.Linear(input_dim, fusion_hidden)
        self.head = nn.Linear(fusion_hidden, n_classes)
        self.criterion = nn.BCEWithLogitsLoss()  # Multilabel

    def forward(self, batch):
        x = []
        input_ids = batch.get('input_ids', None)
        attn_mask = batch.get('attention_mask', None)
        if input_ids is not None and attn_mask is not None:
            bert_out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
            pooled = bert_out.last_hidden_state[:, 0, :]
            x.append(pooled)

        if 'image_emb' in batch and batch['image_emb'] is not None:
            x.append(batch['image_emb'])
        if 'graph_emb' in batch and batch['graph_emb'] is not None:
            x.append(batch['graph_emb'])
        if 'tabular' in batch and batch['tabular'] is not None:
            x.append(batch['tabular'])
        if not x:
            raise ValueError("No modality found in batch for EarlyFusionClassifier!")

        fused = torch.cat(x, dim=1)
        fused = self.dropout(fused)
        fused = torch.relu(self.fusion(fused))
        return self.head(fused)

    def training_step(self, batch, batch_idx):
        y = batch['label'].float()
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()  # This is micro accuracy
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label'].float()
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['wd']
        )
