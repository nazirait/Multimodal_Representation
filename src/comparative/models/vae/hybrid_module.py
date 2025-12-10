# src/comparative/models/vae/hybrid_module.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from lightning.pytorch.utilities.types import OptimizerLRScheduler



@dataclass
class HybridVAEConfig:
    # backbone
    text_encoder_name: str = "distilbert-base-uncased"

    # classification head
    num_labels: int = 2
    multilabel: bool = False  # MovieLens => True, others => False

    # latent + side modality
    latent_dim: int = 128
    side_dim: int = 0  # 0 => no side branch (Amazon)

    # training hyperparams
    lr: float = 3e-4
    weight_decay: float = 0.01
    eps: float = 1e-8
    max_steps: Optional[int] = 10_000

    # regularization weights
    beta_kl: float = 1e-3
    beta_recon: float = 1.0

    dropout: float = 0.1
    freeze_text: bool = False


class HybridVAEClassifier(pl.LightningModule):
    """
    A simple generative–discriminative hybrid:

    - Encodes text (and optional side embedding) into a latent z using a VAE.
    - Uses z for supervised prediction (classification or multilabel).
    - Optionally reconstructs the side embedding from z (MSE), acting as a
      multimodal bottleneck.

    Works across:
      * Amazon:   text only (side_dim = 0, no reconstruction branch)
      * Fashion:  text + image_emb (512)
      * MovieLens: text + image_emb (= genome vectors, 1129)
    """

    def __init__(self, **cfg: Any):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = HybridVAEConfig(**cfg)

        #  text encoder (shared across datasets) 
        self.text_encoder = AutoModel.from_pretrained(self.cfg.text_encoder_name)
        text_hidden_dim = self.text_encoder.config.hidden_size  # e.g. 768

        if self.cfg.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        #  side modality encoder (optional) 
        # We map side embedding -> same dim as text_hidden_dim
        if self.cfg.side_dim and self.cfg.side_dim > 0:
            self.side_proj = nn.Linear(self.cfg.side_dim, text_hidden_dim)
        else:
            self.side_proj = None

        # VAE latent layers 
        # concat(text_rep, side_rep?) -> mu / logvar -> z
        enc_in_dim = text_hidden_dim + (text_hidden_dim if self.side_proj is not None else 0)

        self.enc_fc = nn.Sequential(
            nn.Linear(enc_in_dim, enc_in_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.fc_mu = nn.Linear(enc_in_dim, self.cfg.latent_dim)
        self.fc_logvar = nn.Linear(enc_in_dim, self.cfg.latent_dim)

        #  classifier on latent z 
        self.cls_head = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.latent_dim, self.cfg.num_labels),
        )

        #  optional reconstruction of side embedding from z 
        if self.side_proj is not None:
            self.recon_side = nn.Linear(self.cfg.latent_dim, self.cfg.side_dim)
        else:
            self.recon_side = None

        #  losses 
        if self.cfg.multilabel:
            self.cls_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()

    #  core helpers 
    def _encode_text(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Encodes BERT-style tokens into a pooled text representation.
        Expects:
          - input_ids
          - attention_mask
        from the datamodules (Amazon / Fashion / MovieLens).
        """
        enc = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        # token_type_ids is optional
        if "token_type_ids" in batch:
            enc["token_type_ids"] = batch["token_type_ids"]

        outputs = self.text_encoder(**enc)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            txt_rep = outputs.pooler_output
        else:
            txt_rep = outputs.last_hidden_state.mean(dim=1)
        return txt_rep  # (B, H)

    def _get_side_emb(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Resolves the 'side' modality embedding in a general way.
        For FashionAI / MovieLens we rely on 'image_emb' (already set
        by their datamodules). For Amazon, this returns None.
        """
        side = None
        if "image_emb" in batch:
            side = batch["image_emb"]
        elif "tabular" in batch:
            side = batch["tabular"]
        elif "graph_emb" in batch:
            side = batch["graph_emb"]

        if side is not None and self.side_proj is not None:
            side = self.side_proj(side)
        return side

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x) || N(0, I)) per batch
        # 0.5 * sum(mu^2 + sigma^2 - log sigma^2 - 1)
        return 0.5 * torch.mean(
            torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1.0, dim=1)
        )

    #  forward / step logic 
    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          logits, mu, logvar, side_recon (or None)
        """
        txt_rep = self._encode_text(batch)  # (B, H)
        side_rep = self._get_side_emb(batch)  # (B, H) or None

        if side_rep is not None:
            h = torch.cat([txt_rep, side_rep], dim=-1)
        else:
            h = txt_rep

        h = self.enc_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)

        logits = self.cls_head(z)

        side_recon = None
        if self.recon_side is not None and "image_emb" in batch:
            side_recon = self.recon_side(z)

        return logits, mu, logvar, side_recon

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        logits, mu, logvar, side_recon = self.forward(batch)

        if self.cfg.multilabel:
            labels = batch["labels"].float()
        else:
            labels = batch["labels"].long()

        # sanity check: logits and labels must agree on last dimension
        if self.cfg.multilabel:
            assert (
                logits.shape[-1] == labels.shape[-1]
            ), f"Logits dim ({logits.shape[-1]}) != labels dim ({labels.shape[-1]}). " \
               f"Check model.num_labels vs MovieLens genre count."

        #  classification loss 
        if self.cfg.multilabel:
            labels = batch["labels"].float()
            cls_loss = self.cls_loss_fn(logits, labels)
        else:
            labels = batch["labels"].long()
            cls_loss = self.cls_loss_fn(logits, labels)

        #  KL divergence 
        kl = self._compute_kl(mu, logvar)

        #  optional reconstruction (side modality) 
        recon_loss = torch.tensor(0.0, device=self.device)
        if side_recon is not None and "image_emb" in batch:
            recon_loss = F.mse_loss(side_recon, batch["image_emb"])

        loss = cls_loss + self.cfg.beta_kl * kl + self.cfg.beta_recon * recon_loss

        # names with slash, for nice grouping
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/kl", kl, prog_bar=False, on_step=False, on_epoch=True)
        if side_recon is not None:
            self.log(f"{stage}/recon_loss", recon_loss, prog_bar=False, on_step=False, on_epoch=True)

        # classic names without slash, so EarlyStopping & other callbacks find them
        if stage == "train":
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        elif stage == "val":
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "test":
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # quick accuracy for single-label tasks (Amazon, Fashion)
        if not self.cfg.multilabel:
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    #  optimizers 
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.max_steps or 10_000
        )

        # Return a dict (single optimizer + single scheduler) so both
        # Lightning and Pylance are happy.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step every training step
            },
        }
