# src/comparative/models/vae/mvae_module_visualgenome.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel

import pytorch_lightning as pl


@dataclass
class MVAECfg:
    text_encoder_name: str = "distilbert-base-uncased"
    num_labels: int = 20          # 20 genres on MovieLens
    multilabel: bool = True       # MovieLens is multi-label

    latent_dim: int = 128
    side_dim: int = 0             # 0 = no side modality; 1129 = genome only, etc.

    lr: float = 3e-4
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_steps: int = 100_000      # used for cosine LR, if you enable it

    beta_kl: float = 1e-3
    beta_side_recon: float = 1.0

    dropout: float = 0.1
    freeze_text: bool = False


class MVAEClassifier(pl.LightningModule):
    """
    Multimodal VAE classifier for MovieLens-style data.

    Inputs:
      - batch["input_ids"], batch["attention_mask"]    : tokenized text
      - batch["label"] or batch["labels"]             : multi-hot genres (B, C)
      - (optional) batch["image_emb"]                 : side embedding (B, side_dim)
    """

    def __init__(self, cfg: Optional[MVAECfg] = None, **hparams: Any):
        super().__init__()

        if cfg is None:
            cfg = MVAECfg(**hparams)
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.text_encoder_name)
        text_hidden_dim = self.text_encoder.config.hidden_size  # 768 for DistilBERT

        if cfg.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Encoder over [text_rep ; side]
        enc_in_dim = text_hidden_dim + cfg.side_dim
        self.enc_fc = nn.Sequential(
            nn.Linear(enc_in_dim, enc_in_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.fc_mu = nn.Linear(enc_in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(enc_in_dim, cfg.latent_dim)

        # Decoder for side modality (if any)
        if cfg.side_dim > 0:
            self.dec_side = nn.Sequential(
                nn.Linear(cfg.latent_dim, enc_in_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(enc_in_dim, cfg.side_dim),
            )
        else:
            self.dec_side = None

        # Classifier head on z
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, cfg.num_labels),
        )

    # helpers
    def encode_text(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean-pool token embeddings (mask-aware)."""
        out = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = out.last_hidden_state              # (B, T, H)
        mask = batch["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = hidden * mask
        denom = mask.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / denom            # (B, H)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward for inference / analysis.

        Returns:
          logits: (B, num_labels)
          aux:    dict with z, mu, logvar, side_recon (if any)
        """
        # text 
        txt_rep = self.encode_text(batch)  # (B, H)

        # side modality 
        side = batch.get("image_emb", None)
        if side is not None:
            # guard: clean NaN/Inf from genome vectors
            side_rep = torch.nan_to_num(side, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if self.cfg.side_dim > 0:
                side_rep = torch.zeros(
                    txt_rep.size(0),
                    self.cfg.side_dim,
                    device=txt_rep.device,
                    dtype=txt_rep.dtype,
                )
            else:
                side_rep = None

        if side_rep is not None:
            h_in = torch.cat([txt_rep, side_rep], dim=-1)
        else:
            h_in = txt_rep

        # VAE encoder 
        h = self.enc_fc(h_in)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # clamp logvar to avoid exp overflow
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        z = self.reparameterize(mu, logvar)

        # Decoder (side reconstruction) 
        side_recon = None
        side_target = None
        if self.dec_side is not None and side_rep is not None:
            side_recon = self.dec_side(z)  # (B, side_dim)
            side_target = side_rep

        # Classifier 
        logits = self.cls_head(z)  # (B, C)

        aux = {
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }
        if side_recon is not None:
            aux["side_recon"] = side_recon
            aux["side_target"] = side_target

        return logits, aux

    # training / validation
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, torch.Tensor]:
        logits, aux = self.forward(batch)

        labels = batch.get("labels", batch.get("label"))
        if labels is None:
            raise RuntimeError("Batch must contain 'labels' or 'label' for MVAEClassifier.")

        labels = labels.float()

        if self.cfg.multilabel:
            cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            cls_loss = F.cross_entropy(logits, labels.long())

        # KL divergence
        mu = aux["mu"]
        logvar = aux["logvar"]
        kl_per_dim = 1 + logvar - mu.pow(2) - logvar.exp()
        kl = -0.5 * kl_per_dim.sum(dim=1).mean()

        # Side reconstruction loss
        side_recon_loss = torch.tensor(0.0, device=self.device)
        if "side_recon" in aux:
            side_target = aux["side_target"]
            side_recon = aux["side_recon"]
            # clean any NaNs that might still sneak in
            side_target = torch.nan_to_num(side_target, nan=0.0, posinf=0.0, neginf=0.0)
            side_recon_loss = F.mse_loss(side_recon, side_target)

        loss = cls_loss + self.cfg.beta_kl * kl + self.cfg.beta_side_recon * side_recon_loss

        # logging 
        bs = labels.size(0)

        # main names with slash (what we used before)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/cls_loss", cls_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/kl", kl, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        if self.cfg.side_dim > 0:
            self.log(f"{stage}/side_recon", side_recon_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)

        # aliases WITHOUT slash, so EarlyStopping("val_loss") works
        if stage == "val":
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        if stage == "train":
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)

        # rough accuracy: compare dominant genre
        with torch.no_grad():
            if self.cfg.multilabel:
                preds = (logits > 0.0).float()
                y_true_dom = labels.argmax(dim=1)
                y_pred_dom = preds.argmax(dim=1)
            else:
                y_true_dom = labels.long()
                y_pred_dom = logits.argmax(dim=1)

            acc = (y_true_dom == y_pred_dom).float().mean()

        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "kl": kl,
            "side_recon": side_recon_loss,
            "acc": acc,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch, stage="train")
        return out["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch, stage="val")
        return out["loss"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch, stage="test")
        return out["loss"]

    # optimizers
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            eps=self.cfg.eps,
        )
        return optimizer

