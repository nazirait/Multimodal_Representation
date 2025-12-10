# src/comparative/models/transformers/clip_emb_module.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler

try:
    from transformers import CLIPTextModel, AutoTokenizer
except Exception:
    CLIPTextModel = None
    AutoTokenizer = None


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class CLIPEmbedConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    eps: float = 1e-8
    max_steps: Optional[int] = None

    text_encoder_name: str = "openai/clip-vit-base-patch16"
    freeze_text: bool = False

    # temperature for InfoNCE
    temperature_init: float = 0.07
    temperature_min: float = 0.01
    temperature_max: float = 0.2

    # NEW: dimension of incoming image_emb (e.g. 512 for CLIP vision;
    #      1129 for MovieLens genome vectors; 1129+gdim if concatenated with graph)
    image_emb_dim: int = 512


class CLIPDualEncoderFromEmbeds(pl.LightningModule):
    """
    Contrastive alignment between:
      - CLIP text tower embeddings (learned / optionally fine-tuned),
      - Precomputed 'image' embeddings from *any* modality.

    Expects batches with:
      batch["text"]: list[str] length B
      batch["image_emb"]: FloatTensor[B, D_in]

    Internally projects image_emb -> D_text to match CLIP text space, then
    applies symmetric InfoNCE.
    """

    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = CLIPEmbedConfig(**cfg)

        if CLIPTextModel is None:
            raise ImportError("transformers not installed or unavailable.")

        # text tower
        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_encoder_name)

        # dimension of CLIP text embeddings
        d_text = getattr(self.text_encoder.config, "hidden_size", None)
        if d_text is None:
            d_text = getattr(self.text_encoder.config, "projection_dim", 512)
        self.d_text = d_text

        # project arbitrary image_emb dim -> CLIP text space
        self.image_proj = nn.Linear(self.cfg.image_emb_dim, self.d_text)

        # learnable temperature, stored as log(1/T)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / self.cfg.temperature_init))
        )

        if self.cfg.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @property
    def temperature(self) -> torch.Tensor:
        # exp(-logit_scale) because param is log(1/T)
        return torch.exp(-self.logit_scale).clamp(
            self.cfg.temperature_min, self.cfg.temperature_max
        )

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize with CLIP's tokenizer and get a D_text embedding per sample.
        """
        max_len = getattr(self.text_encoder.config, "max_position_embeddings", 77) or 77
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(max_len, 77),
            return_tensors="pt",
        ).to(self.device)

        out = self.text_encoder(**tok)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            txt_emb = out.pooler_output  # (B, D)
        else:
            # fallback masked mean over last_hidden_state
            last = out.last_hidden_state  # (B, L, D)
            if "attention_mask" in tok:
                mask = tok["attention_mask"].unsqueeze(-1).float()
                denom = mask.sum(dim=1).clamp(min=1e-6)
                txt_emb = (last * mask).sum(dim=1) / denom
            else:
                txt_emb = last.mean(dim=1)

        txt_emb = l2_normalize(txt_emb)
        return txt_emb  # (B, D_text)

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # text -> CLIP text tower -> normalized embeddings
        raw = batch["text"]
        texts = raw if isinstance(raw, (list, tuple)) else [raw] if isinstance(raw, str) else None
        if texts is None:
            raise ValueError("batch['text'] must be str or List[str].")
        txt_emb = self.encode_text(texts)  # (B, D_text)

        # image_emb -> project into CLIP text space -> normalize
        img_vec = batch["image_emb"].to(self.device).float()  # (B, D_in)
        img_proj = self.image_proj(img_vec)                   # (B, D_text)
        img_emb = l2_normalize(img_proj)

        return txt_emb, img_emb

    def training_step(self, batch, batch_idx):
        txt_emb, img_emb = self.forward(batch)

        # symmetric InfoNCE
        logits_i2t = (img_emb @ txt_emb.t()) / self.temperature  # (B,B)
        logits_t2i = logits_i2t.t()

        targets = torch.arange(len(txt_emb), device=self.device)
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        loss = 0.5 * (loss_i2t + loss_t2i)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/temp", self.temperature.detach(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        txt_emb, img_emb = self.forward(batch)
        logits = (txt_emb @ img_emb.t()) / self.temperature  # (B,B)
        sim_diag = torch.diag(logits).mean()
        val_loss = -sim_diag  # "higher sim_diag is better", so negative is a loss proxy

        self.log("val/sim_diag", sim_diag, prog_bar=True, on_epoch=True)
        self.log("val_loss", val_loss, prog_bar=False, on_epoch=True)
        return {"txt": txt_emb.detach(), "img": img_emb.detach(), "logits": logits.detach()}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # simple optimizer dict, Lightning will handle stepping
        return {"optimizer": optimizer}
