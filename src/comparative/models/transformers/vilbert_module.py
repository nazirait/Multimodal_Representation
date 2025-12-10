from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler  # typing alias

try:
    from transformers import AutoModel, AutoTokenizer, ViTModel, ViTImageProcessor
except Exception:
    AutoModel = None
    AutoTokenizer = None
    ViTModel = None
    ViTImageProcessor = None


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.attn = nn.MultiheadAttention(dim_q, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim_q, ff_mult * dim_q), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * dim_q, dim_q)
        )
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(kv)
        v_proj = self.v_proj(kv)
        out, _ = self.attn(q_proj, k_proj, v_proj)
        q = self.norm1(q + out)
        q = self.norm2(q + self.ff(q))
        return q


class ViLBERTStyleModule(pl.LightningModule):
    """
    Pragmatic ViLBERT-style late/hybrid fusion:
      - Text tower: AutoModel (e.g., DistilBERT/Roberta) with raw strings tokenized here
      - Vision tower: ViTModel via ViTImageProcessor
      - Fusion: stack of cross-attention blocks (text queries attend to vision keys/values)
      - Head: pooled fused text tokens -> classifier

    Batch expected:
      {
        "image": PIL list or FloatTensor[B, 3, H, W],
        "text":  List[str] or str,
        "labels": FloatTensor[B, C] (multilabel) or LongTensor[B] (multiclass)
      }
    """

    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if AutoModel is None:
            raise ImportError("transformers not installed or unavailable.")

        self.text_name = cfg.get("text_encoder_name", "distilbert-base-uncased")
        self.vision_name = cfg.get("vision_encoder_name", "google/vit-base-patch16-224")
        self.num_labels = int(cfg.get("num_labels", 139))
        self.multilabel = bool(cfg.get("multilabel", True))
        self.freeze_backbones = bool(cfg.get("freeze_backbones", True))
        self.num_xattn = int(cfg.get("num_xattn", 2))
        self.lr = float(cfg.get("lr", 3e-5))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.max_len = int(cfg.get("max_len", 128))  # cap for text tokenizer

        # encoders
        self.text_encoder = AutoModel.from_pretrained(self.text_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_name)
        self.vision_encoder = ViTModel.from_pretrained(self.vision_name)
        self.image_processor = ViTImageProcessor.from_pretrained(self.vision_name)

        d_text = self.text_encoder.config.hidden_size
        d_vision = self.vision_encoder.config.hidden_size
        self.text_proj = nn.Linear(d_text, d_text)
        self.vision_proj = nn.Linear(d_vision, d_text)

        self.xattn_blocks = nn.ModuleList([CrossAttentionBlock(d_text, d_text) for _ in range(self.num_xattn)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_text, self.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

        if self.freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    # helpers
    def _prep_text_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "text" not in batch:
            raise KeyError("ViLBERTStyleModule expects raw strings in batch['text'].")

        raw = batch["text"]
        if isinstance(raw, (list, tuple)):
            texts = list(raw)
        elif isinstance(raw, str):
            texts = [raw]
        else:
            raise ValueError("batch['text'] must be str or List[str].")

        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    def _prep_images(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "image" not in batch:
            raise KeyError("ViLBERTStyleModule expects images in batch['image'].")

        if isinstance(batch["image"], torch.Tensor):
            # Assume already normalized/resize handled upstream
            return batch["image"].to(self.device)
        else:
            processed = self.image_processor(images=batch["image"], return_tensors="pt")
            return processed["pixel_values"].to(self.device)

    @staticmethod
    def _extract_labels(batch: Dict[str, Any], multilabel: bool) -> torch.Tensor:
        if multilabel:
            if "labels" in batch:
                return batch["labels"]
            raise KeyError("Multilabel task expects 'labels' in batch.")
        else:
            if "labels" in batch:
                return batch["labels"]
            if "label" in batch:
                return batch["label"]
            raise KeyError("Multiclass task expects 'label' or 'labels' in batch.")

    # core 
    def encode(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self._prep_text_inputs(batch)
        t_out = self.text_encoder(**tok)
        t_hidden = self.text_proj(t_out.last_hidden_state)  # (B, Lt, D)

        pixel = self._prep_images(batch)
        v_out = self.vision_encoder(pixel)
        v_hidden = self.vision_proj(v_out.last_hidden_state)  # (B, Lv, D)

        return t_hidden, v_hidden

    def fuse(self, t_hidden: torch.Tensor, v_hidden: torch.Tensor) -> torch.Tensor:
        h = t_hidden
        for blk in self.xattn_blocks:
            h = blk(h, v_hidden)
        h = h.transpose(1, 2)             # (B, D, L)
        h = self.pool(h).squeeze(-1)      # (B, D)
        return h

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        t_hidden, v_hidden = self.encode(batch)
        fused = self.fuse(t_hidden, v_hidden)
        logits = self.classifier(fused)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = self._extract_labels(batch, self.multilabel)
        if self.multilabel:
            labels = labels.float()
        loss = self.loss_fn(logits, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = self._extract_labels(batch, self.multilabel)
        if self.multilabel:
            labels = labels.float()
        loss = self.loss_fn(logits, labels)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return {"logits": logits.detach(), "labels": labels.detach()}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # cosine works well here too; step-wise annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.get("max_steps", 10000)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val_loss",
            },
        }
