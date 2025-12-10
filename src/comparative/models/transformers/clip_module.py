# src/comparative/models/transformers/clip_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler

try:
    from transformers import CLIPTextModel, CLIPTokenizerFast
except Exception:
    CLIPTextModel = None
    CLIPTokenizerFast = None


@dataclass
class CLIPModuleConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    eps: float = 1e-8
    max_steps: int = 10_000

    text_encoder_name: str = "openai/clip-vit-base-patch16"
    classifier_hidden: int = 512

    # SAFE DEFAULTS FOR AMAZON 
    num_labels: int = 2
    multilabel: bool = False
    freeze_text: bool = False


class CLIPTextOnlyClassifier(pl.LightningModule):
    """CLIP text tower → classifier. Expects batch['text'] as str | List[str]."""

    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = CLIPModuleConfig(**cfg)

        if CLIPTextModel is None:
            raise ImportError("transformers is not available.")

        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.text_encoder_name)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.cfg.text_encoder_name)

        d = getattr(self.text_encoder.config, "hidden_size", 512)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, self.cfg.classifier_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.classifier_hidden, self.cfg.num_labels),
        )

        self.loss_fn = nn.BCEWithLogitsLoss() if self.cfg.multilabel else nn.CrossEntropyLoss()

        if self.cfg.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self._expected_out = int(self.cfg.num_labels)

    def _prep_text(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "text" not in batch:
            raise KeyError("CLIPTextOnlyClassifier expects batch['text'] = str | List[str].")
        raw = batch["text"]
        texts = raw if isinstance(raw, (list, tuple)) else [raw] if isinstance(raw, str) else None
        if texts is None:
            raise ValueError("batch['text'] must be str or List[str].")

        max_pos = getattr(self.text_encoder.config, "max_position_embeddings", 77) or 77
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=min(77, max_pos), return_tensors="pt")
        return {k: v.to(self.device) for k, v in tok.items()}

    @staticmethod
    def _labels(batch: Dict[str, Any], multilabel: bool) -> torch.Tensor:
        if multilabel:
            if "labels" not in batch:
                raise KeyError("Multilabel task expects batch['labels'] with shape (B, C).")
            return batch["labels"]
        if "label" in batch:
            return batch["label"].long()
        if "labels" in batch:
            return batch["labels"].long()
        raise KeyError("Single-label task expects batch['label'] or batch['labels'] as indices.")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        tok = self._prep_text(batch)
        out = self.text_encoder(**tok)
        last = out.last_hidden_state  # (B, L, D)
        mask = tok.get("attention_mask", torch.ones_like(last[..., 0])).unsqueeze(-1).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # (B, D)
        logits = self.head(emb)  # (B, num_labels)
        if logits.shape[-1] != self._expected_out:
            raise RuntimeError(
                f"Head/output mismatch: logits dim {logits.shape[-1]} != num_labels {self._expected_out}. "
                "Check config/ckpt."
            )
        return logits

    def training_step(self, batch, _):
        logits = self.forward(batch)
        y = self._labels(batch, self.cfg.multilabel)
        loss = self.loss_fn(logits, y.float() if self.cfg.multilabel else y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        logits = self.forward(batch)
        y = self._labels(batch, self.cfg.multilabel)
        loss = self.loss_fn(logits, y.float() if self.cfg.multilabel else y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)
        if not self.cfg.multilabel:
            acc = (logits.argmax(-1) == y).float().mean()
            self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch, _):
        logits = self.forward(batch)
        y = self._labels(batch, self.cfg.multilabel)
        loss = self.loss_fn(logits, y.float() if self.cfg.multilabel else y)
        self.log("test/loss", loss, on_epoch=True)
        if not self.cfg.multilabel:
            acc = (logits.argmax(-1) == y).float().mean()
            self.log("test/acc", acc, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        head_params = list(self.head.parameters())
        enc_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": head_params, "lr": max(self.cfg.lr * 10, 1e-4)},
             {"params": enc_params, "lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay}],
            eps=self.cfg.eps
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.cfg.max_steps, 1000))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
