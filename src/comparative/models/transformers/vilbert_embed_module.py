from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


class CrossAttentionBlock(nn.Module):
    """
    One block of (text attends to visual summary) + FFN + residual + LayerNorm.
    """
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_hidden: torch.Tensor, vis_hidden: torch.Tensor) -> torch.Tensor:
        # text_hidden: (B, Lt, D)
        # vis_hidden:  (B, Lv, D)  [Lv=1]
        q = self.q_proj(text_hidden)
        k = self.k_proj(vis_hidden)
        v = self.v_proj(vis_hidden)

        attn_out, _ = self.attn(q, k, v)  # (B, Lt, D)
        x = self.norm1(text_hidden + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ViLBERTEmbedClassifier(pl.LightningModule):
    """
    Cross-attention fusion classifier: text tokens attend to a projected visual embedding token.

    Expects batch:
      {
        "text": [str, ...],
        "image_emb": FloatTensor[B, D_img],
        "labels": 
            - LongTensor[B]       for single-label (e.g. FashionAI), or
            - FloatTensor[B, C]   for multi-label (e.g. MovieLens, multi-hot)
      }

    Modes:
      - single-label: CrossEntropyLoss over num_labels classes
      - multi-label:  BCEWithLogitsLoss over num_labels outputs
    """

    def __init__(
        self,
        text_encoder_name: str = "distilbert-base-uncased",
        num_labels: int = 138,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        max_len: int = 128,
        num_xattn_layers: int = 2,
        img_emb_dim: int = 512,
        dropout: float = 0.1,
        freeze_text: bool = False,
        multilabel: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if AutoModel is None:
            raise ImportError("transformers not installed or unavailable.")

        self.num_labels = num_labels
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.img_emb_dim = img_emb_dim
        self.freeze_text = freeze_text
        self.multilabel = multilabel

        # text encoder backbone (e.g. DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

        d_text = self.text_encoder.config.hidden_size  # e.g. 768

        # project precomputed "visual" embedding (image_emb) to same hidden dim
        self.visual_proj = nn.Linear(img_emb_dim, d_text)

        # stack of cross-attention layers
        self.xattn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=d_text,
                    num_heads=8,
                    ff_mult=4,
                    dropout=dropout,
                )
                for _ in range(num_xattn_layers)
            ]
        )

        # pool text tokens after fusion
        self.pooler = nn.AdaptiveAvgPool1d(1)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_text, d_text),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_text, num_labels),
        )

        # loss: single-label vs multi-label
        if self.multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        labels = batch["labels"].to(self.device)
        if self.multilabel:
            return labels.float()  # (B, C) multi-hot
        return labels.long()      # (B,) class indices

    def _top1(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Top-1 accuracy.

        - single-label: argmax(logits) == label
        - multi-label:  argmax(logits) is in the positive label set
        """
        preds = torch.argmax(logits, dim=1)  # (B,)

        if self.multilabel:
            # labels: (B, C) float multi-hot
            if labels.ndim == 2:
                # correct if the predicted class has label > 0.5
                idx = torch.arange(labels.size(0), device=labels.device)
                correct = labels[idx, preds] > 0.5
                return correct.float().mean()
            # fallback
            return torch.tensor(0.0, device=logits.device)

        # single-label case
        return (preds == labels).float().mean()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # encode text
        texts = batch["text"]  # list[str]
        tok = self._tokenize_text(texts)
        text_out = self.text_encoder(**tok)
        text_hidden = text_out.last_hidden_state  # (B, Lt, D)

        # project visual/side embedding -> (B,1,D)
        img_vec = batch["image_emb"].to(self.device).float()  # (B, D_img)
        vis_token = self.visual_proj(img_vec).unsqueeze(1)    # (B,1,D)

        # fuse via cross-attention
        fused = text_hidden
        for layer in self.xattn_layers:
            fused = layer(fused, vis_token)  # (B,Lt,D)

        # pool over tokens
        fused_t = fused.transpose(1, 2)  # (B,D,Lt)
        pooled = self.pooler(fused_t).squeeze(-1)  # (B,D)

        # classify
        logits = self.classifier(pooled)  # (B,num_labels)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = self._get_labels(batch)
        loss = self.loss_fn(logits, labels)

        acc = self._top1(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc_top1_epoch", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = self._get_labels(batch)
        loss = self.loss_fn(logits, labels)

        acc = self._top1(logits, labels)

        # EarlyStopping will monitor "val_loss" by default
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc_top1", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"logits": logits.detach(), "labels": labels.detach()}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}
