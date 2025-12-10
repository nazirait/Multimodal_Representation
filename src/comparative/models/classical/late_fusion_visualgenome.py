# src/comparative/models/classical/late_fusion_visualgenome.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Optional

import torchvision.models as tv_models


class LateFusionClassifier(pl.LightningModule):
    """
    Late-fusion multi-label classifier.

    For Visual Genome:
      - text: caption -> BERT -> MLP -> logits_text
      - image: RGB -> ResNet -> MLP -> logits_image
      - graph: predicate distribution -> MLP -> logits_graph

    Final logits = average of available modality logits.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_classes: int = 2,
        lr: float = 2e-5,
        wd: float = 0.01,
        # Image branch
        use_image_cnn: bool = True,
        image_model: str = "resnet18",
        freeze_image_cnn: bool = True,
        # Graph branch
        graph_emb_dim: Optional[int] = None,
        # Hidden sizes per branch
        text_hidden: int = 256,
        image_hidden: int = 256,
        graph_hidden: int = 128,
        max_text_length: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Text encoder branch
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_dim = self.text_encoder.config.hidden_size

        self.text_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(bert_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, n_classes),
        )

        # Image encoder branch 
        self.image_encoder = None
        self.image_head = None

        if use_image_cnn:
            if image_model == "resnet18":
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
                feat_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
            elif image_model == "resnet50":
                backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
                feat_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported image_model: {image_model}")

            if freeze_image_cnn:
                for p in backbone.parameters():
                    p.requires_grad = False

            self.image_encoder = backbone
            self.image_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feat_dim, image_hidden),
                nn.ReLU(),
                nn.Linear(image_hidden, n_classes),
            )

        # Graph branch
        self.graph_head = None
        if graph_emb_dim is not None and graph_emb_dim > 0:
            self.graph_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(graph_emb_dim, graph_hidden),
                nn.ReLU(),
                nn.Linear(graph_hidden, n_classes),
            )

        # loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.max_text_length = max_text_length

    # helpers

    def _tokenize_if_needed(self, batch):
        input_ids = batch.get("input_ids", None)
        attn_mask = batch.get("attention_mask", None)

        if input_ids is not None and attn_mask is not None:
            return input_ids, attn_mask

        if "text" not in batch:
            return None, None

        texts = batch["text"]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        return input_ids, attn_mask

    def encode_text(self, batch):
        input_ids, attn_mask = self._tokenize_if_needed(batch)
        if input_ids is None or attn_mask is None:
            return None
        out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        pooled = out.last_hidden_state[:, 0, :]   # [CLS] / first token
        return pooled

    def encode_image(self, batch):
        if self.image_encoder is None:
            return None
        img = batch.get("image", None)
        if img is None:
            return None
        # [B,3,H,W], already normalized
        feats = self.image_encoder(img)
        return feats

    def get_graph_emb(self, batch):
        if self.graph_head is None:
            return None
        graph_emb = batch.get("graph_emb", None)
        return graph_emb

    def _get_targets(self, batch):
        if "labels" in batch:
            return batch["labels"].float()
        if "label" in batch:
            return batch["label"].float()
        raise KeyError("Batch must contain 'labels' or 'label' for targets.")

    # forward and steps

    def forward(self, batch):
        logits_list = []

        # text branch
        text_emb = self.encode_text(batch)
        if text_emb is not None:
            logits_text = self.text_head(text_emb)
            logits_list.append(logits_text)

        # image branch
        img_emb = self.encode_image(batch)
        if img_emb is not None and self.image_head is not None:
            logits_img = self.image_head(img_emb)
            logits_list.append(logits_img)

        # graph branch
        graph_emb = self.get_graph_emb(batch)
        if graph_emb is not None and self.graph_head is not None:
            logits_graph = self.graph_head(graph_emb)
            logits_list.append(logits_graph)

        if not logits_list:
            raise ValueError("No modality provided to LateFusionClassifier!")

        # Late fusion: average logits from all available modalities
        fused_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return fused_logits

    def training_step(self, batch, batch_idx):
        y = self._get_targets(batch)
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = self._get_targets(batch)
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["wd"]
        )
