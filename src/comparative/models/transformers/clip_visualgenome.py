# src/comparative/models/transformers/clip_visualgenome.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional

from transformers import AutoModel, AutoTokenizer
import torchvision.models as tv_models


class CLIPVisualGenome(pl.LightningModule):
    """
    CLIP-style model for Visual Genome with three modalities:
      - Image: RGB -> CNN -> embedding
      - Text: caption -> Transformer (DistilBERT) -> embedding
      - Graph: predicate distribution -> MLP -> embedding

    Training objectives:
      1) CLIP-like contrastive loss between image and text embeddings
         (optionally including graph as a third view).
      2) Multi-label classification loss on fused embedding -> 500 object labels.
    """

    def __init__(
        self,
            model_name: str = "distilbert-base-uncased",
            n_classes: int = 500,
            lr: float = 2e-5,
            wd: float = 0.01,
            # shared embedding dimension
            embed_dim: int = 256,
            # image branch
            use_image_cnn: bool = True,
            image_model: str = "resnet18",
            freeze_image_cnn: bool = True,
            # graph branch
            graph_emb_dim: Optional[int] = 50,
            graph_hidden: int = 128,
            # text branch
            max_text_length: int = 64,
            # loss weights
            cls_loss_weight: float = 1.0,
            contrastive_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # text branch
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_dim = self.text_encoder.config.hidden_size

        self.text_proj = nn.Linear(bert_dim, embed_dim)

        # image branch
        self.image_encoder = None
        self.image_proj = None
        if use_image_cnn:
            if image_model == "resnet18":
                backbone = tv_models.resnet18(
                    weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
                )
                feat_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
            elif image_model == "resnet50":
                backbone = tv_models.resnet50(
                    weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
                )
                feat_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported image_model: {image_model}")

            if freeze_image_cnn:
                for p in backbone.parameters():
                    p.requires_grad = False

            self.image_encoder = backbone
            self.image_proj = nn.Linear(feat_dim, embed_dim)

        # graph branch 
        self.graph_proj = None
        if graph_emb_dim is not None and graph_emb_dim > 0:
            self.graph_proj = nn.Sequential(
                nn.Linear(graph_emb_dim, graph_hidden),
                nn.ReLU(),
                nn.Linear(graph_hidden, embed_dim),
            )

        # classification head
        # We fuse 3 embeddings (mean) and predict multilabel object classes
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.bce = nn.BCEWithLogitsLoss()

        # CLIP temperature 
        # logit_scale is exp()'d to make it positive
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        self.max_text_length = max_text_length

    # encoding helpers

    def _tokenize_batch(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def encode_text(self, batch):
        if "input_ids" in batch and "attention_mask" in batch:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
        else:
            if "text" not in batch:
                return None
            texts = batch["text"]
            input_ids, attn_mask = self._tokenize_batch(texts)

        out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        # use CLS / first-token embedding
        pooled = out.last_hidden_state[:, 0, :]  # [B, bert_dim]
        z = self.text_proj(pooled)               # [B, embed_dim]
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

    def encode_image(self, batch):
        if self.image_encoder is None:
            return None
        img = batch.get("image", None)
        if img is None:
            return None
        feats = self.image_encoder(img)          # [B, feat_dim]
        z = self.image_proj(feats)               # [B, embed_dim]
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

    def encode_graph(self, batch):
        if self.graph_proj is None:
            return None
        graph_emb = batch.get("graph_emb", None)
        if graph_emb is None:
            return None
        z = self.graph_proj(graph_emb)           # [B, embed_dim]
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

    def _get_targets(self, batch):
        if "labels" in batch:
            return batch["labels"].float()
        if "label" in batch:
            return batch["label"].float()
        raise KeyError("Batch must contain 'labels' or 'label' for targets.")

    # losses

    def clip_contrastive_loss(self, img_z, txt_z):
        """
        Standard CLIP loss between image and text embeddings.
        """
        if img_z is None or txt_z is None:
            return torch.tensor(0.0, device=self.device)

        batch_size = img_z.size(0)
        logit_scale = self.logit_scale.exp().clamp(1e-2, 100.0)

        logits_per_image = logit_scale * img_z @ txt_z.t()    # [B, B]
        logits_per_text = logits_per_image.t()                # [B, B]

        labels = torch.arange(batch_size, device=self.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        return loss

    def classification_loss(self, fused_z, targets):
        logits = self.classifier(fused_z)
        loss = self.bce(logits, targets)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == targets).float().mean()
        return loss, acc, logits

    # forward & steps 

    def forward(self, batch):
        """
        Returns:
          fused_z: fused embedding of image + text + graph
          logits: classification logits over n_classes
        """
        img_z = self.encode_image(batch)
        txt_z = self.encode_text(batch)
        graph_z = self.encode_graph(batch)

        zs = []
        if img_z is not None:
            zs.append(img_z)
        if txt_z is not None:
            zs.append(txt_z)
        if graph_z is not None:
            zs.append(graph_z)

        if not zs:
            raise ValueError("No modality available in CLIPVisualGenome.forward")

        # simple late fusion in embedding space: mean of normalized embeddings
        fused_z = torch.stack(zs, dim=0).mean(dim=0)  # [B, embed_dim]

        logits = self.classifier(fused_z)
        return fused_z, logits

    def training_step(self, batch, batch_idx):
        y = self._get_targets(batch)

        img_z = self.encode_image(batch)
        txt_z = self.encode_text(batch)
        graph_z = self.encode_graph(batch)

        # classification on fused embedding
        zs = []
        if img_z is not None:
            zs.append(img_z)
        if txt_z is not None:
            zs.append(txt_z)
        if graph_z is not None:
            zs.append(graph_z)
        fused_z = torch.stack(zs, dim=0).mean(dim=0)

        cls_loss, acc, logits = self.classification_loss(fused_z, y)

        # CLIP contrastive between image and text
        contrastive_loss = self.clip_contrastive_loss(img_z, txt_z)

        loss = (
            self.hparams.cls_loss_weight * cls_loss
            + self.hparams.contrastive_loss_weight * contrastive_loss
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_clip_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = self._get_targets(batch)

        img_z = self.encode_image(batch)
        txt_z = self.encode_text(batch)
        graph_z = self.encode_graph(batch)

        zs = []
        if img_z is not None:
            zs.append(img_z)
        if txt_z is not None:
            zs.append(txt_z)
        if graph_z is not None:
            zs.append(graph_z)
        fused_z = torch.stack(zs, dim=0).mean(dim=0)

        cls_loss, acc, logits = self.classification_loss(fused_z, y)
        contrastive_loss = self.clip_contrastive_loss(img_z, txt_z)
        loss = (
            self.hparams.cls_loss_weight * cls_loss
            + self.hparams.contrastive_loss_weight * contrastive_loss
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_cls_loss", cls_loss, on_epoch=True, prog_bar=False)
        self.log("val_clip_loss", contrastive_loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
