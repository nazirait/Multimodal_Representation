import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, Any

import torchvision.models as tv_models


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
        # visual-genome specific
        use_image_cnn: bool = False,
        image_model: str = "resnet18",
        freeze_image_cnn: bool = True,
        max_text_length: int = 64,
    ):
        """
        Generic early-fusion classifier:
        - Text encoder: HuggingFace Transformer
        - Optional: image CNN backbone (ResNet)
        - Optional: graph embedding (dense vector)
        - Optional: tabular features
        """
        super().__init__()
        self.save_hyperparameters()

        # text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        bert_dim = self.text_encoder.config.hidden_size

        # img encoder
        self.use_image_cnn = use_image_cnn
        self.image_feat_dim = 0
        self.image_backbone = None

        if self.use_image_cnn:
            if image_model == "resnet18":
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            elif image_model == "resnet50":
                backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported image_model: {image_model}")

            # Replace FC with identity to get global pooled features
            self.image_feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.image_backbone = backbone

            if freeze_image_cnn:
                for p in self.image_backbone.parameters():
                    p.requires_grad = False

        # Input dimension for fusion
        input_dim = bert_dim

        # Image branch
        if self.use_image_cnn:
            input_dim += self.image_feat_dim
        elif image_emb_dim is not None:
            # for pre-computed image embeddings scenario
            input_dim += image_emb_dim

        # Graph branch
        self.graph_emb_dim = graph_emb_dim
        if graph_emb_dim is not None:
            input_dim += graph_emb_dim

        # Tabular
        self.tabular_dim = tabular_dim
        if tabular_dim is not None:
            input_dim += tabular_dim

        # fusion & head
        self.dropout = nn.Dropout(0.2)
        self.fusion = nn.Linear(input_dim, fusion_hidden)
        self.head = nn.Linear(fusion_hidden, n_classes)

        # Multilabel loss
        self.criterion = nn.BCEWithLogitsLoss()

    # Helper: encode text
    def encode_text(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Handles two cases:
        - Already tokenized: batch['input_ids'], batch['attention_mask']
        - Raw text list: batch['text']
        """
        if "input_ids" in batch and "attention_mask" in batch:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # CLS / first-token representation
            pooled = outputs.last_hidden_state[:, 0, :]
            return pooled

        elif "text" in batch:
            texts = batch["text"]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.hparams.max_text_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.text_encoder(**enc)
            pooled = outputs.last_hidden_state[:, 0, :]
            return pooled

        else:
            raise ValueError("Batch must contain either ('input_ids','attention_mask') or 'text'.")

    # forward
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        pieces = []

        # text
        text_emb = self.encode_text(batch)
        pieces.append(text_emb)

        # img
        if self.use_image_cnn and "image" in batch and batch["image"] is not None:
            # batch["image"]: (B, C, H, W)
            img = batch["image"].to(self.device)
            img_feat = self.image_backbone(img)  # (B, image_feat_dim)
            pieces.append(img_feat)
        elif "image_emb" in batch and batch["image_emb"] is not None:
            pieces.append(batch["image_emb"])

        # graph: use
        if "graph" in batch and batch["graph"] is not None and self.graph_emb_dim is not None:
            pieces.append(batch["graph"].to(self.device))

        # tabulr, if any
        if "tabular" in batch and batch["tabular"] is not None and self.tabular_dim is not None:
            pieces.append(batch["tabular"].to(self.device))

        if not pieces:
            raise ValueError("No modalities found in batch for EarlyFusionClassifier!")

        fused = torch.cat(pieces, dim=1)
        fused = self.dropout(fused)
        fused = F.relu(self.fusion(fused))
        logits = self.head(fused)
        return logits

    # Training / Validation / Test steps
    def _step(self, batch: Dict[str, Any], stage: str):
        # Allow both 'labels' (VisualGenome, MovieLens VAEs) and 'label' (old code)
        y = batch.get("labels", batch.get("label")).float().to(self.device)
        logits = self.forward(batch)
        loss = self.criterion(logits, y)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=y.size(0))
        self.log(f"{stage}_acc", acc, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=y.size(0))

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, "test")

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
