# src/comparative/models/transformers/vilbert_visualgenome.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional

from transformers import AutoModel, AutoTokenizer
import torchvision.models as tv_models


class VisualGenomeViLBERT(pl.LightningModule):
    """
    Simplified ViLBERT-style model for Visual Genome.

    - Text: caption -> Transformer (e.g. DistilBERT) -> token sequence
    - Image: RGB -> CNN -> global feature -> single "image token"
    - Graph: predicate distribution -> MLP -> single "graph token"

    We then:
      - project all tokens into a shared d_model
      - add modality (segment) embeddings + position embeddings
      - run a small Transformer encoder over [graph_token, image_token, text_tokens...]
      - pool the final sequence and predict multi-label object classes.

    This is not full original ViLBERT (with two separate streams),
    but it's a co-encoded multimodal transformer in the ViLBERT spirit.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_classes: int = 500,
        lr: float = 2e-5,
        wd: float = 0.01,
        # shared transformer dimension
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # image branch
        use_image_cnn: bool = True,
        image_model: str = "resnet18",
        freeze_image_cnn: bool = True,
        # graph branch
        graph_emb_dim: Optional[int] = 50,
        graph_hidden: int = 128,
        # text
        max_text_length: int = 64,
        # pooling
        pool_type: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        self.save_hyperparameters()

        # Text encoder (pretrained) 
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_dim = self.text_encoder.config.hidden_size

        # project text hidden size -> d_model
        self.text_proj = nn.Linear(bert_dim, d_model)

        # Image encoder (CNN -> image token) 
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
            self.image_proj = nn.Linear(feat_dim, d_model)

        # Graph encoder (graph token) 
        self.graph_proj = None
        if graph_emb_dim is not None and graph_emb_dim > 0:
            self.graph_proj = nn.Sequential(
                nn.Linear(graph_emb_dim, graph_hidden),
                nn.ReLU(),
                nn.Linear(graph_hidden, d_model),
            )

        # Modality (segment) embeddings 
        # 0 = graph, 1 = image, 2 = text
        self.modality_embeddings = nn.Embedding(3, d_model)

        # Positional embeddings
        # we provision a reasonably large max length
        self.max_positions = 256
        self.pos_embeddings = nn.Embedding(self.max_positions, d_model)

        # Joint Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, L, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier head
        self.classifier = nn.Linear(d_model, n_classes)
        self.bce = nn.BCEWithLogitsLoss()

        self.max_text_length = max_text_length
        self.pool_type = pool_type

    # helpers

    def _tokenize_batch(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def encode_text_tokens(self, batch):
        """
        Returns:
          text_tokens: [B, L, d_model]
          text_mask:   [B, L] (True for real tokens, False for padding)
        """
        if "input_ids" in batch and "attention_mask" in batch:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
        else:
            if "text" not in batch:
                return None, None
            texts = batch["text"]
            input_ids, attn_mask = self._tokenize_batch(texts)

        out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state  # [B, L, bert_dim]
        tokens = self.text_proj(hidden)  # [B, L, d_model]

        # text_mask: True for non-pad tokens
        text_mask = attn_mask.bool()  # [B, L]
        return tokens, text_mask

    def encode_image_token(self, batch):
        """
        Returns:
          img_token: [B, 1, d_model] or None
        """
        if self.image_encoder is None:
            return None
        img = batch.get("image", None)
        if img is None:
            return None
        feats = self.image_encoder(img)  # [B, feat_dim] (global pooled)
        img_token = self.image_proj(feats)  # [B, d_model]
        img_token = img_token.unsqueeze(1)  # [B, 1, d_model]
        return img_token

    def encode_graph_token(self, batch):
        """
        Returns:
          graph_token: [B, 1, d_model] or None
        """
        if self.graph_proj is None:
            return None
        graph_emb = batch.get("graph_emb", None)
        if graph_emb is None:
            return None
        g = self.graph_proj(graph_emb)  # [B, d_model]
        g = g.unsqueeze(1)              # [B, 1, d_model]
        return g

    def _get_targets(self, batch):
        if "labels" in batch:
            return batch["labels"].float()
        if "label" in batch:
            return batch["label"].float()
        raise KeyError("Batch must contain 'labels' or 'label'.")

    # Core forward 

    def build_sequence(self, batch):
        """
        Build multimodal sequence:
          [graph_token] + [image_token] + [text_tokens...]

        Returns:
          seq:        [B, L_total, d_model]
          attn_mask:  [B, L_total] (True for real tokens, False for padding)
        """
        B = None

        # Text
        text_tokens, text_mask = self.encode_text_tokens(batch)  # [B, L_text, d_model], [B, L_text]
        if text_tokens is None:
            raise ValueError("Text is required for VisualGenomeViLBERT.")
        B, L_text, _ = text_tokens.shape

        # Image token
        img_token = self.encode_image_token(batch)  # [B, 1, d_model] or None

        # Graph token
        graph_token = self.encode_graph_token(batch)  # [B, 1, d_model] or None

        seq_parts = []
        modality_ids = []

        # Graph token first (if available)
        if graph_token is not None:
            seq_parts.append(graph_token)
            modality_ids.append(torch.full((B, 1), 0, dtype=torch.long, device=self.device))

        # Image token second (if available)
        if img_token is not None:
            seq_parts.append(img_token)
            modality_ids.append(torch.full((B, 1), 1, dtype=torch.long, device=self.device))

        # Text tokens
        seq_parts.append(text_tokens)
        modality_ids.append(torch.full((B, L_text), 2, dtype=torch.long, device=self.device))

        # Concatenate along sequence dimension
        seq = torch.cat(seq_parts, dim=1)              # [B, L_total, d_model]
        modality_ids = torch.cat(modality_ids, dim=1)  # [B, L_total]

        L_total = seq.size(1)
        if L_total > self.max_positions:
            L_total = self.max_positions
            seq = seq[:, :L_total, :]
            modality_ids = modality_ids[:, :L_total]
            # text_mask also needs to be trimmed appropriately
            # but we only need attention mask for padding tokens,
            # and text_mask only covers the last part of the sequence.
        # Build attention mask: all tokens from graph/image/text are real except padded text
        # Since we used tokenizer padding for text, we need an attn mask that matches concatenation:
        #   [graph,image] tokens are always valid
        #   text_mask for text part

        # base: all ones (True)
        attn_mask = torch.ones((B, seq.size(1)), dtype=torch.bool, device=self.device)

        # Last L_text positions of seq correspond to text tokens in our construction,
        # so we map the text_mask there.
        # (If seq has graph and image tokens, text starts later.)
        text_start = seq.size(1) - L_text
        attn_mask[:, text_start:] = text_mask

        # Positional embeddings
        positions = torch.arange(seq.size(1), device=self.device).unsqueeze(0)  # [1, L_total]
        pos_emb = self.pos_embeddings(positions)                                # [1, L_total, d_model]

        # Modality embeddings
        mod_emb = self.modality_embeddings(modality_ids)                        # [B, L_total, d_model]

        seq = seq + pos_emb + mod_emb

        return seq, attn_mask

    def forward(self, batch):
        """
        Returns:
          pooled: [B, d_model]
          logits: [B, n_classes]
        """
        seq, attn_mask = self.build_sequence(batch)  # [B, L, d_model], [B, L]

        # Transformer expects key_padding_mask: True for PAD tokens
        key_padding_mask = ~attn_mask  # invert

        enc_out = self.encoder(seq, src_key_padding_mask=key_padding_mask)  # [B, L, d_model]

        if self.pool_type == "cls":
            # Use the first token as 'CLS' (graph if present, else image, else first text)
            pooled = enc_out[:, 0, :]  # [B, d_model]
        else:
            # mean over non-padding tokens
            mask = attn_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (enc_out * mask).sum(dim=1)    # [B, d_model]
            denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
            pooled = summed / denom

        logits = self.classifier(pooled)  # [B, n_classes]
        return pooled, logits

    # Training & validation steps 

    def training_step(self, batch, batch_idx):
        targets = self._get_targets(batch)
        pooled, logits = self.forward(batch)
        loss = self.bce(logits, targets)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == targets).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = self._get_targets(batch)
        pooled, logits = self.forward(batch)
        loss = self.bce(logits, targets)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
