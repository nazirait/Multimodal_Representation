from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class FashionDataset(Dataset):
    """
    Dataset for Fashion *without raw images*.

    Each sample returns:
      {
        "text": str,                             # raw description (for CLIP text encoder)
        "image_emb": torch.FloatTensor(512,),    # precomputed image embedding
        "label": LongTensor([]),                 # class index
        "labels": LongTensor([]),                # alias
        "input_ids": LongTensor[max_len],        # optional for text-only heads
        "attention_mask": LongTensor[max_len],
        "token_type_ids": (optional)
      }
    """

    def __init__(
        self,
        csv_path: Path,
        image_emb_path: Path,
        tokenizer_name: str,
        max_len: int,
        label2idx: Dict[str, int],
        n_samples: Optional[int] = None,
    ):
        df = pd.read_csv(csv_path)

        # load precomputed vision embeddings aligned by row index
        img_emb_arr = np.load(image_emb_path)  # shape [N, 512]
        assert len(df) == img_emb_arr.shape[0], (
            f"row count mismatch: {csv_path} has {len(df)} rows but {image_emb_path} "
            f"has {img_emb_arr.shape[0]} embeddings"
        )

        if n_samples is not None:
            df = df.iloc[:n_samples].reset_index(drop=True)
            img_emb_arr = img_emb_arr[:n_samples]

        self.texts: List[str] = df["description"].fillna("").astype(str).tolist()
        raw_labels = df["label"].astype(str).tolist()

        self.label2idx = label2idx
        self.labels_idx: List[int] = [self.label2idx[lbl] for lbl in raw_labels]

        self.img_emb = torch.from_numpy(img_emb_arr).float()  # (N, 512)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        img_vec = self.img_emb[idx]  # (512,)

        label_id = self.labels_idx[idx]
        label_t = torch.tensor(label_id, dtype=torch.long)

        # tokenized text for (optional) supervised heads
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        sample = {
            "text": text,
            "image_emb": img_vec,  # already a float tensor
            "label": label_t,
            "labels": label_t,
            **enc,
        }
        return sample


class FashionDatamodule(pl.LightningDataModule):
    """
    Produces batches like:
      {
        "text":        [str, str, ...]          # len B
        "image_emb":   FloatTensor[B,512]
        "label":       LongTensor[B]
        "labels":      LongTensor[B]
        "input_ids":         LongTensor[B,max_len]
        "attention_mask":    LongTensor[B,max_len]
        "token_type_ids":    LongTensor[B,max_len] (if provided by tokenizer)
      }

    This now supports:
      - CLIPDualEncoderFromEmbeds (new)
        which will align CLIP text embeddings with precomputed visual embeddings.
    """

    def __init__(
        self,
        data_dir,
        tokenizer_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        max_len: int = 128,
        img_size: int = 224,  # kept for hydra compatibility, unused now
        num_workers: int = 12,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.img_size = img_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples

        self.label2idx: Dict[str, int] = {}

    def setup(self, stage=None):
        train_csv = self.data_dir / "train.csv"
        val_csv   = self.data_dir / "val.csv"
        test_csv  = self.data_dir / "test.csv"

        train_img_emb = self.data_dir / "train_image_emb.npy"
        val_img_emb   = self.data_dir / "val_image_emb.npy"
        test_img_emb  = self.data_dir / "test_image_emb.npy"

        # build global label map
        all_labels = []
        for fname in ["train.csv", "val.csv", "test.csv"]:
            f = self.data_dir / fname
            if f.exists():
                df_tmp = pd.read_csv(f)
                all_labels.extend(df_tmp["label"].astype(str).tolist())
        unique_classes = sorted(set(all_labels))
        self.label2idx = {lbl: i for i, lbl in enumerate(unique_classes)}

        self.train_ds = FashionAIDataset(
            csv_path=train_csv,
            image_emb_path=train_img_emb,
            tokenizer_name=self.tokenizer_name,
            max_len=self.max_len,
            label2idx=self.label2idx,
            n_samples=self.train_samples,
        )

        self.val_ds = FashionAIDataset(
            csv_path=val_csv,
            image_emb_path=val_img_emb,
            tokenizer_name=self.tokenizer_name,
            max_len=self.max_len,
            label2idx=self.label2idx,
            n_samples=self.val_samples,
        )

        if test_csv.exists() and test_img_emb.exists():
            self.test_ds = FashionAIDataset(
                csv_path=test_csv,
                image_emb_path=test_img_emb,
                tokenizer_name=self.tokenizer_name,
                max_len=self.max_len,
                label2idx=self.label2idx,
                n_samples=None,
            )
        else:
            self.test_ds = None

        print(f"train size: {len(self.train_ds)} classes: {len(self.label2idx)}")
        print(f"val size:   {len(self.val_ds)}")

    def _collate_fn(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts        = [b["text"] for b in batch_list]                                # list[str]
        img_embs     = torch.stack([b["image_emb"] for b in batch_list], dim=0)       # (B,512)
        labels       = torch.stack([b["label"] for b in batch_list], dim=0)           # (B,)

        input_ids        = torch.stack([b["input_ids"] for b in batch_list], dim=0)
        attention_mask   = torch.stack([b["attention_mask"] for b in batch_list], dim=0)
        token_type_ids = (
            torch.stack([b["token_type_ids"] for b in batch_list], dim=0)
            if "token_type_ids" in batch_list[0]
            else None
        )

        out_batch = {
            "text": texts,
            "image_emb": img_embs,
            "label": labels,
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            out_batch["token_type_ids"] = token_type_ids

        return out_batch

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
