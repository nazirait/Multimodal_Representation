# src/comparative/datasets/amazon_datamodule.py

from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class AmazonReviewDataset(Dataset):
    """
    Emits BOTH:
      - pre-tokenized tensors (for BERT-style baselines), AND
      - raw 'text' string (so CLIP can tokenize with its own vocab and max length 77).

    Labels:
      - Source CSV has {1,2}. We convert to {0,1}.
      - We include both 'label' (int) and 'labels' (LongTensor[ ]) for convenience.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer_name: str,
        max_len: int,
        n_samples: Optional[int] = None,
    ):
        df = pd.read_csv(csv_path)
        if n_samples is not None:
            df = df.iloc[:n_samples].reset_index(drop=True)

        self.texts = df["full_text"].astype(str).tolist()
        # map {1,2} -> {0,1}
        self.labels_int = (df["label"].astype(int) - 1).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = int(self.labels_int[idx])

        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}

        # provide labels in both common formats
        item["label"] = torch.tensor(label, dtype=torch.long)
        item["labels"] = item["label"]

        # for CLIP module to tokenize with CLIP's vocab 
        item["text"] = text

        return item


class AmazonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        max_len: int = 128,
        num_workers: int = 12,
        train_samples: int = 100_000,
        val_samples: int = 20_000,
        test_samples: int = 20_000,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def setup(self, stage=None):
        self.train_ds = AmazonReviewDataset(
            self.data_dir / "train.csv",
            self.tokenizer_name,
            self.max_len,
            n_samples=self.train_samples,
        )
        # if val.csv is available, otherwise a slice of train.csv
        self.val_ds = AmazonReviewDataset(
            self.data_dir / "train.csv",
            self.tokenizer_name,
            self.max_len,
            n_samples=self.val_samples,
        )
        self.test_ds = AmazonReviewDataset(
            self.data_dir / "test.csv",
            self.tokenizer_name,
            self.max_len,
            n_samples=self.test_samples,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
