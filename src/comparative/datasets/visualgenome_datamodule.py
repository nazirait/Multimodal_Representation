# src/comparative/datasets/visualgenome_datamodule.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


# dataset
class VisualGenomeDataset(Dataset):
    """
    Wraps the preprocessed JSONL.

    Returns:
      {
        "image": tensor,
        "text": caption string,
        "labels": multi-hot tensor
        "image_id": int
        "graph": tensor(graph_dim)
      }
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        label_to_idx: Dict[str, int],
        project_root: Path,
        image_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.examples = examples
        self.label_to_idx = label_to_idx
        self.project_root = Path(project_root)
        self.image_transform = image_transform

        self.num_classes = len(label_to_idx)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        # img
        rel_path = ex["image_path"]
        img_path = self.project_root / rel_path
        image = Image.open(img_path).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        # text
        caption = ex["caption"]

        # labels
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in ex["object_labels"]:
            j = self.label_to_idx.get(label)
            if j is not None:
                labels[j] = 1.0

        # graph features
        # preprocessing created "graph_features"
        graph_features = ex.get("graph_features")
        if graph_features is not None:
            graph = torch.tensor(graph_features, dtype=torch.float32)
        else:
            graph = torch.zeros(50, dtype=torch.float32)  # fallback

        return {
            "image": image,
            "text": caption,
            "labels": labels,
            "image_id": ex["image_id"],
            "graph": graph,         # renamed from graph_emb → graph
        }


# data module
class VisualGenomeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str = "data/processed/visualgenome/visualgenome_dataset.jsonl",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        min_label_freq: int = 5,
        max_num_labels: Optional[int] = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.min_label_freq = min_label_freq
        self.max_num_labels = max_num_labels
        self.pin_memory = pin_memory

        self.project_root: Optional[Path] = None
        self.label_to_idx: Dict[str, int] = {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:

        if self.train_dataset is not None:
            return

        data_path = Path(self.data_path).resolve()
        self.project_root = data_path.parents[3]

        # load jsonl
        examples = []
        with data_path.open("r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # splitting
        train_examples = [ex for ex in examples if ex["split"] == "train"]
        val_examples   = [ex for ex in examples if ex["split"] == "val"]
        test_examples  = [ex for ex in examples if ex["split"] == "test"]

        # building label vocabulary from TRAIN only 
        label_counter = Counter()
        for ex in train_examples:
            for l in ex["object_labels"]:
                label_counter[l] += 1

        kept = [lbl for lbl, c in label_counter.items() if c >= self.min_label_freq]

        if self.max_num_labels is not None:
            kept = [lbl for lbl, _ in label_counter.most_common(self.max_num_labels)]

        kept = sorted(set(kept))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(kept)}

        print(f"[VisualGenomeDataModule] num_classes = {len(self.label_to_idx)}")

        # filter examples
        def filter_and_fix(exs):
            out = []
            for ex in exs:
                labs = [l for l in ex["object_labels"] if l in self.label_to_idx]
                if not labs:
                    continue
                ex2 = dict(ex)
                ex2["object_labels"] = labs
                out.append(ex2)
            return out

        train_examples = filter_and_fix(train_examples)
        val_examples   = filter_and_fix(val_examples)
        test_examples  = filter_and_fix(test_examples)

        print(f"[VisualGenomeDataModule] After filtering: "
              f"train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}")

        # img transforms
        image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # create datasets
        self.train_dataset = VisualGenomeDataset(train_examples, self.label_to_idx, self.project_root, image_transform)
        self.val_dataset   = VisualGenomeDataset(val_examples,   self.label_to_idx, self.project_root, image_transform)
        self.test_dataset  = VisualGenomeDataset(test_examples,  self.label_to_idx, self.project_root, image_transform)

    # collate function (important)
    def collate_fn(self, batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        graphs = torch.stack([b["graph"] for b in batch])
        texts  = [b["text"] for b in batch]
        image_ids = [b["image_id"] for b in batch]

        return {
            "image": images,
            "text": texts,
            "labels": labels,
            "graph": graphs,
            "image_id": image_ids,
        }

    # data loaders
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
