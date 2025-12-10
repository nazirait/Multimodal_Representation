# src/comparative/datasets/movielens_datamodule.py

import math
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
from typing import Dict, Optional


class MovieLensDataset(Dataset):
    """
    Tokenized text + multi-hot labels.
    Optionally attaches:
      - tabular: genome vectors per movie_idx
      - graph / graph_emb: graph embeddings per movie_idx
      - text: raw text string (for CLIP / ViLBERT)
      - labels: alias of label (for modules that expect 'labels')
      - image_emb: generic second-modality embedding
                  (for CLIPDualEncoderFromEmbeds / ViLBERTEmbedClassifier)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        genre_list,
        max_len: int = 128,
        model_name: str = "distilbert-base-uncased",
        tabular_map: Optional[Dict[int, np.ndarray]] = None,
        graph_map: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["genres"] = self.df["genres"].fillna("").astype(str)
        self.df["tag"] = self.df["tag"].fillna("").astype(str)

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.genre_list = list(genre_list)
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}

        self.tabular_map = tabular_map
        self.graph_map = graph_map

    def __len__(self):
        return len(self.df)

    def _encode_labels(self, genre_str: str) -> torch.Tensor:
        vec = torch.zeros(len(self.genre_list), dtype=torch.float)
        for g in genre_str.split("|"):
            g = g.strip()
            if g in self.genre_to_idx:
                vec[self.genre_to_idx[g]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # text view
        text = f"{row['title']}. {row['genres']}. {row['tag']}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # multilabel genres
        y = self._encode_labels(row["genres"])

        item = {
            # classical DistilBERT models
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),

            # labels for classical models
            "label": y,

            # aliases for CLIP / ViLBERT etc.
            "labels": y,   # (B, C) multi-hot
            "text": text,  # raw string
        }

        mid = int(row["movie_idx"])

        # we'll accumulate a general-purpose "side embedding"
        image_emb = None

        # tabular (genome vectors): optional
        if self.tabular_map is not None:
            vec = self.tabular_map.get(mid, None)
            if vec is not None:
                tab = torch.from_numpy(vec).float()
                item["tabular"] = tab
                image_emb = tab if image_emb is None else torch.cat([image_emb, tab], dim=-1)

        # graph embeddings (optional)
        if self.graph_map is not None:
            gvec = self.graph_map.get(mid, None)
            if gvec is not None:
                gt = torch.from_numpy(gvec).float()
                # for classical fusion baselines
                item["graph"] = gt
                item["graph_emb"] = gt
                # and for CLIP/ViLBERT second modality
                image_emb = gt if image_emb is None else torch.cat([image_emb, gt], dim=-1)

        # generic second modality
        if image_emb is not None:
            # this will be consumed by CLIPDualEncoderFromEmbeds / ViLBERTEmbedClassifier
            item["image_emb"] = image_emb

        return item


class MovieLensDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size: int = 64,
        max_len: int = 128,
        model_name: str = "distilbert-base-uncased",
        num_workers: int = 12,
        # Balanced downsampling knobs:
        limit_train_samples: int | None = None,
        limit_val_samples: int | None = None,
        balance_by_genre: bool = True,
        seed: int = 42,
        # NEW: modality knobs
        use_genome_vectors: bool = False,
        genome_vectors_path: Optional[str] = None,   # .csv or .npy
        use_movie_graph_emb: bool = False,
        movie_graph_emb_path: Optional[str] = None,  # .csv or .npy
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.model_name = model_name
        self.num_workers = num_workers

        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.balance_by_genre = balance_by_genre
        self.seed = seed

        self.use_genome_vectors = use_genome_vectors
        self.genome_vectors_path = Path(genome_vectors_path) if genome_vectors_path else None
        self.use_movie_graph_emb = use_movie_graph_emb
        self.movie_graph_emb_path = Path(movie_graph_emb_path) if movie_graph_emb_path else None

        self.genre_list = None
        self.tabular_map: Optional[Dict[int, np.ndarray]] = None
        self.graph_map: Optional[Dict[int, np.ndarray]] = None

    # helperss
    def _load_df(self, split_name: str) -> pd.DataFrame:
        # include movie_idx so we can join vectors
        cols = ["movie_idx", "title", "genres", "tag"]
        return pd.read_csv(self.data_dir / f"{split_name}.csv", usecols=cols)

    def _build_genre_list(self):
        all_genres = set()
        for split in ["train", "val", "test"]:
            df = self._load_df(split)
            for gs in df["genres"].fillna("").astype(str):
                all_genres.update([g.strip() for g in gs.split("|") if g])
        genre_list = sorted(list(all_genres))
        print(
            f"[MovieLensDataModule] Using {len(genre_list)} genre classes: {genre_list}"
        )
        return genre_list

    def _balanced_subset_multilabel(self, df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        df = df.copy()
        df["genres"] = df["genres"].fillna("").astype(str)
        glist = df["genres"].map(lambda s: [g.strip() for g in s.split("|") if g])
        keep = glist.map(len) > 0
        df = df[keep].reset_index(drop=True)
        glist = glist[keep].reset_index(drop=True)

        avg_labels = float(glist.map(len).mean())
        if avg_labels == 0:
            raise ValueError("No labels found for balancing.")
        n_classes = len(self.genre_list)
        per_target = int(math.ceil(target_rows * avg_labels / n_classes))

        counts = {g: 0 for g in self.genre_list}
        chosen = []
        idxs = np.arange(len(df))
        rng.shuffle(idxs)
        for i in idxs:
            gs = [g for g in glist.iloc[i] if g in counts]
            if not gs:
                continue
            if any(counts[g] < per_target for g in gs):
                chosen.append(i)
                for g in gs:
                    if counts[g] < per_target:
                        counts[g] += 1
                if min(counts.values()) >= per_target:
                    break
            if len(chosen) >= target_rows:
                break

        sub = (
            df.iloc[chosen]
            .sample(frac=1.0, random_state=self.seed)
            .reset_index(drop=True)
        )

        ach = {g: 0 for g in self.genre_list}
        for s in sub["genres"]:
            for g in s.split("|"):
                g = g.strip()
                if g in ach:
                    ach[g] += 1
        print(
            f"[MovieLensDataModule] Balanced subset: rows={len(sub)} | "
            f"avg_labels={avg_labels:.2f} | per_genre_target={per_target} | "
            f"min/max per-genre achieved=({min(ach.values())}, {max(ach.values())})"
        )
        return sub

    def _load_movie2idx(self) -> Optional[pd.DataFrame]:
        p = self.data_dir / "movie2idx.csv"
        if p.exists():
            return pd.read_csv(p)  # columns: movieId, movie_idx
        return None

    def _to_idx_keyed_map(self, df: pd.DataFrame, key_col: str) -> Dict[int, np.ndarray]:
        # keep only numeric feature columns
        num = df.select_dtypes(include=[np.number]).copy()
        if key_col not in num.columns:
            # bring key col from original df if numeric
            if key_col in df.columns:
                num[key_col] = df[key_col].astype(int)
            else:
                raise ValueError(f"Key column {key_col} not found.")
        feat_cols = [c for c in num.columns if c != key_col]
        return {
            int(k): row[feat_cols].to_numpy(dtype=np.float32)
            for k, row in num.set_index(key_col).iterrows()
        }

    def _load_vectors_csv_or_npy(
        self, path: Path, key_hint: str, movie2idx: Optional[pd.DataFrame]
    ) -> Dict[int, np.ndarray]:
        if path.suffix.lower() == ".npy":
            mat = np.load(path)  # assume index == movie_idx
            return {int(i): mat[i].astype(np.float32) for i in range(mat.shape[0])}
        # CSV case
        df = pd.read_csv(path)
        # detect key column
        key_col = None
        for candidate in ["movie_idx", "movieId", "mid", "id"]:
            if candidate in df.columns:
                key_col = candidate
                break
        if key_col is None:
            raise ValueError(
                f"Could not find a movie key column in {path}. "
                "Expected one of ['movie_idx','movieId','mid','id']."
            )
        if key_col != "movie_idx":
            # map to movie_idx if possible
            if movie2idx is None:
                raise ValueError(
                    f"Vectors use '{key_col}', but movie2idx.csv not found to map to movie_idx."
                )
            df = df.merge(
                movie2idx,
                on=key_col if key_col == "movieId" else key_col,
                how="inner",
            )
            key_col = "movie_idx"
        return self._to_idx_keyed_map(df, key_col)

    # pl.LightningDataModule API 
    def setup(self, stage=None):
        self.genre_list = self._build_genre_list()
        movie2idx = self._load_movie2idx()

        # Load splits
        train_df = self._load_df("train")
        val_df = self._load_df("val")
        test_df = self._load_df("test")

        # Balanced downsampling
        if self.balance_by_genre and self.limit_train_samples:
            train_df = self._balanced_subset_multilabel(
                train_df, self.limit_train_samples
            )
        if self.balance_by_genre and self.limit_val_samples:
            val_df = self._balanced_subset_multilabel(
                val_df, self.limit_val_samples
            )

        # Optional: load tabular + graph maps
        self.tabular_map = None
        self.graph_map = None
        if self.use_genome_vectors and self.genome_vectors_path is not None:
            self.tabular_map = self._load_vectors_csv_or_npy(
                self.genome_vectors_path, "genome", movie2idx
            )
            any_vec = next(iter(self.tabular_map.values()))
            print(
                f"[MovieLensDataModule] Loaded genome vectors: "
                f"dim={any_vec.shape[0]} entries={len(self.tabular_map)}"
            )

        if self.use_movie_graph_emb and self.movie_graph_emb_path is not None:
            self.graph_map = self._load_vectors_csv_or_npy(
                self.movie_graph_emb_path, "graph", movie2idx
            )
            any_g = next(iter(self.graph_map.values()))
            print(
                f"[MovieLensDataModule] Loaded graph embeddings: "
                f"dim={any_g.shape[0]} entries={len(self.graph_map)}"
            )

        # Build datasets
        self.train_ds = MovieLensDataset(
            df=train_df,
            genre_list=self.genre_list,
            max_len=self.max_len,
            model_name=self.model_name,
            tabular_map=self.tabular_map,
            graph_map=self.graph_map,
        )
        self.val_ds = MovieLensDataset(
            df=val_df,
            genre_list=self.genre_list,
            max_len=self.max_len,
            model_name=self.model_name,
            tabular_map=self.tabular_map,
            graph_map=self.graph_map,
        )
        self.test_ds = MovieLensDataset(
            df=test_df,
            genre_list=self.genre_list,
            max_len=self.max_len,
            model_name=self.model_name,
            tabular_map=self.tabular_map,
            graph_map=self.graph_map,
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
