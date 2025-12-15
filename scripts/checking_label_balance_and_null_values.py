import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparative.utils.paths import DATA_PROCESSED

# --- datasets
print("-" * 20, "AMAZON", "-" * 20)

amazon_dir = DATA_PROCESSED / "amazon"

df_train = pd.read_csv(amazon_dir / "train.csv")
df_test  = pd.read_csv(amazon_dir / "test.csv")
df_val   = pd.read_csv(amazon_dir / "val.csv")

print("Train label counts:\n", df_train["label"].value_counts(), "\n")
print("Test label counts:\n", df_test["label"].value_counts(), "\n")
print("Validation label counts:\n", df_val["label"].value_counts(), "\n")

print("Train missing values:\n", df_train.isnull().sum(), "\n")
print("Test missing values:\n", df_test.isnull().sum(), "\n")
print("Validation missing values:\n", df_val.isnull().sum(), "\n")


print("-" * 20, "FASHION", "-" * 20)

fashion_dir = DATA_PROCESSED / "fashion"

df_fashion_train = pd.read_csv(fashion_dir / "train.csv")
df_fashion_val   = pd.read_csv(fashion_dir / "val.csv")

print("Fashion Train label counts:\n", df_fashion_train["label"].value_counts(), "\n")
print("Fashion Val label counts:\n", df_fashion_val["label"].value_counts(), "\n")

print("Fashion Train missing values:\n", df_fashion_train.isnull().sum(), "\n")
print("Fashion Val missing values:\n", df_fashion_val.isnull().sum(), "\n")


print("-" * 20, "MOVIELENS", "-" * 20)

movielens_dir = DATA_PROCESSED / "movielens"

df_ml_train = pd.read_csv(movielens_dir / "train.csv")
df_ml_val   = pd.read_csv(movielens_dir / "val.csv")
df_ml_test  = pd.read_csv(movielens_dir / "test.csv")

print("Genres (train, top 10):\n",
      df_ml_train["genres"].value_counts().head(10), "\n")

print("Movielens Train missing values:\n", df_ml_train.isnull().sum(), "\n")
print("Movielens Val missing values:\n", df_ml_val.isnull().sum(), "\n")
print("Movielens Test missing values:\n", df_ml_test.isnull().sum(), "\n")


print("-" * 20, "VISUAL GENOME", "-" * 20)

vg_dir = DATA_PROCESSED / "visualgenome"

vg_dataset_path = vg_dir / "visualgenome_dataset.jsonl"
vg_stats_path   = vg_dir / "visualgenome_stats.json"

# load stast
import json
with open(vg_stats_path, "r") as f:
    vg_stats = json.load(f)

print("Visual Genome stats:")
for k, v in vg_stats.items():
    print(f"{k}: {v}")


# dataset structure (JSONL)
vg_df = pd.read_json(vg_dataset_path, lines=True)

print("VG Overview:")
print("Number of samples:", len(vg_df))
print("Cols:", vg_df.columns.tolist())

print("Missing values per column:")
print(vg_df.isnull().sum())

