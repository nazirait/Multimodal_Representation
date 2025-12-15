import pandas as pd
from pathlib import Path

# RAW = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/raw/amazon/amazon_review_sa_binary_csv')
# PROCESSED = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/processed/amazon')

from pathlib import Path
import sys

# make project importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparative.utils.paths import DATA_RAW, DATA_PROCESSED
RAW = DATA_RAW / 'amazon' / 'amazon_review_sa_binary_csv'
PROCESSED = DATA_PROCESSED / 'amazon'
PROCESSED.mkdir(parents=True, exist_ok=True)


for split in ['train', 'test']:
    df = pd.read_csv(RAW / f'{split}.csv')
    # Basic cleaning (strip, lower), remove nans
    df['review_text'] = df['review_text'].fillna('').astype(str).str.strip().str.lower()
    df['review_title'] = df['review_title'].fillna('').astype(str).str.strip().str.lower()
    # Optional: combine title + text
    df['full_text'] = (df['review_title'] + '. ' + df['review_text']).str.strip()
    # Ensure binary labels
    df['label'] = df['class_index'].astype(int)
    df_out = df[['full_text', 'label']]
    df_out.to_csv(PROCESSED / f'{split}.csv', index=False)
print("Amazon preprocessing is done")
