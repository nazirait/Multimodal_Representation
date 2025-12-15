# temp, split
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparative.utils.paths import DATA_PROCESSED

PROCESSED = DATA_PROCESSED / 'fashion'
# PROCESSED = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/processed/fashion')

df = pd.read_csv(PROCESSED / 'train.csv')

# Remove labels with fewer than 2 examples
counts = df['label'].value_counts()
sufficient = counts[counts >= 2].index
df_filtered = df[df['label'].isin(sufficient)]

print(f"Original train set size: {len(df)}")
print(f"Filtered train set size: {len(df_filtered)} (after removing rare classes)")

# Stratified split: 90% train, 10% test
new_train_df, test_df = train_test_split(
    df_filtered,
    test_size=0.1,
    random_state=42,
    stratify=df_filtered['label']
)

# Save new files
new_train_df.to_csv(PROCESSED / 'train.csv', index=False)
test_df.to_csv(PROCESSED / 'test.csv', index=False)

print(f"New train set: {len(new_train_df)}")
print(f"New test set: {len(test_df)}")
print("Split complete! 🚦")
