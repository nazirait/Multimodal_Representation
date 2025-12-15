from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparative.utils.paths import DATA_PROCESSED
for ds in ['amazon', 'fashion', 'movielens', 'visualgenome']:
    proc = DATA_PROCESSED / ds
    # proc = Path(f'D:/COmparative_Study_of_Multimodal_Represenations/data/processed/{ds}')
    print(f"\n== {ds.upper()} ==")
    for f in proc.glob('*.csv'):
        df = pd.read_csv(f)
        print(f"{f.name}: {df.shape} cols={list(df.columns)}")
        print(df.head(2))
