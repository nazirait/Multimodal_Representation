from pathlib import Path
import pandas as pd

for ds in ['amazon', 'fashion', 'movielens', 'visualgenome']:
    proc = Path(f'D:/COmparative_Study_of_Multimodal_Represenations/data/processed/{ds}')
    print(f"\n== {ds.upper()} ==")
    for f in proc.glob('*.csv'):
        df = pd.read_csv(f)
        print(f"{f.name}: {df.shape} cols={list(df.columns)}")
        print(df.head(2))
