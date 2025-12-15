# src/comparative/utils/paths.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] # 2, adjested for checking_label_balance_and_null_values.py
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "src" / "comparative" / "checkpoints"

def ensure_dirs():
    for p in [DATA_RAW, DATA_PROCESSED, RESULTS, CHECKPOINTS]:
        p.mkdir(parents=True, exist_ok=True)
