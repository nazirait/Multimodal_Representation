# src/comparative/utils/paths.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "src" / "comparative" / "checkpoints"

def ensure_dirs():
    for p in [DATA_RAW, DATA_PROCESSED, RESULTS, CHECKPOINTS]:
        p.mkdir(parents=True, exist_ok=True)
