# Comparative Study of Multimodal Representations

This repository contains the code, data processing pipelines, experiments, and analysis for a MSc thesis on multimodal representation learning. The work compares a range of fusion strategies and model families across multiple datasets and evaluates their effectiveness for retrieval, recommendation and representation analysis.

**Note:** This project is part of the author's Master's thesis (repository owner: `nazirait`). If you use this work, please cite the thesis and refer to `CITATION.cff` for citation metadata.

## Abstract

This study performs a systematic comparison of multimodal representation strategies, including classical fusion baselines, transformer-based models, and generative latent-variable approaches. The goal is to evaluate how different methods combine text, image, and graph modalities across several benchmark datasets and to provide reproducible code and analysis used in the thesis.

## Key Contributions

- A reproducible codebase implementing diverse multimodal architectures (early/late fusion baselines, CLIP, ViLBERT-style transformers, and MVAE-like models).
- Evaluation scripts and analysis notebooks to compare retrieval and representation quality across datasets.
- Data processing pipelines and configs to reproduce the experiments reported in the thesis.

## Repository Structure

- `data/` — processed datasets and raw downloads. Processed splits are under `data/processed/` for `amazon`, `fashion`, `movielens`, and `visualgenome`.
- `notebooks/` — exploratory and analysis notebooks used during development and for figures.
- `scripts/` — helper scripts for preprocessing and sanity checks (e.g. `preprocess_*`, `check_cuda.py`).
- `src/comparative/` — primary source code for datasets, models, training, evaluation and utilities.
  - `src/comparative/configs/` — experiment and model configuration files.
  - `src/comparative/datasets/` — datamodules and dataset loaders.
  - `src/comparative/models/` — model implementations (classical, transformers, VAE).
  - `src/comparative/training/` — training scripts, CLI and callbacks.
  - `src/comparative/evaluation/` — metrics, retrieval and visualization utilities.
- `CITATION.cff` — citation metadata for the project.
- `LICENSE` — project license.

## Requirements

Dependencies are listed in `requirements.txt` and `pyproject.toml`. Create a Python environment and install dependencies with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On systems with a GPU, ensure CUDA drivers are installed and test with `python scripts/check_cuda.py`.

## Quick Start

1. Prepare data: see `data/DATA_README.md` and the scripts in `scripts/` for preprocessing steps. Large raw datasets are expected to be downloaded separately and placed under `data/raw/`.
2. Inspect or modify experiment configs in `src/comparative/configs/`.
3. Run training (example):

```powershell
python src/comparative/training/train.py --config src/comparative/configs/train/config.yaml
```

4. Evaluate models and reproduce results using scripts in `src/comparative/evaluation/` and notebooks in `notebooks/`.

Note: configuration parsing and CLI options are defined in `src/comparative/training/cli.py` and `train.py`. Adjust paths and config names according to your experiment.

## Datasets

The thesis experiments use the following datasets (preprocessed versions are included under `data/processed`):

- Amazon Review (text modality)
- FashionAI (text + image modalities)
- Movielens (text + graph modalities)
- Visual Genome (text + image + graph modalities)

For instructions on obtaining the raw data, see `data/datadownloads.md` and the dataset-specific READMEs inside `data/`.

## Experiments & Notebooks

Exploratory analysis and figures are available in the `notebooks/` folder. Reproduction of specific experiments requires pointing training and evaluation scripts to the appropriate processed dataset splits under `data/processed/`.

## Evaluation

Evaluation code (retrieval metrics, latent analysis, visualization) is available in `src/comparative/evaluation/`. Check `src/comparative/evaluation/metrics.py` and `retrieval.py` to see implemented metrics and evaluation flows.

## Reproducibility

- Config-driven experiments: most hyperparameters and dataset paths are configurable via YAML files in `src/comparative/configs/`.
- Random seeds and logging are handled in the training utilities and checkpointing in `src/comparative/checkpoints`.

## License & Citation

This repository is released under the license in `LICENSE`. If you use this code or results in your work, please cite the thesis and the project (see `CITATION.cff`).

## Contact

For questions about the code or thesis, open an issue or contact the repository owner `nazirait` on GitHub.

---

Last updated: December 10, 2025
