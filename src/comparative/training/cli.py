# src/comparative/training/cli.py

import fire
from comparative.training.train import main as train_main

def train(**overrides):
    """Run training with command-line overrides."""
    train_main(overrides)

if __name__ == "__main__":
    fire.Fire({
        "train": train,
    })
