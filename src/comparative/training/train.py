# src/comparative/training/train.py

from __future__ import annotations
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
import pytorch_lightning as pl

from comparative.training.callbacks import basic_callbacks
from comparative.utils.seed import seed_everything

@hydra.main(config_path="D:/COmparative_Study_of_Multimodal_Represenations/src/comparative/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    rprint("[bold green]Hydra config resolved:[/]")
    rprint(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))

    # Set global seed for reproducibility
    seed_everything(cfg.get("seed", 42))

    # Instantiate DataModule directly with hydra (no registry)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # Lightning Trainer (Hydra can override callbacks/strategy, etc.)
    trainer = hydra.utils.instantiate(
        cfg.train,
        callbacks=basic_callbacks(cfg),
    )
    rprint(f"[cyan]Using Trainer => {trainer.accelerator} | devices={trainer.num_devices} | precision={trainer.precision}[/]")

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
