from __future__ import annotations

import hydra
import lightning
import torch
from omegaconf import DictConfig

import src


torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig) -> None:
    cfg = src.resolve_runtime_paths(cfg)
    lightning.seed_everything(cfg.seed, workers=True, verbose=False)

    data = src.DataModule(**cfg.data)
    model = src.Model(**cfg.model)
    objective = src.ObjectivePretrain(model, **cfg.objective)
    trainer = src.Trainer(**cfg.trainer)

    trainer.fit(objective, datamodule=data)


if __name__ == "__main__": main()