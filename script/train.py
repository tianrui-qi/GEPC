import hydra
import hydra.utils
import lightning
import torch
from omegaconf import DictConfig

import src


torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../config", config_name="pipeline/train")
def main(cfg: DictConfig) -> None:
    lightning.seed_everything(cfg.seed, workers=True, verbose=False)

    data = src.DataModule(**cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    objective = src.Objective(model, **cfg.objective, **cfg.simulator)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(objective, data)


if __name__ == "__main__":
    main()
