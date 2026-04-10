from __future__ import annotations

import os

import lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import torch

from ..utils.config import resolve_repo_path

class Trainer:
    def __init__(
        self,
        max_epochs: int,
        gradient_clip_val: float,
        log_save_fold: str,
        ckpt_save_fold: str,
        ckpt_load_path: str | None = None,
        resume: bool = False,
    ) -> None:
        self.ckpt_load_path = ckpt_load_path
        self.resume = resume
        self.log_save_fold = resolve_repo_path(log_save_fold)
        self.ckpt_save_fold = resolve_repo_path(ckpt_save_fold)
        self.log_save_fold.mkdir(parents=True, exist_ok=True)
        self.ckpt_save_fold.mkdir(parents=True, exist_ok=True)

        logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=str(self.log_save_fold),
            name="",
            version="",
        )
        checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(self.ckpt_save_fold),
            filename="best",
            monitor="loss/valid",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(
            logging_interval="epoch"
        )

        self.trainer = lightning.Trainer(
            precision=32,
            deterministic=True,
            gradient_clip_val=gradient_clip_val,
            max_epochs=max_epochs,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint, lr_monitor],
            benchmark=True,
        )

    def fit(
        self,
        objective,
        datamodule: lightning.LightningDataModule,
    ) -> None:
        if self.ckpt_load_path is not None and not self.resume:
            self.load_model_weights_(objective.model, self.ckpt_load_path)
        self.trainer.fit(
            model=objective,
            datamodule=datamodule,
            ckpt_path=self.ckpt_load_path if self.resume else None,
        )

    @staticmethod
    def find_checkpoint(ckpt_load_fold: str, stem: str = "last") -> str:
        checkpoint_dir = resolve_repo_path(ckpt_load_fold)
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(stem) and filename.endswith(".ckpt"):
                return str(checkpoint_dir / filename)
        raise FileNotFoundError(f"Could not find checkpoint with stem '{stem}' in {checkpoint_dir}")

    @staticmethod
    def load_model_weights_(
        model: torch.nn.Module,
        ckpt_load_path: str,
    ) -> torch.nn.Module:
        checkpoint = torch.load(resolve_repo_path(ckpt_load_path), map_location=torch.device("cpu"))
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {
            key.replace("model.", ""): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if not model_state:
            model_state = state_dict
        model.load_state_dict(model_state, strict=False)
        return model
