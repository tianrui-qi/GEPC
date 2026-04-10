from __future__ import annotations

import lightning
import torch


class ObjectivePretrain(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        # optimizer
        lr: float,
        weight_decay: float,
        # scheduler
        step_size: int,
        gamma: float,
        # metric
        scale: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.scale = scale
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        past: torch.Tensor,
        future_stim: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(past, future_stim)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        prediction = self(batch["past"], batch["future_stim"])
        target = batch["target"]
        loss = torch.nn.functional.mse_loss(prediction, target)
        rmse = torch.sqrt(
            torch.nn.functional.mse_loss(
                prediction * self.scale,
                target * self.scale,
            )
        )
        mae = torch.mean(torch.abs(prediction - target)) * self.scale

        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, logger=True)
        self.log(f"rmse/{stage}", rmse, on_step=False, on_epoch=True, logger=True)
        self.log(f"mae/{stage}", mae, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["past"], batch["future_stim"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
