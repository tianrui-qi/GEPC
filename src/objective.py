import math

import lightning
import torch


__all__ = ["Objective", "DifferentiableODE"]


class Objective(lightning.LightningModule):
    """
    Inverse-control training objective.

    The model predicts future light stimulus. Loss is computed by passing that
    predicted stimulus through a deterministic, differentiable PyTorch ODE and
    comparing the simulated fluorescence against the target response.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        # optimizer
        lr: float,
        weight_decay: float,
        # scheduler
        step_size: int,
        gamma: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.ode = DifferentiableODE(**kwargs)
        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        # scheduler
        self.step_size = step_size
        self.gamma = gamma

    def forward(
        self,
        past: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(past, future_target)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        past = batch["past"]
        future_target = batch["target"]
        cell_E = batch["cell_E"]

        stim_logits = self(past, future_target)
        stim_soft = torch.sigmoid(stim_logits)
        F_sim_au = self.ode(past, stim_soft, cell_E)

        F_sim_scaled = F_sim_au / self.ode.camera_max
        loss = torch.nn.functional.mse_loss(F_sim_scaled, future_target)

        with torch.no_grad():
            rmse = torch.sqrt(
                torch.nn.functional.mse_loss(
                    F_sim_au,
                    future_target * self.ode.camera_max,
                )
            )
            mae = torch.mean(
                torch.abs(F_sim_au - future_target * self.ode.camera_max)
            )

        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, logger=True)
        self.log(f"rmse/{stage}", rmse, on_step=False, on_epoch=True, logger=True)
        self.log(f"mae/{stage}", mae, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

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


class DifferentiableODE(torch.nn.Module):
    """
    Differentiable PyTorch CcaSR ODE layer.

    This layer converts predicted future light into predicted future
    fluorescence while preserving gradients. Data-pool simulation uses
    src.simulator.simulate instead.
    """

    def __init__(
        self,
        sampling: int,
        tau: float,
        eta: float,
        nu: float,
        a: float,
        k_h: float,
        n_h: float,
        camera_mult: float,
        camera_offset: float,
        camera_max: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tau_steps = max(1, math.ceil(float(tau) / float(sampling)))
        self.dt = float(sampling)
        self.eta = eta
        self.nu = nu
        self.a = a
        self.k_h = k_h
        self.n_h = n_h
        self.camera_mult = camera_mult
        self.camera_offset = camera_offset
        self.camera_max = camera_max

    def forward(
        self,
        past: torch.Tensor,
        stim: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        past_stim = past[:, :, 1]
        with torch.no_grad():
            H_init, _ = self.simulate_past(past_stim.detach(), E.detach())
        F_init = self.obs_to_molecules(past[:, -1, 0])
        past_stim_tail = past_stim[:, -self.tau_steps :]
        return self.simulate_future(stim, past_stim_tail, H_init, F_init, E)

    def step(
        self,
        H: torch.Tensor,
        F: torch.Tensor,
        U: torch.Tensor,
        E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dH = self.eta * U - self.nu * H
        dF = self.a * E * self._hill(H) - self.nu * F
        return H + self.dt * dH, F + self.dt * dF

    def simulate_past(
        self,
        past_stim: torch.Tensor,
        E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = past_stim.shape[0]
        H = torch.zeros(
            batch_size,
            device=past_stim.device,
            dtype=past_stim.dtype,
        )
        F = torch.zeros_like(H)
        zero = torch.zeros_like(H)

        for k in range(past_stim.shape[1]):
            U = past_stim[:, k - self.tau_steps] if k >= self.tau_steps else zero
            H, F = self.step(H, F, U, E)

        return H, F

    def simulate_future(
        self,
        stim: torch.Tensor,
        past_stim_tail: torch.Tensor,
        H_init: torch.Tensor,
        F_init: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        stim_context = torch.cat([past_stim_tail, stim], dim=1)

        H, F = H_init, F_init
        trajectory: list[torch.Tensor] = []
        for t in range(stim.shape[1]):
            H, F = self.step(H, F, stim_context[:, t], E)
            trajectory.append(F)

        return self.fluorescence_to_au(torch.stack(trajectory, dim=1))

    def obs_to_molecules(
        self,
        obs_scaled: torch.Tensor,
    ) -> torch.Tensor:
        obs_au = obs_scaled * self.camera_max
        molecules = (obs_au - self.camera_offset) / self.camera_mult
        return torch.clamp(molecules, min=0.0)

    def fluorescence_to_au(
        self,
        F: torch.Tensor,
    ) -> torch.Tensor:
        F_au = F * self.camera_mult + self.camera_offset
        return torch.clamp(F_au, min=0.0, max=self.camera_max)

    def _hill(
        self,
        H: torch.Tensor,
    ) -> torch.Tensor:
        H = torch.clamp(H, min=0.0)
        Hnh = H.pow(self.n_h)
        Knh = self.k_h ** self.n_h
        return Hnh / (Knh + Hnh + 1e-9)
