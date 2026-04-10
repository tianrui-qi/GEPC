from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from ..model import Model
from .config import resolve_repo_path


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(ckpt_dir: str | Path, checkpoint_name: str | None = None) -> Path:
    ckpt_dir = resolve_repo_path(ckpt_dir)
    if checkpoint_name is not None:
        path = ckpt_dir / checkpoint_name
        if path.exists():
            return path
    for candidate in ("best.ckpt", "last.ckpt"):
        path = ckpt_dir / candidate
        if path.exists():
            return path
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return checkpoints[0]


def load_model_from_checkpoint(
    cfg: DictConfig,
    *,
    ckpt_path: str | Path | None = None,
    device: str | None = None,
) -> Model:
    target_device = _resolve_device(device)
    checkpoint_path = (
        find_checkpoint(cfg.trainer.ckpt_save_fold)
        if ckpt_path is None
        else resolve_repo_path(ckpt_path)
    )

    model = Model(**cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.replace("model.", ""): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=False)
    model.to(target_device)
    model.eval()
    return model


@torch.no_grad()
def predict_arrays(
    model: torch.nn.Module,
    *,
    past: np.ndarray,
    future_stim: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    target_device = _resolve_device(device)
    model = model.to(target_device)
    predictions: list[np.ndarray] = []
    for start in range(0, len(past), batch_size):
        end = start + batch_size
        batch_past = torch.as_tensor(past[start:end], dtype=torch.float32, device=target_device)
        batch_future = torch.as_tensor(future_stim[start:end], dtype=torch.float32, device=target_device)
        batch_prediction = model(batch_past, batch_future).detach().cpu().numpy()
        predictions.append(batch_prediction)
    return np.concatenate(predictions, axis=0)


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    scale: float,
) -> dict[str, float]:
    diff = (predictions - targets) * scale
    sample_rmse = np.sqrt(np.mean(diff**2, axis=1))
    sample_mae = np.mean(np.abs(diff), axis=1)
    return {
        "rmse_mean": float(np.mean(sample_rmse)),
        "rmse_median": float(np.median(sample_rmse)),
        "rmse_std": float(np.std(sample_rmse)),
        "mae_mean": float(np.mean(sample_mae)),
    }
