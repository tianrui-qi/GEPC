from __future__ import annotations

import torch

from .lstm import LSTMForecast
from .transformer import TransformerForecast


class Model(torch.nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        self.name = name

        if name == "lstm":
            self.network = LSTMForecast(**kwargs)
        elif name == "transformer":
            self.network = TransformerForecast(**kwargs)
        else:
            raise ValueError(f"Unsupported model name: {name}")

    def forward(
        self,
        past: torch.Tensor,
        future_stim: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(past, future_stim)
