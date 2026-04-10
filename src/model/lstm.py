from __future__ import annotations

from collections.abc import Sequence

import torch


class LSTMForecast(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        encoder_hidden_size: int,
        latent_dim: int,
        future_hidden_size: int,
        decoder_hidden_sizes: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_0 = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=encoder_hidden_size,
            batch_first=True,
        )
        self.encoder_1 = torch.nn.LSTM(
            input_size=encoder_hidden_size,
            hidden_size=latent_dim,
            batch_first=True,
        )
        self.future_projection = torch.nn.Sequential(
            torch.nn.Linear(horizon, future_hidden_size),
            torch.nn.ReLU(),
        )

        layers: list[torch.nn.Module] = []
        decoder_input_dim = latent_dim + future_hidden_size
        for hidden_size in decoder_hidden_sizes:
            layers.append(torch.nn.Linear(decoder_input_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            decoder_input_dim = hidden_size
        layers.append(torch.nn.Linear(decoder_input_dim, horizon))
        self.decoder = torch.nn.Sequential(*layers)

    def forward(
        self,
        past: torch.Tensor,
        future_stim: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _ = self.encoder_0(past)
        _, (hidden, _) = self.encoder_1(encoded)
        past_context = hidden[-1]
        future_context = self.future_projection(future_stim)
        features = torch.cat([past_context, future_context], dim=-1)
        return self.decoder(features)
