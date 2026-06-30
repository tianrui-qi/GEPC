import torch


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else 1)
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        super().__init__(*args, padding=kernel_size - 1, **kwargs)
        self._trim = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x[..., :-self._trim] if self._trim > 0 else x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        max_seq_len: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        future_hidden_size: int,
        decoder_hidden_sizes: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = CausalConv1d(
            input_dim,
            model_dim,
            kernel_size=kernel_size,
        )
        self.position_embedding = torch.nn.Parameter(
            torch.zeros(1, max_seq_len, model_dim)
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = torch.nn.LayerNorm(model_dim)
        self.future_projection = torch.nn.Sequential(
            torch.nn.Linear(horizon, future_hidden_size),
            torch.nn.ReLU(),
        )

        layers: list[torch.nn.Module] = []
        decoder_input_dim = model_dim + future_hidden_size
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
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        x = past.transpose(1, 2)
        x = self.input_projection(x).transpose(1, 2)
        x = x + self.position_embedding[:, : x.shape[1]]
        x = self.encoder(x)
        past_context = self.norm(x).mean(dim=1)
        future_context = self.future_projection(future_target)
        features = torch.cat([past_context, future_context], dim=-1)
        return self.decoder(features)
