from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a fully connected PINN."""

    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: int = 4
    hidden_units: int = 64
    fourier_features: int | None = None
    fourier_scale: float = 1.0


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier feature encoder for high-frequency inputs."""

    def __init__(self, input_dim: int, num_features: int, scale: float) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        matrix = scale * torch.randn(input_dim, num_features)
        self.register_buffer("projection", matrix)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * torch.pi * inputs @ self.projection
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class FullyConnectedPINN(nn.Module):
    """Fully connected network with tanh activations and Xavier initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if config.hidden_units < 1:
            raise ValueError("hidden_units must be at least 1.")

        features_in = config.input_dim
        self.encoder: nn.Module | None = None
        if config.fourier_features is not None:
            self.encoder = FourierFeatureEmbedding(
                input_dim=config.input_dim,
                num_features=config.fourier_features,
                scale=config.fourier_scale,
            )
            features_in = 2 * config.fourier_features

        layers: list[nn.Module] = [nn.Linear(features_in, config.hidden_units), nn.Tanh()]
        for _ in range(config.hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(config.hidden_units, config.hidden_units),
                    nn.Tanh(),
                ]
            )
        layers.append(nn.Linear(config.hidden_units, config.output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs) if self.encoder is not None else inputs
        return self.network(encoded)

