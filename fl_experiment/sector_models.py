"""Model factory for sector-agnostic forecast tasks."""

from typing import Optional

import torch.nn as nn


class ResidualMLP(nn.Module):
    """Generic residual MLP; final layer zero-init so predictions start at seasonal naive."""

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def build_forecast_model(n_features: int) -> ResidualMLP:
    return ResidualMLP(n_features=n_features)
