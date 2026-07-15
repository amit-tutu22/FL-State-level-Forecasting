"""Simple autoencoder for welfare leakage reconstruction scores."""

from typing import Optional

import torch
import torch.nn as nn

from leakage_utils import FEATURE_COLS


class LeakageAutoencoder(nn.Module):
    def __init__(self, n_features: Optional[int] = None, latent: int = 4):
        super().__init__()
        n_features = n_features or len(FEATURE_COLS)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16),
            nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=1)
