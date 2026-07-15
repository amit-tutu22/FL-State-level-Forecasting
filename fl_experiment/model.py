from typing import Optional

import torch.nn as nn

from utils import FEATURE_COLS


class RevenueModel(nn.Module):
    """Residual MLP: final layer starts near zero so predictions begin at seasonal naive."""

    def __init__(self, n_features: Optional[int] = None):
        super().__init__()
        n_features = n_features or len(FEATURE_COLS)
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)
