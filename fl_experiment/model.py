from typing import Optional

import torch.nn as nn

from utils import FEATURE_COLS


class RevenueModel(nn.Module):
    def __init__(self, n_features: Optional[int] = None):
        super().__init__()
        n_features = n_features or len(FEATURE_COLS)
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
