"""Small feed-forward next-month forecaster shared by all state clients."""

from torch import nn


class RevenueModel(nn.Module):
    """Predicts the normalized log target from normalized inputs."""

    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """Return one prediction per input row."""
        return self.net(x)
