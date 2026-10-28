"""Compact GRU next-month forecaster shared by local and federated training."""

from torch import nn


class RevenueModel(nn.Module):
    """Reads a (batch, window, features) sequence and predicts the normalized log target."""

    def __init__(self, n_features, hidden=16, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        """Return one prediction per input window from the final hidden state."""
        _, hidden = self.gru(x)
        return self.head(hidden[-1])
