import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predicts log-domain duration per phoneme."""

    def __init__(self, d_model=256, d_hidden=256, kernel_size=3, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = d_model
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, d_hidden, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(d_hidden),  # will be applied after transpose
                nn.Dropout(dropout),
            ]
            in_ch = d_hidden
        self.net = nn.ModuleList(layers)
        self.proj = nn.Linear(d_hidden, 1)

    def forward(self, x):
        # x: (B, T, d_model)
        h = x.transpose(1, 2)  # (B, d_model, T)
        i = 0
        while i < len(self.net):
            conv = self.net[i]
            relu = self.net[i + 1]
            ln = self.net[i + 2]
            drop = self.net[i + 3]
            h = drop(ln(relu(conv(h)).transpose(1, 2)).transpose(1, 2))
            i += 4
        return self.proj(h.transpose(1, 2)).squeeze(-1)  # (B, T)
