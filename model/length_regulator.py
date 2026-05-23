import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    """Expand encoder hidden states by repeating each frame according to duration."""

    def forward(self, x, durations):
        # x: (B, T_ph, d_model), durations: (B, T_ph) int
        outputs = []
        for i in range(x.shape[0]):
            repeated = torch.repeat_interleave(x[i], durations[i], dim=0)
            outputs.append(repeated)
        max_len = max(o.shape[0] for o in outputs)
        padded = torch.zeros(x.shape[0], max_len, x.shape[2], device=x.device)
        mel_lengths = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for i, o in enumerate(outputs):
            padded[i, : o.shape[0]] = o
            mel_lengths[i] = o.shape[0]
        return padded, mel_lengths
