import torch.nn as nn
from model.encoder import FFTBlock, PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=2, d_ff=1024,
                 kernel_size=9, max_seq_len=2000, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, mel_lengths=None):
        import torch
        key_padding_mask = None
        if mel_lengths is not None:
            B, T, _ = x.shape
            key_padding_mask = (
                torch.arange(T, device=x.device).unsqueeze(0) >= mel_lengths.unsqueeze(1)
            )
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
