import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.shape[1]])


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block: multi-head attention + conv FFN."""

    def __init__(self, d_model, n_heads, d_ff, kernel_size=9, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        ff = self.conv2(self.drop(self.act(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
        return self.norm2(x + self.drop(ff))


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=2, d_ff=1024,
                 kernel_size=9, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)
        ])

    def forward(self, phoneme_ids, phoneme_lengths=None):
        key_padding_mask = None
        if phoneme_lengths is not None:
            B, T = phoneme_ids.shape
            key_padding_mask = (
                torch.arange(T, device=phoneme_ids.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(1)
            )
        x = self.pos_enc(self.embed(phoneme_ids))
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
