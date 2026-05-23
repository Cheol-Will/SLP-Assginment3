import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.duration_predictor import DurationPredictor
from model.length_regulator import LengthRegulator


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=256, n_enc_layers=4, n_dec_layers=4,
                 n_heads=2, d_ff=1024, kernel_size=9, max_seq_len=1000, max_mel_len=2000,
                 dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, n_enc_layers, n_heads, d_ff, kernel_size, max_seq_len, dropout
        )
        self.duration_predictor = DurationPredictor(d_model, d_model, 3, 2, dropout)
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(
            d_model, n_dec_layers, n_heads, d_ff, kernel_size, max_mel_len, dropout
        )
        self.proj = nn.Linear(d_model, n_mels)

    def forward(self, phoneme_ids, phoneme_lengths=None, durations=None):
        enc_out = self.encoder(phoneme_ids, phoneme_lengths)  # (B, T_ph, d)
        log_dur_pred = self.duration_predictor(enc_out)  # (B, T_ph)

        if durations is not None:
            # teacher-forced: use GT durations
            regulated, mel_lengths = self.length_regulator(enc_out, durations)
        else:
            # inference: use predicted durations
            dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, min=1).round().long()
            if phoneme_lengths is not None:
                for i in range(dur_pred.shape[0]):
                    dur_pred[i, phoneme_lengths[i]:] = 0
            regulated, mel_lengths = self.length_regulator(enc_out, dur_pred)

        dec_out = self.decoder(regulated, mel_lengths)  # (B, T_mel, d)
        mel_out = self.proj(dec_out)  # (B, T_mel, n_mels)
        return mel_out, log_dur_pred, mel_lengths
