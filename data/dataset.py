import json
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

# Mel-spectrogram config matching HiFi-GAN
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 8000

_mel_transform = None

def get_mel_transform():
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )
    return _mel_transform


def build_vocab(records):
    phonemes = set()
    for r in records:
        phonemes.update(r["phoneme_sequence"])
    vocab = {"<pad>": 0}
    for p in sorted(phonemes):
        vocab[p] = len(vocab)
    return vocab


def duration_to_frames(alignment):
    return [max(1, round(a["duration"] * SAMPLE_RATE / HOP_LENGTH)) for a in alignment]


class LJSpeechDataset(Dataset):
    def __init__(self, records, vocab, data_dir):
        self.records = records
        self.vocab = vocab
        self.data_dir = data_dir
        self.mel_fn = get_mel_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        phoneme_ids = torch.tensor(
            [self.vocab.get(p, 0) for p in r["phoneme_sequence"]], dtype=torch.long
        )
        dur_frames = torch.tensor(duration_to_frames(r["phoneme_alignment"]), dtype=torch.long)

        wav_path = os.path.join(self.data_dir, r["wav_path"])
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(data.copy())
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        mel = self.mel_fn(waveform).squeeze(0)  # (n_mels, T)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.T  # (T, n_mels)

        total_frames = dur_frames.sum().item()
        if mel.shape[0] > total_frames:
            mel = mel[:total_frames]
        elif mel.shape[0] < total_frames:
            pad = total_frames - mel.shape[0]
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))

        return {
            "utterance_id": r["utterance_id"],
            "phoneme_ids": phoneme_ids,
            "durations": dur_frames,
            "mel": mel,
        }


def collate_fn(batch):
    phoneme_ids = [b["phoneme_ids"] for b in batch]
    durations = [b["durations"] for b in batch]
    mels = [b["mel"] for b in batch]
    ids = [b["utterance_id"] for b in batch]

    max_ph = max(p.shape[0] for p in phoneme_ids)
    max_mel = max(m.shape[0] for m in mels)
    n_mels = mels[0].shape[1]

    ph_padded = torch.zeros(len(batch), max_ph, dtype=torch.long)
    dur_padded = torch.zeros(len(batch), max_ph, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), max_mel, n_mels)
    ph_lens = torch.zeros(len(batch), dtype=torch.long)
    mel_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, (p, d, m) in enumerate(zip(phoneme_ids, durations, mels)):
        ph_padded[i, : p.shape[0]] = p
        dur_padded[i, : d.shape[0]] = d
        mel_padded[i, : m.shape[0]] = m
        ph_lens[i] = p.shape[0]
        mel_lens[i] = m.shape[0]

    return {
        "utterance_ids": ids,
        "phoneme_ids": ph_padded,
        "durations": dur_padded,
        "mel": mel_padded,
        "phoneme_lengths": ph_lens,
        "mel_lengths": mel_lens,
    }


def load_splits(data_dir, manifest_path, train_n=12500, val_n=100, test_n=500):
    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # deterministic split: first train_n / next val_n / next test_n
    train_records = records[:train_n]
    val_records = records[train_n: train_n + val_n]
    test_records = records[train_n + val_n: train_n + val_n + test_n]

    vocab = build_vocab(records)

    train_ds = LJSpeechDataset(train_records, vocab, data_dir)
    val_ds = LJSpeechDataset(val_records, vocab, data_dir)
    test_ds = LJSpeechDataset(test_records, vocab, data_dir)

    return train_ds, val_ds, test_ds, vocab
