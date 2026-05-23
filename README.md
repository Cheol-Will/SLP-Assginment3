# FastSpeech2 TTS — LJ Speech

FastSpeech2-style non-autoregressive TTS trained on LJ Speech with IPA phoneme sequences and phoneme-level duration labels.

## Setup

```bash
# install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install deps
uv sync

# log in to wandb
uv run wandb login
```

## Run

```bash
uv run python main.py \
  --data_dir /path/to/SLP-Assignment3 \
  --output_dir outputs \
  --run_name fastspeech2 \
  --max_steps 100000 \
  --batch_size 16 \
  --lr 1e-3 \
  --warmup_steps 4000 \
  --val_interval 2000 \
  --log_interval 100
```

This runs **train → test → generate** in sequence:

1. **train**: trains FastSpeech2, logs to wandb, saves `outputs/ckpt_best.pt`
2. **test**: runs inference on 500 test samples with predicted durations, computes WER via Whisper large-v3-turbo, saves `outputs/wer.txt`
3. **generate**: synthesizes 5 specific utterances, saves `.wav` to `outputs/samples/`, logs audio to wandb

## Data layout expected

```
<data_dir>/
├── manifests/ljspeech_ipa_compressed.jsonl
└── data/raw/LJSpeech-1.1/wavs/*.wav
```

## Model architecture

```
phoneme_ids
  → Embedding + Positional Encoding
  → Encoder (4× FFT blocks: multi-head attention + conv FFN)
  → Duration Predictor (log-domain, 2× Conv layers)
  → Length Regulator (repeat hidden states by predicted/GT durations)
  → Decoder (4× FFT blocks)
  → Linear → mel-spectrogram (80 bands, 22050 Hz, hop 256)
  → HiFi-GAN vocoder → waveform
```

Mel config matches HiFi-GAN pretrained: `sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0, fmax=8000`.
