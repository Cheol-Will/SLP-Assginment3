import argparse
import json
import math
import os

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import collate_fn, load_splits
from model.fastspeech2 import FastSpeech2

SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 256
MANIFEST = "manifests/ljspeech_ipa_compressed.jsonl"



def make_padding_mask(lengths, max_len):
    """True where position >= length (padding positions)."""
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def masked_loss(pred, target, lengths, loss_fn):
    mask = ~make_padding_mask(lengths, target.shape[1])
    return loss_fn(pred[mask], target[mask])


def get_vocoder(device):
    # jik876/hifi-gan has no hubconf.py; use pseudoinverse mel + Griffin-Lim instead.
    fb = torchaudio.transforms.MelScale(
        n_mels=N_MELS, sample_rate=SAMPLE_RATE,
        f_min=0, f_max=8000, n_stft=513, norm="slaney", mel_scale="slaney",
    ).fb.to(device)  # (n_stft, n_mels)
    fb_pinv = torch.linalg.pinv(fb.T)  # (n_stft, n_mels) — pseudoinverse of mel filterbank
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=HOP_LENGTH, win_length=1024, n_iter=32, power=1.0,
    ).to(device)
    return fb_pinv, griffin_lim


def mel_to_wav(vocoder, mel, device):
    fb_pinv, griffin_lim = vocoder
    if not torch.is_tensor(mel):
        mel = torch.tensor(mel, dtype=torch.float32)
    mel_amp = torch.exp(mel.to(device)).T                       # (n_mels, T) — undo log
    spec = torch.clamp(fb_pinv @ mel_amp, min=0).unsqueeze(0)  # (1, n_stft, T)
    with torch.no_grad():
        wav = griffin_lim(spec).squeeze().cpu().numpy()
    return wav


def save_checkpoint(model, optimizer, scheduler, step, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def build_model(vocab_size, args, device):
    return FastSpeech2(
        vocab_size=vocab_size,
        n_mels=N_MELS,
        d_model=args.d_model,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        kernel_size=9,
        dropout=0.1,
    ).to(device)



def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    manifest_path = os.path.join(args.data_dir, MANIFEST)
    train_ds, val_ds, _, vocab = load_splits(args.data_dir, manifest_path)
    vocab_size = len(vocab)
    print(f"[train] vocab_size={vocab_size}, train={len(train_ds)}, val={len(val_ds)}")

    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model = build_model(vocab_size, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    def lr_lambda(step):
        step = max(step, 1)
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return max(0.05, math.sqrt(args.warmup_steps / step))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mel_loss_fn = nn.L1Loss()
    dur_loss_fn = nn.MSELoss()


    global_step = 0
    best_val_loss = float("inf")
    best_ckpt = os.path.join(args.output_dir, "ckpt_best.pt")

    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations = batch["durations"].to(device)
            mel_gt = batch["mel"].to(device)
            ph_lens = batch["phoneme_lengths"].to(device)
            mel_lens = batch["mel_lengths"].to(device)

            mel_pred, log_dur_pred, _ = model(phoneme_ids, ph_lens, durations)

            T = mel_gt.shape[1]
            if mel_pred.shape[1] < T:
                mel_pred = nn.functional.pad(mel_pred, (0, 0, 0, T - mel_pred.shape[1]))
            else:
                mel_pred = mel_pred[:, :T, :]

            loss_mel = masked_loss(mel_pred, mel_gt, mel_lens, mel_loss_fn)
            log_dur_gt = torch.log(durations.float() + 1)
            loss_dur = masked_loss(log_dur_pred, log_dur_gt, ph_lens, dur_loss_fn)
            total_loss = loss_mel + loss_dur

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]

            if global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss_total": total_loss.item(),
                    "train/loss_mel": loss_mel.item(),
                    "train/loss_dur": loss_dur.item(),
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm.item(),
                    "step": global_step,
                })
                print(
                    f"[step {global_step}/{args.max_steps}] "
                    f"loss={total_loss.item():.4f} mel={loss_mel.item():.4f} "
                    f"dur={loss_dur.item():.4f} lr={current_lr:.2e} "
                    f"grad_norm={grad_norm.item():.4f}"
                )

            if global_step % args.val_interval == 0:
                val_mel, val_dur, val_total = _validate(model, val_loader, device, mel_loss_fn, dur_loss_fn)
                wandb.log({
                    "val/loss_total": val_total,
                    "val/loss_mel": val_mel,
                    "val/loss_dur": val_dur,
                    "step": global_step,
                })
                print(f"[val {global_step}] total={val_total:.4f} mel={val_mel:.4f} dur={val_dur:.4f}")
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    save_checkpoint(model, optimizer, scheduler, global_step, best_ckpt)
                    print(f"  => best checkpoint (val={best_val_loss:.4f})")
                model.train()

    if not os.path.exists(best_ckpt):
        save_checkpoint(model, optimizer, scheduler, global_step, best_ckpt)

    print("[train] done")


def _validate(model, val_loader, device, mel_loss_fn, dur_loss_fn):
    model.eval()
    total_mel, total_dur, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            phoneme_ids = batch["phoneme_ids"].to(device)
            durations = batch["durations"].to(device)
            mel_gt = batch["mel"].to(device)
            ph_lens = batch["phoneme_lengths"].to(device)
            mel_lens = batch["mel_lengths"].to(device)

            mel_pred, log_dur_pred, _ = model(phoneme_ids, ph_lens, durations)
            T = mel_gt.shape[1]
            if mel_pred.shape[1] < T:
                mel_pred = nn.functional.pad(mel_pred, (0, 0, 0, T - mel_pred.shape[1]))
            else:
                mel_pred = mel_pred[:, :T, :]

            total_mel += masked_loss(mel_pred, mel_gt, mel_lens, mel_loss_fn).item()
            log_dur_gt = torch.log(durations.float() + 1)
            total_dur += masked_loss(log_dur_pred, log_dur_gt, ph_lens, dur_loss_fn).item()
            n += 1
    avg_mel, avg_dur = total_mel / n, total_dur / n
    return avg_mel, avg_dur, avg_mel + avg_dur


# ─── test ─────────────────────────────────────────────────────────────────────

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}")

    manifest_path = os.path.join(args.data_dir, MANIFEST)
    _, _, test_ds, _ = load_splits(args.data_dir, manifest_path)

    with open(os.path.join(args.output_dir, "vocab.json")) as f:
        vocab = json.load(f)

    model = build_model(len(vocab), args, device)
    load_checkpoint(model, os.path.join(args.output_dir, "ckpt_best.pt"), device)
    model.eval()

    vocoder = get_vocoder(device)

    import whisper
    asr = whisper.load_model("large-v3-turbo", device=device)

    from jiwer import wer as compute_wer

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_hyps, all_refs = [], []
    print(f"[test] running inference on {len(test_ds)} samples...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="test")):
            phoneme_ids = batch["phoneme_ids"].to(device)
            ph_lens = batch["phoneme_lengths"].to(device)
            mel_pred, _, _ = model(phoneme_ids, ph_lens, durations=None)

            wav = mel_to_wav(vocoder, mel_pred[0].cpu(), device)
            result = asr.transcribe(wav, fp16=(device.type == "cuda"))
            all_hyps.append(result["text"].strip())
            all_refs.append(test_ds.records[i]["transcript"])

    wer = compute_wer(all_refs, all_hyps)
    wer_path = os.path.join(args.output_dir, "wer.txt")
    with open(wer_path, "w") as f:
        f.write(f"WER: {wer:.4f}\n\n")
        for ref, hyp in zip(all_refs, all_hyps):
            f.write(f"REF: {ref}\nHYP: {hyp}\n\n")

    wandb.log({"test/wer": wer})
    print(f"[test] WER={wer:.4f}")
    print(f"[test] results saved to {wer_path}")


# ─── generate ─────────────────────────────────────────────────────────────────

GENERATE_IDS = ["LJ012-0091", "LJ014-0066", "LJ003-0011", "LJ002-0272", "LJ015-0205"]


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}")

    manifest_path = os.path.join(args.data_dir, MANIFEST)
    with open(manifest_path) as f:
        all_records = {json.loads(l)["utterance_id"]: json.loads(l) for l in f if l.strip()}

    with open(os.path.join(args.output_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)

    model = build_model(len(vocab), args, device)
    load_checkpoint(model, os.path.join(args.output_dir, "ckpt_best.pt"), device)
    model.eval()

    vocoder = get_vocoder(device)

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    samples = []
    for uid in GENERATE_IDS:
        if uid not in all_records:
            print(f"[generate] {uid} not found, skipping")
            continue
        r = all_records[uid]
        phoneme_ids = torch.tensor(
            [vocab.get(p, 0) for p in r["phoneme_sequence"]], dtype=torch.long
        ).unsqueeze(0).to(device)
        ph_lens = torch.tensor([phoneme_ids.shape[1]], dtype=torch.long, device=device)

        with torch.no_grad():
            mel_pred, _, _ = model(phoneme_ids, ph_lens, durations=None)

        wav = mel_to_wav(vocoder, mel_pred[0].cpu(), device)
        wav_path = os.path.join(samples_dir, f"{uid}.wav")
        sf.write(wav_path, wav, SAMPLE_RATE)
        samples.append((uid, wav_path))
        print(f"[generate] {uid} => {wav_path}")
        print(f"           transcript: {r['transcript']}")

    wandb.log({
        "generated_samples": [
            wandb.Audio(path, sample_rate=SAMPLE_RATE, caption=uid)
            for uid, path in samples
        ]
    })
    print("[generate] done")



def parse_args():
    parser = argparse.ArgumentParser(description="FastSpeech2 TTS — train / test / generate")
    parser.add_argument("--data_dir", default=".", help="root directory containing manifests/ and data/")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name", default="fastspeech2")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_enc_layers", type=int, default=4)
    parser.add_argument("--n_dec_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--phases", nargs="+", default=["train", "test", "generate"],
                        choices=["train", "test", "generate"],
                        help="which phases to run (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="fastspeech2-tts", name=args.run_name, config=vars(args))
    if "train" in args.phases:
        train(args)
    if "test" in args.phases:
        test(args)
    if "generate" in args.phases:
        generate(args)
    wandb.finish()
