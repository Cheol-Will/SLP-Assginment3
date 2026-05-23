#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
set -o allexport; source "${DATA_DIR}/.env"; set +o allexport

bs=64
steps=10000
warmup_steps=4000
val_interval=2000
log_interval=100
d_model=512
n_heads=4
d_ff=2048

# lr=1e-3
lr=3e-4

n_enc_layers=4; n_dec_layers=4

run_name="lr${lr}_enc${n_enc_layers}_dec${n_dec_layers}_s${steps}"
mkdir -p "${DATA_DIR}/results/${run_name}"

uv run python "${DATA_DIR}/main.py" \
    --data_dir      "${DATA_DIR}" \
    --output_dir    "${DATA_DIR}/results/${run_name}" \
    --run_name      "${run_name}" \
    --max_steps     "${steps}" \
    --batch_size    "${bs}" \
    --lr            "${lr}" \
    --warmup_steps  "${warmup_steps}" \
    --val_interval  "${val_interval}" \
    --log_interval  "${log_interval}" \
    --d_model       "${d_model}" \
    --n_enc_layers  "${n_enc_layers}" \
    --n_dec_layers  "${n_dec_layers}" \
    --n_heads       "${n_heads}" \
    --d_ff          "${d_ff}" \
    2>&1 | tee "${DATA_DIR}/results/${run_name}/run.log"


lr=1e-4
# n_enc_layers=6; n_dec_layers=6
run_name="lr${lr}_enc${n_enc_layers}_dec${n_dec_layers}_s${steps}"
mkdir -p "${DATA_DIR}/results/${run_name}"

uv run python "${DATA_DIR}/main.py" \
    --data_dir      "${DATA_DIR}" \
    --output_dir    "${DATA_DIR}/results/${run_name}" \
    --run_name      "${run_name}" \
    --max_steps     "${steps}" \
    --batch_size    "${bs}" \
    --lr            "${lr}" \
    --warmup_steps  "${warmup_steps}" \
    --val_interval  "${val_interval}" \
    --log_interval  "${log_interval}" \
    --d_model       "${d_model}" \
    --n_enc_layers  "${n_enc_layers}" \
    --n_dec_layers  "${n_dec_layers}" \
    --n_heads       "${n_heads}" \
    --d_ff          "${d_ff}" \
    2>&1 | tee "${DATA_DIR}/results/${run_name}/run.log"