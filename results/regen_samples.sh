#!/usr/bin/env bash
# Re-run only the generate phase for all existing checkpoints using the fixed vocoder.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)"

set -o allexport
source "${DATA_DIR}/.env"
set +o allexport

# Skip smoke tests and the script itself
SKIP="_smoke_test _smoke_test2"

for exp_dir in "${DATA_DIR}/results"/*/; do
    name="$(basename "${exp_dir}")"

    # Skip smoke tests
    [[ "${name}" == _smoke_test* ]] && continue
    # Skip if no checkpoint
    [[ ! -f "${exp_dir}/ckpt_best.pt" ]] && continue
    # Skip if no vocab (can't run generate without it)
    [[ ! -f "${exp_dir}/vocab.json" ]] && continue

    echo "=== regenerating samples: ${name} ==="

    # Detect model size from vocab/checkpoint to pass correct arch flags
    n_params=$(uv run python -c "
import torch
ckpt = torch.load('${exp_dir}/ckpt_best.pt', map_location='cpu')
print(sum(p.numel() for p in ckpt['model'].values()))
" 2>/dev/null)

    if [ "${n_params}" -gt 100000000 ]; then
        arch="--d_model 512 --n_enc_layers 4 --n_dec_layers 4 --n_heads 4 --d_ff 2048"
    else
        arch="--d_model 256 --n_enc_layers 4 --n_dec_layers 4 --n_heads 2 --d_ff 1024"
    fi

    uv run python "${DATA_DIR}/main.py" \
        --data_dir  "${DATA_DIR}" \
        --output_dir "${exp_dir}" \
        --run_name  "${name}_regen" \
        --phases generate \
        ${arch} 2>&1 | tee "${exp_dir}/regen.log"

    echo "=== done: ${name} ==="
done

echo "All samples regenerated."
