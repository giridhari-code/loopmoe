#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip install -e . -q
echo "=== PRETRAIN ==="
python -m sentinel_train.pretrain.train --config configs/pretrain.yaml
echo "=== MIDTRAIN ==="
python -m sentinel_train.midtrain.train --config configs/midtrain.yaml
echo "=== SFT ==="
python -m sentinel_train.posttrain.sft --config configs/sft.yaml
echo "=== DPO ==="
python -m sentinel_train.posttrain.dpo --config configs/dpo.yaml
echo "Done: outputs/dpo/final/model.safetensors"
