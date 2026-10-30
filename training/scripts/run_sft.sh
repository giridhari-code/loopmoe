#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e . -q
python -m sentinel_train.posttrain.sft --config configs/sft.yaml "$@"
