#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e . -q
python -m sentinel_train.pretrain.train --config configs/pretrain.yaml "$@"
