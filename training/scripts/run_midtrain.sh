#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e . -q
python -m sentinel_train.midtrain.train --config configs/midtrain.yaml "$@"
