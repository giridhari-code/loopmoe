# Sentinel v1.0 — Training Pipeline

```text
pretrain  →  midtrain  →  sft  →  dpo
 (raw text)  (domain/long) (instruct) (preference)
```

Weights: `model.safetensors` → Rust Sentinel inference.

## Stages

| Stage | Module | Data |
|-------|--------|------|
| **Pre** | `pretrain.train` | `{"text"}` jsonl |
| **Mid** | `midtrain.train` | curated text, longer seq |
| **SFT** | `posttrain.sft` | instruction / chat |
| **DPO** | `posttrain.dpo` | prompt/chosen/rejected |

## Setup

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

```bash
./scripts/run_pretrain.sh
./scripts/run_midtrain.sh
./scripts/run_sft.sh
./scripts/run_dpo.sh
# or
./scripts/run_all.sh
```

Point `data.path` and `data.tokenizer` in `configs/*.yaml`.
