"""Sentinel v1.0 MID-TRAINING."""
from __future__ import annotations
import argparse
from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentinel_train.models import SentinelConfig, SentinelForCausalLM
from sentinel_train.data import CausalTextDataset, collate_lm
from sentinel_train.utils import load_yaml, save_checkpoint, load_checkpoint

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    model_cfg = SentinelConfig.from_dict(cfg.get("model", {}))
    train, data = cfg["train"], cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentinelForCausalLM(model_cfg).to(device)
    if train.get("bf16", True) and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    assert train.get("init_from"), "midtrain requires train.init_from"
    load_checkpoint(model, train["init_from"])
    tok = AutoTokenizer.from_pretrained(data.get("tokenizer", "gpt2"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = train.get("max_length", 4096)
    ds = CausalTextDataset(data["path"], tok, max_length=max_len)
    loader = DataLoader(ds, batch_size=train.get("batch_size", 1), shuffle=True,
                        collate_fn=partial(collate_lm, pad_id=tok.pad_token_id or 0, max_length=max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=train.get("lr", 5e-5), weight_decay=0.1, betas=(0.9, 0.95))
    step, max_steps = 0, train.get("max_steps", 5_000)
    out_dir, accum = train.get("output_dir", "outputs/midtrain"), train.get("grad_accum", 1)
    model.train(); opt.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out["loss"] / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % train.get("log_every", 10) == 0:
                print(f"[midtrain] step={step} loss={out['loss'].item():.4f}")
            if step % train.get("save_every", 1000) == 0:
                save_checkpoint(model, opt, step, f"{out_dir}/step-{step}", {**model_cfg.__dict__, "stage": "midtrain"})
            if step >= max_steps:
                break
    save_checkpoint(model, opt, step, f"{out_dir}/final", {**model_cfg.__dict__, "stage": "midtrain"})
    print(f"[midtrain] done → {out_dir}/final")

if __name__ == "__main__":
    main()
