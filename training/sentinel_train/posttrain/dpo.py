"""Sentinel v1.0 DPO."""
from __future__ import annotations
import argparse
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentinel_train.models import SentinelConfig, SentinelForCausalLM
from sentinel_train.data import DPODataset, collate_dpo
from sentinel_train.utils import load_yaml, save_checkpoint, load_checkpoint

def logps(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"][:, :-1]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    logp = F.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    model_cfg = SentinelConfig.from_dict(cfg.get("model", {}))
    train, data = cfg["train"], cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SentinelForCausalLM(model_cfg).to(device)
    assert train.get("init_from")
    load_checkpoint(policy, train["init_from"])
    if train.get("bf16", True) and device.type == "cuda":
        policy = policy.to(dtype=torch.bfloat16)
    ref = deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(data.get("tokenizer", "gpt2"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = train.get("max_length", 2048)
    ds = DPODataset(data["path"], tok, max_length=max_len)
    loader = DataLoader(ds, batch_size=train.get("batch_size", 1), shuffle=True,
                        collate_fn=partial(collate_dpo, pad_id=tok.pad_token_id or 0, max_length=max_len))
    beta = train.get("beta", 0.1)
    opt = torch.optim.AdamW(policy.parameters(), lr=train.get("lr", 5e-7), weight_decay=0.0)
    out_dir, max_steps, step = train.get("output_dir", "outputs/dpo"), train.get("max_steps", 2_000), 0
    policy.train()
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                ref_c = logps(ref, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                ref_r = logps(ref, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            pol_c = logps(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            pol_r = logps(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % train.get("log_every", 10) == 0:
                print(f"[dpo] step={step} loss={loss.item():.4f}")
            if step % train.get("save_every", 500) == 0:
                save_checkpoint(policy, opt, step, f"{out_dir}/step-{step}", {**model_cfg.__dict__, "stage": "dpo"})
            if step >= max_steps:
                break
    save_checkpoint(policy, opt, step, f"{out_dir}/final", {**model_cfg.__dict__, "stage": "dpo"})
    print(f"[dpo] done → {out_dir}/final")

if __name__ == "__main__":
    main()
