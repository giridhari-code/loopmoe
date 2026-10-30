from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

def save_checkpoint(model, optimizer, step: int, out_dir: str, cfg: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(cpu_state, str(out / "model.safetensors"))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, out / "trainer.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

def load_checkpoint(model, path: str, optimizer=None):
    p = Path(path)
    weights = p / "model.safetensors" if p.is_dir() else p
    state = load_file(str(weights))
    model.load_state_dict(state, strict=False)
    step = 0
    trainer = p / "trainer.pt" if p.is_dir() else None
    if trainer and trainer.exists() and optimizer is not None:
        blob = torch.load(trainer, map_location="cpu")
        optimizer.load_state_dict(blob["optimizer"])
        step = int(blob.get("step", 0))
    return step
