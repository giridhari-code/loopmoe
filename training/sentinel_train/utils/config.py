from __future__ import annotations
import yaml
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
