"""Data loaders for pre / mid / post stages."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class CausalTextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tok = tokenizer
        self.max_length = max_length
        self.rows = []
        p = Path(path)
        if p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    obj = json.loads(line)
                    self.rows.append(obj.get("text") or obj.get("content") or "")
        else:
            self.rows = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = self.tok.encode(self.rows[i], add_special_tokens=True)[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tok = tokenizer
        self.max_length = max_length
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.rows)

    def _format(self, row):
        if "messages" in row:
            parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in row["messages"][:-1]]
            prompt = "\n".join(parts) + "\nassistant:"
            answer = row["messages"][-1].get("content", "")
            return prompt, answer
        inst = row.get("instruction") or row.get("prompt") or ""
        inp = row.get("input") or ""
        out = row.get("output") or row.get("response") or ""
        prompt = f"### Instruction:\n{inst}\n"
        if inp:
            prompt += f"### Input:\n{inp}\n"
        prompt += "### Response:\n"
        return prompt, out

    def __getitem__(self, i):
        prompt, answer = self._format(self.rows[i])
        full = prompt + answer
        ids = self.tok.encode(full, add_special_tokens=True)[: self.max_length]
        prompt_ids = self.tok.encode(prompt, add_special_tokens=True)
        labels = ids.copy()
        plen = min(len(prompt_ids), len(labels))
        labels[:plen] = [-100] * plen
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tok = tokenizer
        self.max_length = max_length
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = r.get("prompt") or r.get("instruction") or ""
        chosen = r["chosen"] if isinstance(r["chosen"], str) else r["chosen"][-1]["content"]
        rejected = r["rejected"] if isinstance(r["rejected"], str) else r["rejected"][-1]["content"]
        def enc(text):
            return torch.tensor(self.tok.encode(text, add_special_tokens=True)[: self.max_length], dtype=torch.long)
        return {"prompt_ids": enc(prompt), "chosen_ids": enc(prompt + chosen), "rejected_ids": enc(prompt + rejected)}

def _pad(seqs, pad_id, max_len=None):
    m = max(s.size(0) for s in seqs)
    if max_len:
        m = min(m, max_len)
    out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), m, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = min(s.size(0), m)
        out[i, :n] = s[:n]
        attn[i, :n] = 1
    return out, attn

def collate_lm(batch, pad_id=0, max_length=2048):
    ids, attn = _pad([b["input_ids"] for b in batch], pad_id, max_length)
    labels = ids.clone()
    labels[attn == 0] = -100
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

def collate_sft(batch, pad_id=0, max_length=2048):
    ids, attn = _pad([b["input_ids"] for b in batch], pad_id, max_length)
    labels_list = []
    for b in batch:
        lab = b["labels"]
        if lab.size(0) < ids.size(1):
            lab = torch.cat([lab, torch.full((ids.size(1) - lab.size(0),), -100)])
        labels_list.append(lab[: ids.size(1)])
    labels = torch.stack(labels_list)
    labels[attn == 0] = -100
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

def collate_dpo(batch, pad_id=0, max_length=2048):
    c_ids, c_attn = _pad([b["chosen_ids"] for b in batch], pad_id, max_length)
    r_ids, r_attn = _pad([b["rejected_ids"] for b in batch], pad_id, max_length)
    return {"chosen_input_ids": c_ids, "chosen_attention_mask": c_attn, "rejected_input_ids": r_ids, "rejected_attention_mask": r_attn}
