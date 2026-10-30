"""Sentinel v1.0 — Looped MoE causal LM (training)."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SentinelConfig:
    model_name: str = "Sentinel-v1.0"
    vocab_size: int = 128_256
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_layers: int = 8
    loop_count: int = 24
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 1408
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"
    norm_topk_prob: bool = True
    max_position_embeddings: int = 131_072
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    moe_aux_loss_coef: float = 0.001
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SentinelConfig":
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in keys})

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(n + self.eps) * self.weight).type_as(self.weight)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq, theta):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
    def forward(self, seq_len, device, dtype):
        return self.cos[:seq_len].to(device=device, dtype=dtype), self.sin[:seq_len].to(device=device, dtype=dtype)

class Attention(nn.Module):
    def __init__(self, cfg: SentinelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)
    def forward(self, x, cos, sin, attn_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if attn_mask is not None:
            att = att + attn_mask
        att = att.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        out = (att @ v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)

class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, cfg: SentinelConfig):
        super().__init__()
        self.cfg = cfg
        h, inter = cfg.hidden_size, cfg.moe_intermediate_size
        self.gate = nn.Linear(h, cfg.n_routed_experts, bias=False)
        self.routed = nn.ModuleList([Expert(h, inter) for _ in range(cfg.n_routed_experts)])
        self.shared = nn.ModuleList([Expert(h, inter) for _ in range(cfg.n_shared_experts)])
        self.aux_loss = torch.tensor(0.0)
    def forward(self, x):
        b, s, h = x.shape
        flat = x.reshape(-1, h)
        logits = self.gate(flat)
        scores = logits.sigmoid() if self.cfg.scoring_func == "sigmoid" else logits
        topk_w, topk_ids = scores.topk(self.cfg.num_experts_per_tok, dim=-1)
        if self.cfg.scoring_func != "sigmoid":
            topk_w = topk_w.softmax(dim=-1)
        elif self.cfg.norm_topk_prob:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)
        with torch.no_grad():
            hits = torch.zeros(self.cfg.n_routed_experts, device=x.device)
            hits.scatter_add_(0, topk_ids.reshape(-1), torch.ones(topk_ids.numel(), device=x.device))
        probs = scores.softmax(-1).mean(0)
        self.aux_loss = (probs * hits / hits.sum().clamp_min(1)).sum() * self.cfg.n_routed_experts
        out = sum((e(flat) for e in self.shared), torch.zeros_like(flat))
        routed_out = torch.zeros_like(flat)
        for k in range(self.cfg.num_experts_per_tok):
            ids_k, w_k = topk_ids[:, k], topk_w[:, k].unsqueeze(1)
            for e in range(self.cfg.n_routed_experts):
                mask = ids_k == e
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                routed_out.index_add_(0, idx, self.routed[e](flat[idx]) * w_k[idx])
        return (out + routed_out * self.cfg.routed_scaling_factor).view(b, s, h)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MoE(cfg)
    def forward(self, x, cos, sin, attn_mask=None):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        return h + self.mlp(self.post_attention_layernorm(h))

class SentinelForCausalLM(nn.Module):
    def __init__(self, cfg: SentinelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rotary = RotaryEmbedding(cfg.head_dim, min(8192, cfg.max_position_embeddings), cfg.rope_theta)

    def forward(self, input_ids, labels=None, attention_mask=None):
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(s, x.device, x.dtype)
        causal = torch.triu(torch.full((s, s), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        attn_mask = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad = (1 - attention_mask.to(x.dtype)) * torch.finfo(x.dtype).min
            attn_mask = attn_mask + pad.unsqueeze(1).unsqueeze(2)
        aux = x.new_zeros(())
        for _ in range(self.cfg.loop_count):
            for layer in self.layers:
                x = layer(x, cos, sin, attn_mask)
                aux = aux + layer.mlp.aux_loss
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
            loss = loss + self.cfg.moe_aux_loss_coef * (aux / max(1, self.cfg.loop_count * self.cfg.num_layers))
        return {"logits": logits, "loss": loss}
