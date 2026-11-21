"""The M16 decoder (0025 §2), written from scratch, with a graph llama.cpp already runs.

Pre-norm decoder-only transformer: RMSNorm, rotary embeddings in the
*interleaved* (GPT-J / original Llama) layout, which is llama.cpp's `llama`
rope type, so the GGUF export needs no weight permutation. SwiGLU MLP, grouped-
query attention, no biases, tied input/output embeddings. Optional per-head
q/k RMSNorm (`qk_norm`, an ablation in 0025 §3; it exports as `qwen3`).

Tensor names follow llama.cpp's (attn_q, attn_k, ..., ffn_gate/up/down) to keep
the export a rename-free mapping.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import ModelConfig


def rope_tables(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    f = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, hd); cos/sin: (T, hd/2). Rotates adjacent pairs (x0, x1), (x2, x3), ..."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        assert c.d_model % c.n_heads == 0 and c.n_heads % c.n_kv_heads == 0
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.attn_q = nn.Linear(c.d_model, self.h * self.hd, bias=False)
        self.attn_k = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.attn_v = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.attn_output = nn.Linear(self.h * self.hd, c.d_model, bias=False)
        self.qk_norm = c.qk_norm
        if c.qk_norm:
            self.attn_q_norm = nn.RMSNorm(self.hd, eps=c.norm_eps)
            self.attn_k_norm = nn.RMSNorm(self.hd, eps=c.norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.attn_q(x).view(B, T, self.h, self.hd)
        k = self.attn_k(x).view(B, T, self.kv, self.hd)
        v = self.attn_v(x).view(B, T, self.kv, self.hd)
        if self.qk_norm:
            q, k = self.attn_q_norm(q), self.attn_k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, enable_gqa=self.kv != self.h)
        return self.attn_output(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class FeedForward(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.kind = c.mlp
        if c.mlp == "swiglu":
            self.ffn_gate = nn.Linear(c.d_model, c.hidden, bias=False)
        elif c.mlp != "relu2":
            raise ValueError(f"unknown mlp {c.mlp!r}")
        self.ffn_up = nn.Linear(c.d_model, c.hidden, bias=False)
        self.ffn_down = nn.Linear(c.hidden, c.d_model, bias=False)

    def forward(self, x):
        if self.kind == "relu2":
            return self.ffn_down(F.relu(self.ffn_up(x)).square())
        return self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.ffn = FeedForward(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Decoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.token_embd = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output_norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        cos, sin = rope_tables(c.max_seq_len, c.head_dim, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)  # residual projections (GPT-2 / Llama practice)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                nn.init.ones_(p)  # RMSNorm weights
            elif name.endswith(("attn_output.weight", "ffn_down.weight")):
                nn.init.normal_(p, 0.0, out_std)
            else:
                nn.init.normal_(p, 0.0, std)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.token_embd(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.output_norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):
        """Logits (B, T, V) without targets; else the mean cross-entropy over unmasked positions."""
        logits = F.linear(self.hidden(idx), self.token_embd.weight)  # tied head
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is None:
            return loss.mean()
        m = loss_mask.reshape(-1).float()
        return (loss * m).sum() / m.sum().clamp(min=1)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.token_embd.weight.numel() if non_embedding else n

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, eos_id: int | None = None) -> torch.Tensor:
        """Greedy, no KV cache: only for the trainer's short sample generations."""
        for _ in range(max_new):
            nxt = self(idx)[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
