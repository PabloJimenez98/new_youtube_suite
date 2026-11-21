"""Export our decoder + tokenizer to a GGUF that the app's llama.cpp runtime loads (0025 §10).

    python -m llm.export_gguf <weights.pt> <tokenizer.json> <out.gguf> [--f32]

Graph mapping (0025 §2): architecture `llama` (or `qwen3` when `qk_norm`).
- Rotary layout: our model rotates adjacent pairs, which is llama.cpp's `llama`
  rope type (NORM), so q/k weights are written as they are. `qwen3` uses the
  NEOX (half-split) layout, so for it q/k rows are permuted from interleaved
  to half-split order per head.
- Tied embeddings: no `output.weight`; llama.cpp then uses `token_embd`.
- Tokenizer: `gpt2`-type BPE with the `llama-bpe` pre-tokenizer (the Llama-3
  split regex our tokenizer uses), the ChatML template, EOS = `<|im_end|>`, no BOS.
"""

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from llm.config import ModelConfig
from llm.tokenizer import CHAT_TEMPLATE, EOT, IM_END, PAD, SPECIALS


def preload_llama_cpp():
    """The CUDA build of llama-cpp-python needs libcudart/libcublas on the loader path (as app/llm/cuda_libs)."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for rel in ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12"):
            p = Path(root) / rel
            if p.is_file():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)


def _interleaved_to_halves(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Rows of q/k (n_heads*hd, d) from per-head (x0,x1,x2,x3...) pairs to (x0,x2,...,x1,x3,...)."""
    out, d = w.shape
    hd = out // n_heads
    return w.view(n_heads, hd // 2, 2, d).transpose(1, 2).reshape(out, d)


def export(state: dict, cfg: ModelConfig, tokenizer_json: Path, out: Path, f32: bool = False, name: str = "yts-own"):
    import gguf

    arch = "qwen3" if cfg.qk_norm else "llama"
    if cfg.mlp != "swiglu":
        raise ValueError(f"mlp={cfg.mlp!r} has no llama.cpp graph (0025 §2)")
    w = gguf.GGUFWriter(str(out), arch=arch)
    w.add_name(name)
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.d_model)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.hidden)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_vocab_size(cfg.vocab_size)
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if f32 else gguf.LlamaFileType.MOSTLY_F16)

    tj = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    for t in tj.get("added_tokens", []):
        vocab[t["content"]] = t["id"]
    if len(vocab) != cfg.vocab_size:
        raise ValueError(f"tokenizer has {len(vocab)} tokens, model {cfg.vocab_size}")
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    types = [gguf.TokenType.CONTROL if t in SPECIALS else gguf.TokenType.NORMAL for t in tokens]
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("llama-bpe")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(vocab[EOT])
    w.add_add_bos_token(False)
    w.add_eos_token_id(vocab[IM_END])
    w.add_eot_token_id(vocab[IM_END])
    w.add_pad_token_id(vocab[PAD])
    w.add_chat_template(CHAT_TEMPLATE)

    def put(name: str, t: torch.Tensor):
        a = t.detach().float().cpu().numpy()
        if a.ndim >= 2 and not f32:
            a = a.astype(np.float16)
        w.add_tensor(name, a)

    s = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    put("token_embd.weight", s["token_embd.weight"])
    put("output_norm.weight", s["output_norm.weight"])
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        q, k = s[p + "attn.attn_q.weight"], s[p + "attn.attn_k.weight"]
        if arch == "qwen3":
            q, k = _interleaved_to_halves(q, cfg.n_heads), _interleaved_to_halves(k, cfg.n_kv_heads)
            # The per-head q/k norm weights act on head_dim elements in the permuted order too.
            hd = cfg.head_dim
            perm = torch.arange(hd).view(hd // 2, 2).t().reshape(-1)
            put(f"blk.{i}.attn_q_norm.weight", s[p + "attn.attn_q_norm.weight"][perm])
            put(f"blk.{i}.attn_k_norm.weight", s[p + "attn.attn_k_norm.weight"][perm])
        put(f"blk.{i}.attn_norm.weight", s[p + "attn_norm.weight"])
        put(f"blk.{i}.attn_q.weight", q)
        put(f"blk.{i}.attn_k.weight", k)
        put(f"blk.{i}.attn_v.weight", s[p + "attn.attn_v.weight"])
        put(f"blk.{i}.attn_output.weight", s[p + "attn.attn_output.weight"])
        put(f"blk.{i}.ffn_norm.weight", s[p + "ffn_norm.weight"])
        put(f"blk.{i}.ffn_gate.weight", s[p + "ffn.ffn_gate.weight"])
        put(f"blk.{i}.ffn_up.weight", s[p + "ffn.ffn_up.weight"])
        put(f"blk.{i}.ffn_down.weight", s[p + "ffn.ffn_down.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def main(argv=None):
    from llm import config as cfgmod

    ap = argparse.ArgumentParser(prog="python -m llm.export_gguf")
    ap.add_argument("weights", help="final_model.pt or a checkpoint with 'model' and 'config'")
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--name", default="yts-own")
    a = ap.parse_args(argv)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    cfg = cfgmod.from_dict(ck["config"]).model
    export(ck["model"], cfg, Path(a.tokenizer), Path(a.out), a.f32, a.name)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
