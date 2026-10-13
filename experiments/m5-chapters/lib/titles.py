"""Chapter-title model: mT5-small, shared embedding + LM head frozen (0009 §5).

Measured on the 1060: full fine-tuning OOMs at batch 1, while freezing
the 250k-vocab matrices (~256 M of 300 M params) fits at batch 4 x 512
tokens (docs/research/m5-chapter-model.md §5).

- `clean_title` applies light cleaning to creator titles.
- `chapter_text` gives a chapter's transcript: ASR sentences starting
  inside [start, next_start).
- `make_pairs` builds (text, title) training pairs from videos.
- `train_titles` is Adafactor + gradient checkpointing with gradient
  accumulation.
- `generate` uses beam search on CPU or GPU.
- `score` computes BLEU-1..4 and CIDEr with pycocoevalcap (VidChapters'
  metrics), with plain lowercase/whitespace tokenisation instead of
  their Java PTB tokenizer.
"""

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor

from lib.vidchapters import Video

MODEL_NAME = "google/mt5-small"
_ENUM = re.compile(r"^\s*(?:\(?\d{1,3}[\.\)\-:]|[-•–—*#])\s*")
_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def clean_title(title: str) -> str | None:
    """Light cleaning; None if nothing usable remains.

    Collapse whitespace, drop leading list enumerators ("1.", "02)",
    "- ", "•"), strip unbalanced quotes and brackets, and cap the length
    at 100 chars. A title needs at least 2 letters.
    """
    t = " ".join(str(title).split())
    t = _ENUM.sub("", t).strip()
    if t.count('"') % 2:
        t = t.replace('"', "")
    for o, c in _PAIRS:
        if t.count(o) != t.count(c):
            t = t.replace(o, "").replace(c, "")
    t = t.strip(" -–—:|,;").strip()[:100].strip()
    return t if sum(ch.isalpha() for ch in t) >= 2 else None


def chapter_text(video: Video, i: int, max_chars: int = 2000) -> str:
    start = video.chapter_starts[i]
    end = video.chapter_starts[i + 1] if i + 1 < len(video.chapter_starts) else video.duration
    a = video.asr
    parts = [t.strip() for t, s in zip(a["text"], a["start"]) if start <= s < end]
    return " ".join(parts)[:max_chars]


def make_pairs(videos: list[Video], min_text_chars: int = 40) -> list[dict]:
    out = []
    for v in videos:
        for i, raw in enumerate(v.chapter_titles):
            title = clean_title(raw)
            text = chapter_text(v, i)
            if title and len(text) >= min_text_chars:
                out.append({"vid": v.vid, "lang": v.lang, "i": i, "text": text, "title": title})
    return out


@dataclass
class TitleConfig:
    max_input_tokens: int = 256
    max_target_tokens: int = 32
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-3
    epochs: int = 1
    seed: int = 0
    eval_every: int = 2000


def load_model(device: str, path: str | Path = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    return tok, model


def freeze_embeddings(model) -> int:
    """Freeze the shared embedding and the LM head; returns trainable params."""
    model.shared.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sentinel_ids(tok) -> list[int]:
    """mT5's span-corruption sentinels (<extra_id_N>); never valid in a title."""
    return [i for t, i in tok.get_vocab().items() if t.startswith("<extra_id_")]


@torch.inference_mode()
def generate(tok, model, texts: list[str], device: str, cfg: TitleConfig = TitleConfig(),
             batch_size: int = 16, num_beams: int = 4) -> list[str]:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i : i + batch_size], max_length=cfg.max_input_tokens, truncation=True,
                padding=True, return_tensors="pt").to(device)
        ids = model.generate(**b, max_new_tokens=cfg.max_target_tokens, num_beams=num_beams,
                             suppress_tokens=sentinel_ids(tok))
        out.extend(t.strip() for t in tok.batch_decode(ids, skip_special_tokens=True))
    return out


def _tok(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower())) or "_"


def score(refs: list[str], hyps: list[str]) -> dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    g = {i: [_tok(r)] for i, r in enumerate(refs)}
    h = {i: [_tok(x)] for i, x in enumerate(hyps)}
    bleu, _ = Bleu(4).compute_score(g, h, verbose=0)
    cider, _ = Cider().compute_score(g, h)
    return {"B1": 100 * bleu[0], "B4": 100 * bleu[3], "CIDEr": 100 * cider}


def train_titles(tok, model, train_pairs: list[dict], val_pairs: list[dict], cfg: TitleConfig,
                 out_dir: Path, device: str) -> list[dict]:
    """Returns the history; saves the best-CIDEr checkpoint to out_dir."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainable = freeze_embeddings(model)
    opt = Adafactor([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                    relative_step=False, scale_parameter=False, warmup_init=False)
    history, best, step, t0 = [], -1.0, 0, time.time()

    def evaluate_now():
        nonlocal best
        hyps = generate(tok, model, [p["text"] for p in val_pairs], device, cfg)
        s = score([p["title"] for p in val_pairs], hyps)
        row = {"step": step, "sec": round(time.time() - t0), "trainable_params": trainable, **s,
               "samples": [(p["title"], h) for p, h in zip(val_pairs[:8], hyps[:8])]}
        history.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
        if s["CIDEr"] > best:
            best = s["CIDEr"]
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")
        model.train()

    for _ in range(cfg.epochs):
        pairs = train_pairs[:]
        rng.shuffle(pairs)
        model.train()
        n_batches = math.ceil(len(pairs) / cfg.batch_size)
        for bi in range(n_batches):
            batch = pairs[bi * cfg.batch_size : (bi + 1) * cfg.batch_size]
            x = tok([p["text"] for p in batch], max_length=cfg.max_input_tokens, truncation=True,
                    padding=True, return_tensors="pt").to(device)
            y = tok(text_target=[p["title"] for p in batch], max_length=cfg.max_target_tokens,
                    truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
            y[y == tok.pad_token_id] = -100
            loss = model(**x, labels=y).loss / cfg.grad_accum
            loss.backward()
            if (bi + 1) % cfg.grad_accum == 0 or bi + 1 == n_batches:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.eval_every == 0:
                    evaluate_now()
    evaluate_now()
    (out_dir / "history.json").write_text(json.dumps(history, indent=1, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return history
