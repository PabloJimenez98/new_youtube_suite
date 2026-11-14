"""Relevance v1: frozen multilingual-e5 window embeddings -> BiLSTM -> two heads.

See docs/decisions/0024-relevance-bilstm-m15.md §1, §3, §5 and §6. The two heads
predict the teacher's two components (heuristic, LLM); the score combines them
with the teacher's weights, so it is on the same scale as relevance v0.

A bundle is a directory:

    config.json    kind, label, encoder {dir, name, prefix, max_length},
                   model_config, combine {heuristic, llm}, window spec,
                   training details (lambda, seed, teacher, split hash)
    weights.pt     the BiLSTM state_dict
    metrics.json   training (val) and eval (test) metrics
    encoder/       the frozen sentence encoder (HF save_pretrained)

Synchronous and blocking: callers use `asyncio.to_thread`.
"""

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from app.config import get_settings
from app.relevance.pipeline import HEURISTIC_WEIGHT, LLM_WEIGHT, RelevanceResult, WindowScore
from app.relevance.windows import BLOCK_S, BLOCKS_PER_WINDOW, Window, Word, build_blocks, build_windows

BUNDLE_KIND = "bilstm"
ENCODER_MAX_LENGTH = 128  # a 15 s window is ~40-50 words, well under 128 tokens

# (texts) -> [N, D] float32, L2-normalized
Encoder = Callable[[list[str]], np.ndarray]


class RelevanceBundleError(Exception):
    """The live version's bundle is missing or unreadable."""


class RelevanceBiLSTM(nn.Module):
    """[B, T, D] window embeddings -> ([B, T] heuristic head, [B, T] LLM head), both in (0, 1)."""

    def __init__(self, emb_dim: int = 384, proj: int = 128, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.3, head: int = 64):
        super().__init__()
        self.config = dict(emb_dim=emb_dim, proj=proj, hidden=hidden, layers=layers, dropout=dropout, head=head)
        self.inp = nn.Sequential(nn.Linear(emb_dim, proj), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)

        def _head() -> nn.Module:
            return nn.Sequential(nn.Linear(2 * hidden, head), nn.ReLU(), nn.Linear(head, 1))

        self.heuristic_head = _head()
        self.llm_head = _head()

    def forward(self, emb: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inp(emb)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=emb.shape[1])
        return (torch.sigmoid(self.heuristic_head(h).squeeze(-1)),
                torch.sigmoid(self.llm_head(h).squeeze(-1)))


def combine_heads(h: torch.Tensor, l: torch.Tensor,  # noqa: E741
                  wh: float = HEURISTIC_WEIGHT, wl: float = LLM_WEIGHT) -> torch.Tensor:
    return wh * h + wl * l


def distillation_loss(h_hat: torch.Tensor, l_hat: torch.Tensor, h: torch.Tensor, l: torch.Tensor,  # noqa: E741
                      mask: torch.Tensor, lam: float, *, delta: float = 0.1,
                      wh: float = HEURISTIC_WEIGHT, wl: float = LLM_WEIGHT) -> torch.Tensor:
    """Huber on each head + lam * mean squared step of the combined score (0024 §3).

    `mask` [B, T] is 1 on real windows. `l` is NaN where the teacher's LLM gave
    nothing: those windows drop out of the LLM term only.
    """
    mask = mask.float()
    l_mask = mask * (~torch.isnan(l)).float()
    l_clean = torch.nan_to_num(l, nan=0.0)
    hub_h = nn.functional.huber_loss(h_hat, h, reduction="none", delta=delta)
    hub_l = nn.functional.huber_loss(l_hat, l_clean, reduction="none", delta=delta)
    loss_h = (hub_h * mask).sum() / mask.sum().clamp(min=1)
    loss_l = (hub_l * l_mask).sum() / l_mask.sum().clamp(min=1)
    s = combine_heads(h_hat, l_hat, wh, wl)
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = ((s[:, 1:] - s[:, :-1]) ** 2 * pair).sum() / pair.sum().clamp(min=1)
    return loss_h + loss_l + lam * smooth


@torch.inference_mode()
def predict(model: RelevanceBiLSTM, emb: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """One episode's [T, D] embeddings -> (heuristic head [T], LLM head [T])."""
    model.eval()
    if len(emb) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))[None].to(device)
    h, l = model(x, torch.tensor([len(emb)]))  # noqa: E741
    return h[0].cpu().numpy(), l[0].cpu().numpy()


# ---- encoder -----------------------------------------------------------------


class SentenceEncoder:
    """Mean-pooled, L2-normalized sentence embeddings: the chapters bundle's convention (0009 §3)."""

    def __init__(self, encoder_dir: Path, prefix: str = "passage: ", max_length: int = ENCODER_MAX_LENGTH,
                 device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.dir, self.prefix, self.max_length, self.device = Path(encoder_dir), prefix, max_length, device
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModel.from_pretrained(self.dir).eval().to(device)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = []
        for i in range(0, len(order), batch_size):
            batch = [self.prefix + texts[j] for j in order[i : i + batch_size]]
            b = self.tok(batch, padding=True, truncation=True, max_length=self.max_length,
                         return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        emb = np.empty((len(texts), self.dim), dtype=np.float32)
        if out:
            emb[order] = np.vstack(out)
        return emb


def embedding_cache_key(encoder_name: str, prefix: str, max_length: int, texts: list[str]) -> str:
    h = hashlib.sha1(f"{encoder_name}\x00{prefix}\x00{max_length}\x00".encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cached_embeddings(cache_dir: Path | None, key: str, texts: list[str], encode: Encoder) -> np.ndarray:
    """The frozen encoder's output is a pure function of the texts: compute once, reuse (0024 §1)."""
    path = cache_dir / f"{key}.npy" if cache_dir is not None else None
    if path is not None and path.is_file():
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            return emb
    emb = encode(texts)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(path)
    return emb


def default_cache_dir() -> Path:
    return get_settings().storage_base_path / "cache" / "relevance-embeddings"


def resolve_device(requested: str) -> str:
    return "cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested


# ---- bundle --------------------------------------------------------------------


# How the score is formed (0024 §9, i2 lever 1):
# - "heads": both components from the BiLSTM.
# - "exact_heuristic": the heuristic computed exactly (cheap, no LLM) + the BiLSTM's LLM head.
COMBINE_MODES = ("heads", "exact_heuristic")
LID_FILE = "lid.176.ftz"


def combine_scores(mode: str, h_head: np.ndarray, l_head: np.ndarray, exact_heuristic: np.ndarray | None,
                   wh: float = HEURISTIC_WEIGHT, wl: float = LLM_WEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """(heuristic component used, combined score)."""
    if mode == "exact_heuristic":
        if exact_heuristic is None:
            raise ValueError("combine mode 'exact_heuristic' needs the exact heuristic scores")
        h = np.asarray(exact_heuristic, dtype=np.float32)
    elif mode == "heads":
        h = np.asarray(h_head, dtype=np.float32)
    else:
        raise ValueError(f"Unknown combine mode {mode!r}")
    return h, (wh * h + wl * np.asarray(l_head, dtype=np.float32)).astype(np.float32)


@dataclass(frozen=True)
class BundleConfig:
    label: str
    encoder_name: str
    encoder_prefix: str
    encoder_max_length: int
    model_config: dict
    combine_heuristic: float = HEURISTIC_WEIGHT
    combine_llm: float = LLM_WEIGHT
    combine_mode: str = "heads"
    training: dict | None = None

    def to_json(self) -> dict:
        return {
            "kind": BUNDLE_KIND,
            "label": self.label,
            "encoder": {"dir": "encoder", "name": self.encoder_name, "prefix": self.encoder_prefix,
                        "max_length": self.encoder_max_length},
            "model_config": self.model_config,
            "combine": {"heuristic": self.combine_heuristic, "llm": self.combine_llm, "mode": self.combine_mode},
            "window": {"block_s": BLOCK_S, "blocks_per_window": BLOCKS_PER_WINDOW},
            "training": self.training or {},
        }

    @classmethod
    def from_json(cls, data: dict) -> "BundleConfig":
        if data.get("kind") != BUNDLE_KIND:
            raise RelevanceBundleError(f"Not a relevance BiLSTM bundle (kind={data.get('kind')!r})")
        w = data.get("window", {})
        if (w.get("block_s"), w.get("blocks_per_window")) != (BLOCK_S, BLOCKS_PER_WINDOW):
            raise RelevanceBundleError(f"Bundle window spec {w} does not match this code's windows")
        mode = data["combine"].get("mode", "heads")
        if mode not in COMBINE_MODES:
            raise RelevanceBundleError(f"Unknown combine mode {mode!r}")
        e = data["encoder"]
        return cls(label=data["label"], encoder_name=e["name"], encoder_prefix=e["prefix"],
                   encoder_max_length=e["max_length"], model_config=data["model_config"],
                   combine_heuristic=data["combine"]["heuristic"], combine_llm=data["combine"]["llm"],
                   combine_mode=mode, training=data.get("training"))


def save_bundle(bundle_dir: Path, config: BundleConfig, model: RelevanceBiLSTM, metrics: dict,
                encoder_src: Path | None, lid_src: Path | None = None) -> None:
    """Write config, weights and metrics; copy the encoder (and the LID model) unless already there."""
    import shutil

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), bundle_dir / "weights.pt")
    (bundle_dir / "config.json").write_text(json.dumps(config.to_json(), indent=1))
    (bundle_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if encoder_src is not None and not (bundle_dir / "encoder").exists():
        shutil.copytree(encoder_src, bundle_dir / "encoder")
    if lid_src is not None and not (bundle_dir / LID_FILE).exists():
        shutil.copyfile(lid_src, bundle_dir / LID_FILE)


def load_model(bundle_dir: Path, device: str = "cpu") -> tuple[BundleConfig, RelevanceBiLSTM]:
    bundle_dir = Path(bundle_dir)
    try:
        config = BundleConfig.from_json(json.loads((bundle_dir / "config.json").read_text()))
        model = RelevanceBiLSTM(**config.model_config)
        model.load_state_dict(torch.load(bundle_dir / "weights.pt", map_location=device, weights_only=True))
    except RelevanceBundleError:
        raise
    except Exception as exc:  # missing files, bad JSON, corrupt or mismatched weights
        raise RelevanceBundleError(f"Cannot load the relevance bundle at {bundle_dir}: {exc}") from exc
    return config, model.eval().to(device)


def check_bundle(bundle_dir: Path) -> None:
    """Fail fast on a bundle that can't serve (used before a live switch)."""
    config, _ = load_model(bundle_dir)
    if not (Path(bundle_dir) / "encoder" / "config.json").is_file():
        raise RelevanceBundleError(f"The relevance bundle at {bundle_dir} has no encoder/")
    if config.combine_mode == "exact_heuristic" and not (Path(bundle_dir) / LID_FILE).is_file():
        raise RelevanceBundleError(f"The relevance bundle at {bundle_dir} has no {LID_FILE} (needed by exact_heuristic)")


# ---- inference ---------------------------------------------------------------


def episode_windows(words: list[Word], duration: float) -> list[Window]:
    return build_windows(build_blocks(words, duration))


class BiLSTMScorer:
    """A live bundle, loaded once per process (0024 §6)."""

    def __init__(self, bundle_dir: Path, device: str = "cpu", encoder: Encoder | None = None,
                 cache_dir: Path | None = None, lid: Callable[[str], tuple[str, float]] | None = None):
        self.dir = Path(bundle_dir)
        self.device = device
        self.config, self.model = load_model(self.dir, device)
        self.encode = encoder or SentenceEncoder(self.dir / "encoder", self.config.encoder_prefix,
                                                 self.config.encoder_max_length, device)
        self.cache_dir = cache_dir
        self.lid = lid
        if self.config.combine_mode == "exact_heuristic" and self.lid is None:
            from app.chapters.text import LanguageId

            self.lid = LanguageId(self.dir / LID_FILE)

    def embed(self, windows: list[Window]) -> np.ndarray:
        texts = [w.text for w in windows]
        key = embedding_cache_key(self.config.encoder_name, self.config.encoder_prefix,
                                  self.config.encoder_max_length, texts)
        return cached_embeddings(self.cache_dir, key, texts, self.encode)

    def exact_heuristic(self, windows: list[Window]) -> tuple[str, list[float]]:
        """The teacher's heuristic, with the teacher's language rule (pipeline.RelevanceScorer.run)."""
        from app.relevance import heuristics

        text = " ".join(w.text for w in windows)
        detected, _ = self.lid(text[:5000])
        lang = "es" if detected == "es" else "en"
        return lang, heuristics.heuristic_scores(windows, lang)

    def score_windows(self, windows: list[Window]) -> tuple[str, list[WindowScore]]:
        h_head, l_head = predict(self.model, self.embed(windows), self.device)
        lang, exact = "", None
        if self.config.combine_mode == "exact_heuristic":
            lang, exact = self.exact_heuristic(windows)
        h, s = combine_scores(self.config.combine_mode, h_head, l_head, exact,
                              self.config.combine_heuristic, self.config.combine_llm)
        return lang, [WindowScore(w, float(a), float(b), float(c))
                      for w, a, b, c in zip(windows, h, l_head, s, strict=True)]

    def run(self, words: list[Word], duration: float) -> RelevanceResult:
        lang, scores = self.score_windows(episode_windows(words, duration))
        return RelevanceResult(lang, scores)

_scorers: dict[str, BiLSTMScorer] = {}
_lock = threading.Lock()


def get_scorer(bundle_dir: str | Path) -> BiLSTMScorer:
    key = str(Path(bundle_dir))
    with _lock:
        if key not in _scorers:
            if not (Path(bundle_dir) / "config.json").is_file():
                raise RelevanceBundleError(
                    f"The live relevance version's bundle is missing at {bundle_dir} -- "
                    "switch back with `python -m app.cli models set-live relevance <label>`"
                )
            _scorers.clear()  # one live bundle at a time
            _scorers[key] = BiLSTMScorer(Path(bundle_dir), resolve_device(get_settings().relevance_device),
                                         cache_dir=default_cache_dir())
        return _scorers[key]


def score_with_bundle(bundle_dir: str | Path, words: list[Word], duration: float) -> RelevanceResult:
    """Synchronous -- run in a thread."""
    return get_scorer(bundle_dir).run(words, duration)
