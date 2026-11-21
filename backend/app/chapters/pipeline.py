"""Chapter generation from an exported model bundle (docs/decisions/0009 §7).

A bundle is a directory exported by `experiments/m5-chapters/05_eval`:

    manifest.json   version, boundary {kind, unit, decode, baseline_window},
                    titles {max_input_tokens, max_target_tokens, num_beams}
    encoder/        frozen sentence encoder (HF save_pretrained)
    boundary.pt     BiLSTM state_dict + model_config (only if kind == "bilstm")
    titles/         fine-tuned mT5-small (HF save_pretrained)

Bundle v2 (docs/decisions/0010) adds an LLM for the titles, with mT5 as
the per-chapter fallback:

    manifest titles.llm {file, n_ctx, max_words, max_tokens, prompt_version}
    manifest titles.lid {file}      fastText LID for the episode language
    llm/<model>.gguf

Everything runs on CPU and is synchronous: callers use
`asyncio.to_thread`. It never touches the GPU, so it doesn't take the
worker's GPU lock (0009 §7).
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from app.chapters.baseline import boundary_scores
from app.chapters.decoding import DecodeConfig, decode
from app.chapters.model import BoundaryTagger
from app.chapters.text import LanguageId
from app.chapters.units import Atom, Unit, UnitConfig, build_units
from app.config import get_settings
from app.llm import registry, runtime


class ChaptersUnavailableError(Exception):
    """No usable model bundle is configured."""


@dataclass(frozen=True)
class ChapterResult:
    start: float
    title: str


def chapter_text(units: list[Unit], start: float, end: float, max_chars: int | None = 2000) -> str:
    """The chapter's transcript -- same definition as the title model's training pairs."""
    return " ".join(u.text for u in units if start <= u.start < end)[:max_chars]


class Chapterer:
    def __init__(self, bundle_dir: Path, device: str = "cpu", llm_threads: int = 6):
        self.dir = Path(bundle_dir)
        self.device = device
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        b = self.manifest["boundary"]
        self.version: str = self.manifest["version"]
        self.kind: str = b["kind"]
        self.unit_cfg = UnitConfig(**b["unit"])
        self.decode_cfg = DecodeConfig(**b["decode"])
        self.baseline_window: int = b.get("baseline_window", 10)
        self.encoder_prefix: str = b.get("encoder_prefix", "")

        self.enc_tok = AutoTokenizer.from_pretrained(self.dir / "encoder")
        self.encoder = AutoModel.from_pretrained(self.dir / "encoder").eval().to(device)
        self.tagger: BoundaryTagger | None = None
        if self.kind == "bilstm":
            ck = torch.load(self.dir / "boundary.pt", map_location=device)
            self.tagger = BoundaryTagger(**ck["model_config"]).eval().to(device)
            self.tagger.load_state_dict(ck["state_dict"])
        elif self.kind != "baseline":
            raise ChaptersUnavailableError(f"Unknown boundary kind {self.kind!r} in {self.dir}")

        t = self.manifest["titles"]
        self.title_cfg = t
        self.title_tok = AutoTokenizer.from_pretrained(self.dir / "titles")
        self.title_model = AutoModelForSeq2SeqLM.from_pretrained(self.dir / "titles").eval().to(device)
        self.sentinels = [i for tok, i in self.title_tok.get_vocab().items() if tok.startswith("<extra_id_")]

        # Bundle v2: LLM titles, mT5 as the fallback (0010 §1).
        self.llm = self.lid = None
        self.llm_threads = llm_threads
        self._own_titler = None  # M16: our own LLM, built on first use (0025 §11)
        if llm := t.get("llm"):
            from app.chapters.llm_titles import LlamaTitler

            n_ctx = llm.get("n_ctx", 4096)
            # One Llama per process, shared with show notes (0011 §1).
            shared = runtime.get_llama(self.dir / llm["file"], n_ctx, llm_threads)
            self.llm = LlamaTitler(self.dir / llm["file"], n_ctx=n_ctx, n_threads=llm_threads,
                                   max_tokens=llm.get("max_tokens", 32), max_words=llm.get("max_words", 900),
                                   llm=shared)
            self.lid = LanguageId(self.dir / t["lid"]["file"])

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = []
        for i in range(0, len(order), batch_size):
            batch = [self.encoder_prefix + texts[j] for j in order[i : i + batch_size]]
            b = self.enc_tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            h = self.encoder(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        emb = np.empty((len(texts), self.encoder.config.hidden_size), dtype=np.float32)
        if out:
            emb[order] = np.vstack(out)
        return emb

    @torch.inference_mode()
    def boundary_scores(self, emb: np.ndarray, unit_starts: list[float], duration: float) -> np.ndarray:
        if self.tagger is None:
            return boundary_scores(emb, window=self.baseline_window)
        e = torch.from_numpy(emb)[None].to(self.device)
        pos = (torch.tensor(unit_starts, dtype=torch.float32) / duration)[None].to(self.device)
        return torch.sigmoid(self.tagger(e, pos, torch.tensor([len(unit_starts)]))).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def titles(self, texts: list[str]) -> list[str]:
        b = self.title_tok(texts, max_length=self.title_cfg["max_input_tokens"], truncation=True,
                           padding=True, return_tensors="pt").to(self.device)
        ids = self.title_model.generate(**b, max_new_tokens=self.title_cfg["max_target_tokens"],
                                        num_beams=self.title_cfg.get("num_beams", 4),
                                        suppress_tokens=self.sentinels)
        return [t.strip() for t in self.title_tok.batch_decode(ids, skip_special_tokens=True)]

    def titler(self, choice: str):
        """The LLM titler for this choice (0025 §11): the bundle's Qwen titler, or one over our own
        GGUF with the same prompts and limits. None = no LLM (mT5 titles only)."""
        if choice == registry.QWEN:
            return self.llm
        if self.lid is None:
            raise ChaptersUnavailableError("Our own LLM needs a chapter bundle with a language-ID model (v2)")
        if self._own_titler is None:
            from app.chapters.llm_titles import LlamaTitler

            spec = registry.resolve(choice)
            cfg = self.title_cfg.get("llm", {})
            shared = runtime.get_llama(spec.path, spec.n_ctx, self.llm_threads)
            self._own_titler = LlamaTitler(spec.path, n_ctx=spec.n_ctx, n_threads=self.llm_threads,
                                           max_tokens=cfg.get("max_tokens", 32), max_words=cfg.get("max_words", 900),
                                           llm=shared)
        return self._own_titler

    def llm_titles(self, units: list[Unit], starts: list[float], ends: list[float],
                   fallback_texts: list[str], titler=None) -> list[str]:
        """One LLM call per chapter, in order, with the titles given so far; mT5 where it gives none."""
        titler = titler or self.llm
        lang, _ = self.lid(" ".join(u.text for u in units)[:5000])
        out: list[str | None] = []
        for s, e in zip(starts, ends, strict=True):
            with runtime.llm_lock:
                r = titler.title(chapter_text(units, s, e, max_chars=None) or units[0].text, lang,
                                 [t for t in out if t], lid=self.lid)
            out.append(r.title)
        missing = [i for i, t in enumerate(out) if not t]
        if missing:
            for i, t in zip(missing, self.titles([fallback_texts[i] for i in missing]), strict=True):
                out[i] = t
        return out

    def run(self, atoms: list[Atom], duration: float, llm: str = registry.QWEN) -> list[ChapterResult]:
        titler = self.titler(llm)
        units = build_units(atoms, self.unit_cfg)
        if not units:
            return []
        starts = [u.start for u in units]
        if len(units) < 2:
            chapter_starts = [0.0]
        else:
            scores = self.boundary_scores(self.encode([u.text for u in units]), starts, duration)
            chapter_starts = decode(scores, starts, duration, self.decode_cfg)
        ends = [*chapter_starts[1:], duration]
        texts = [chapter_text(units, s, e) or units[0].text for s, e in zip(chapter_starts, ends, strict=True)]
        titles = self.titles(texts) if titler is None else self.llm_titles(units, chapter_starts, ends, texts, titler)
        return [ChapterResult(s, t or f"Chapter {i + 1}") for i, (s, t) in enumerate(zip(chapter_starts, titles, strict=True))]


_chapterer: Chapterer | None = None
_lock = threading.Lock()


def _drop() -> None:
    """Forget the cached object: its `Llama` was released (runtime.release_all)."""
    global _chapterer
    with _lock:
        _chapterer = None


runtime.on_release(_drop)


def get_chapterer() -> Chapterer:
    """Load the configured bundle once per process."""
    global _chapterer
    with _lock:
        if _chapterer is None:
            settings = get_settings()
            bundle = settings.chapters_model_dir
            if bundle is None or not (Path(bundle) / "manifest.json").is_file():
                raise ChaptersUnavailableError(
                    "No chapter model bundle configured -- set CHAPTERS_MODEL_DIR to an "
                    "exported bundle (see README, 'Chapter models')"
                )
            _chapterer = Chapterer(Path(bundle), settings.chapters_device, settings.chapters_llm_threads)
        return _chapterer


def generate_chapters(atoms: list[Atom], duration: float, llm: str = registry.QWEN) -> tuple[str, list[ChapterResult]]:
    """(version, chapters). Synchronous -- run in a thread.

    The version is the bundle's; with our own LLM it also names it (`<bundle>+titles:<label>`), so
    `chapters.model_version` records which model wrote the titles (M16, 0025 §11)."""
    c = get_chapterer()
    results = c.run(atoms, duration, llm)
    if llm == registry.QWEN:
        return c.version, results
    return f"{c.version}+titles:{registry.resolve(llm).label}", results
