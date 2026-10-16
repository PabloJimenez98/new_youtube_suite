"""One llama.cpp model per worker process, shared between tools.

`chapters` (titles, 0010) and `shownotes` (0011) use the same GGUF.
Loading it once per process keeps the worker ~3.5 GB lighter than one
instance per tool. A `Llama` is not thread-safe and arq runs several
jobs at once (`max_jobs` defaults to 10), so every inference call takes
`llm_lock`. See docs/decisions/0011-show-notes-local-llm.md §1.
"""

import threading
from pathlib import Path

from app.llm import cuda_libs

_cache: dict[tuple[str, int], object] = {}
_load_lock = threading.Lock()
# Held around every generate/complete call on a shared instance.
llm_lock = threading.Lock()


def _load(path: Path, n_ctx: int, n_threads: int):
    cuda_libs.preload()
    from llama_cpp import Llama

    # n_gpu_layers=0: the weights stay in RAM. With the CUDA build,
    # llama.cpp still runs large prompt batches (the prefill) on the GPU,
    # ~13x faster than the CPU here, using little VRAM; generation stays
    # on the CPU. See docs/decisions/0014 §2.
    return Llama(model_path=str(path), n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=0, verbose=False)


def get_llama(path: Path, n_ctx: int = 4096, n_threads: int = 6):
    """The process-wide `Llama` for this file and context size, loaded on first use."""
    key = (str(Path(path).resolve()), n_ctx)
    with _load_lock:
        if key not in _cache:
            _cache[key] = _load(Path(path), n_ctx, n_threads)
        return _cache[key]


# Callbacks that drop objects holding a cached `Llama` (scorers, generators).
_release_hooks: list = []


def on_release(fn) -> None:
    """Register `fn` to run when the shared models are released."""
    _release_hooks.append(fn)


def release_all() -> None:
    """Close every loaded `Llama` and let holders rebuild on next use (M9, 0014 §2).

    With the CUDA build a llama.cpp context keeps ~1 GB of VRAM while it
    exists, which left too little room for batched Whisper on the 3 GB
    GPU (season-1 run: CUDA OOM). The LLM handlers call this at the end
    of each job; reloading costs a few seconds (the GGUF is in the page
    cache).
    """
    with _load_lock, llm_lock:
        for llm in _cache.values():
            llm.close()
        _cache.clear()
    for fn in _release_hooks:
        fn()


def chat_tokens(llm) -> tuple[str, str, str]:
    """(chat template, BOS, EOS) from the GGUF metadata -- same rules as `LlamaTitler`."""
    md = llm.metadata
    template = md.get("tokenizer.chat_template")
    if not template:
        raise ValueError("GGUF has no chat template in its metadata")
    tok = llm._model
    # Qwen's GGUF declares no BOS; only use one the file actually declares.
    wants_bos = ("tokenizer.ggml.bos_token_id" in md
                 and md.get("tokenizer.ggml.add_bos_token", "true") != "false")
    bos = tok.token_get_text(tok.token_bos()) if wants_bos and tok.token_bos() >= 0 else ""
    eos = tok.token_get_text(tok.token_eos()) if tok.token_eos() >= 0 else ""
    return template, bos, eos
