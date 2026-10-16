"""One llama.cpp model per worker process, shared between tools.

`chapters` (titles, 0010) and `shownotes` (0011) use the same GGUF.
Loading it once per process keeps the worker ~3.5 GB lighter than one
instance per tool. A `Llama` is not thread-safe and arq runs several
jobs at once (`max_jobs` defaults to 10), so every inference call takes
`llm_lock`. See docs/decisions/0011-show-notes-local-llm.md §1.
"""

import threading
from pathlib import Path

_cache: dict[tuple[str, int], object] = {}
_load_lock = threading.Lock()
# Held around every generate/complete call on a shared instance.
llm_lock = threading.Lock()


def _load(path: Path, n_ctx: int, n_threads: int):
    from llama_cpp import Llama

    return Llama(model_path=str(path), n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=0, verbose=False)


def get_llama(path: Path, n_ctx: int = 4096, n_threads: int = 6):
    """The process-wide `Llama` for this file and context size, loaded on first use."""
    key = (str(Path(path).resolve()), n_ctx)
    with _load_lock:
        if key not in _cache:
            _cache[key] = _load(Path(path), n_ctx, n_threads)
        return _cache[key]


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
