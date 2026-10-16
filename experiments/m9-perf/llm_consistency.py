"""M9: are GPU-prefill relevance scores consistent with CPU ones? (docs/research/m9-performance.md)

Scores the same relevance batches (real episode words) with one config and
dumps the integer scores, so two runs (CPU build vs CUDA build, or two
thread counts) can be compared batch by batch.

    llm_consistency.py WORDS.tsv OUT.json N_GPU_LAYERS THREADS FIRST_BATCH N_BATCHES
"""

import json
import sys
from pathlib import Path

from app.llm import cuda_libs

cuda_libs.preload()
from llama_cpp import Llama  # noqa: E402

from app.llm.chat import LlamaChat  # noqa: E402
from app.relevance.llm_scoring import BATCH, llm_blocks, score_batch  # noqa: E402
from app.relevance.windows import Word, build_blocks  # noqa: E402

words_tsv, out, gpu, threads, first, n = sys.argv[1:]
words = [Word(float(a) if a else None, w) for a, _b, w, _s in (l.rstrip("\n").split("\t") for l in open(words_tsv))]
texts = [t for t, _ in llm_blocks(build_blocks(words, max(w.start or 0 for w in words) + 1)) if t]
llm = Llama(model_path="models/chapters/current/llm/Qwen3.5-4B-Q4_K_M.gguf", n_ctx=4096, n_threads=int(threads),
            n_gpu_layers=int(gpu), verbose=False)
chat = LlamaChat(llm, 4096)
res = []
for b in range(int(first), int(first) + int(n)):
    batch = texts[b * BATCH:(b + 1) * BATCH]
    scores = score_batch(chat, batch, "es", [None] * len(batch))
    res.append([round(s * 100) for s in scores] if scores else None)
    print(b, res[-1], flush=True)
Path(out).write_text(json.dumps(res))
