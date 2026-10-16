"""M9: score relevance blocks one per call (batch of 1) and dump the scores.

    llm_single.py WORDS.tsv OUT.json N_GPU_LAYERS FIRST N
(run with CUDA_VISIBLE_DEVICES= for the CPU)
"""

import json
import sys
import time
from pathlib import Path

from app.llm import cuda_libs

cuda_libs.preload()
from llama_cpp import Llama  # noqa: E402

from app.llm.chat import LlamaChat  # noqa: E402
from app.relevance.llm_scoring import llm_blocks, score_batch  # noqa: E402
from app.relevance.windows import Word, build_blocks  # noqa: E402

words_tsv, out, gpu, first, n = sys.argv[1:]
words = [Word(float(a) if a else None, w) for a, _b, w, _s in (l.rstrip("\n").split("\t") for l in open(words_tsv))]
texts = [t for t, _ in llm_blocks(build_blocks(words, max(w.start or 0 for w in words) + 1)) if t]
chat = LlamaChat(Llama(model_path="models/chapters/current/llm/Qwen3.5-4B-Q4_K_M.gguf", n_ctx=4096, n_threads=6,
                       n_gpu_layers=int(gpu), verbose=False), 4096)
t0 = time.time()
scores = []
for t in texts[int(first):int(first) + int(n)]:
    s = score_batch(chat, [t], "es", [None])
    scores.append(round(s[0] * 100) if s else None)
el = time.time() - t0
print(json.dumps(scores), f"{el:.1f}s, {el / int(n):.2f}s/call", flush=True)
Path(out).write_text(json.dumps({"scores": scores, "seconds": el}))
