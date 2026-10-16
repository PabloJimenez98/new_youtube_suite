"""M9: relevance as the expected value over digit probabilities (one block per pass).

Instead of reading the integer the model writes (greedy, which flips
on tiny numeric changes), read the next-token distribution over the
digits 0-9 and take its expectation: a continuous score in [0, 1].

    llm_expected.py WORDS.tsv OUT.json N_GPU_LAYERS FIRST N
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from app.llm import cuda_libs

cuda_libs.preload()
import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from app.chapters.llm_titles import render_prompt  # noqa: E402
from app.llm import runtime  # noqa: E402
from app.relevance.llm_scoring import llm_blocks  # noqa: E402
from app.relevance.windows import Word, build_blocks  # noqa: E402

SYSTEM = "Valoras momentos de un episodio de podcast para clips de vídeo cortos."
USER = ("Fragmento de 15 segundos de un episodio de podcast:\n\n«{text}»\n\n"
        "¿Funcionaría como YouTube Short independiente: interesante, autocontenido y atractivo "
        "(gracioso, revelador, sorprendente, polémico o emotivo)? El relleno, la logística, la publicidad "
        "y lo que no se entiende solo puntúan bajo.\nResponde solo con un número del 0 (nada) al 9 (excelente).")

words_tsv, out, gpu, first, n = sys.argv[1:]
words = [Word(float(a) if a else None, w) for a, _b, w, _s in (l.rstrip("\n").split("\t") for l in open(words_tsv))]
texts = [t for t, _ in llm_blocks(build_blocks(words, max(w.start or 0 for w in words) + 1)) if t]
llm = Llama(model_path="models/chapters/current/llm/Qwen3.5-4B-Q4_K_M.gguf", n_ctx=4096, n_threads=6,
            n_gpu_layers=int(gpu), verbose=False)
template, bos, eos = runtime.chat_tokens(llm)
digits = [llm.tokenize(str(d).encode(), add_bos=False, special=False) for d in range(10)]
assert all(len(d) == 1 for d in digits), digits
digit_ids = [d[0] for d in digits]
t0 = time.time()
scores, argmax = [], []
for t in texts[int(first):int(first) + int(n)]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(text=t)}]
    ids = llm.tokenize(render_prompt(template, msgs, bos, eos).encode(), add_bos=False, special=True)
    llm.reset()
    llm.eval(ids)
    last = llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
    logits = np.ctypeslib.as_array(last, shape=(llm.n_vocab(),))[digit_ids].astype(np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    scores.append(round(float((p * np.arange(10)).sum() / 9), 3))
    argmax.append(int(p.argmax()))
el = time.time() - t0
print(json.dumps(scores), json.dumps(argmax), f"{el:.1f}s, {el / int(n):.2f}s/call", flush=True)
Path(out).write_text(json.dumps({"scores": scores, "argmax": argmax, "seconds": el}))
