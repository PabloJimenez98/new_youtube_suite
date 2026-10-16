"""Transcribe one WAV with one model/variant and dump the words (M9 benchmark).

    asr_variant.py MODEL VARIANT WAV OUT.json     # VARIANT: seq-bN | batchK-bN

One variant per process: a CUDA OOM in faster-whisper leaves the device
unusable for the rest of the process.
"""

import json
import re
import sys
import time

from faster_whisper import BatchedInferencePipeline, WhisperModel

model_name, variant, wav, out = sys.argv[1:]
m = re.fullmatch(r"(seq|batch(\d+))-b(\d+)", variant)
device = "cpu" if model_name.endswith("@cpu") else "cuda"
model = WhisperModel(model_name.removesuffix("@cpu"), device=device, compute_type="int8", cpu_threads=12)
kwargs = dict(language="es", beam_size=int(m.group(3)), vad_filter=True, word_timestamps=True)
t0 = time.time()
if m.group(1) == "seq":
    segs, _ = model.transcribe(wav, **kwargs)
else:
    segs, _ = BatchedInferencePipeline(model).transcribe(wav, batch_size=int(m.group(2)), **kwargs)
words = [{"start": w.start, "word": w.word.strip()} for s in segs for w in (s.words or [])]
json.dump({"seconds": time.time() - t0, "words": words}, open(out, "w"), ensure_ascii=False)
print(variant, model_name, round(time.time() - t0, 1), len(words), flush=True)
