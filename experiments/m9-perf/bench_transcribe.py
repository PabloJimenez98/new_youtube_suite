"""M9: transcription speed/quality benchmark (docs/research/m9-performance.md).

Runs faster-whisper variants on one 16 kHz WAV and compares each with a
reference transcript (a JSON list of {start, word}, e.g. dumped from the
DB): wall time, real-time factor, WER, word-timestamp drift on matched
words, detected language and peak VRAM.

    cd backend && LD_LIBRARY_PATH=... .venv/bin/python \
        ../experiments/m9-perf/bench_transcribe.py WAV REF.json OUT.json VARIANT [VARIANT...]

Variants: seq-b5, seq-b1, batch8-b5, batch16-b5, batch8-b1, batch16-b1, ...
"""

import difflib
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import jiwer
from faster_whisper import BatchedInferencePipeline, WhisperModel

MODEL = "large-v3-turbo"


def norm(words: list[str]) -> list[str]:
    out = []
    for w in words:
        w = re.sub(r"[^\w']+", "", w.lower())
        if w:
            out.append(w)
    return out


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            if out:
                self.peak = max(self.peak, int(out.splitlines()[0]))
            time.sleep(0.5)


def run_variant(model: WhisperModel, wav: str, name: str):
    m = re.fullmatch(r"(seq|batch(\d+))-b(\d+)", name)
    beam = int(m.group(3))
    kwargs = dict(language=None, beam_size=beam, vad_filter=True, word_timestamps=True)
    t0 = time.time()
    if m.group(1) == "seq":
        segs, info = model.transcribe(wav, **kwargs)
    else:
        segs, info = BatchedInferencePipeline(model).transcribe(wav, batch_size=int(m.group(2)), **kwargs)
    words = [(w.start, w.word.strip()) for s in segs for w in (s.words or [])]
    return time.time() - t0, info, words


def compare(ref: list[dict], hyp: list[tuple]) -> dict:
    r = norm([w["word"] for w in ref])
    h = norm([w for _, w in hyp])
    wer = jiwer.wer(" ".join(r), " ".join(h))
    # Timestamp drift on words that match in sequence.
    ref_t = [w["start"] for w in ref if norm([w["word"]])]
    hyp_t = [t for t, w in hyp if norm([w])]
    deltas = []
    for a, b, n in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks():
        for k in range(n):
            if ref_t[a + k] is not None and hyp_t[b + k] is not None:
                deltas.append(abs(ref_t[a + k] - hyp_t[b + k]))
    deltas.sort()
    q = lambda p: round(deltas[int(p * (len(deltas) - 1))], 3) if deltas else None
    return {"wer": round(wer, 4), "ref_words": len(r), "hyp_words": len(h),
            "drift_p50": q(0.5), "drift_p95": q(0.95), "drift_p99": q(0.99)}


def main() -> None:
    wav, ref_path, out_path, *variants = sys.argv[1:]
    ref = json.loads(Path(ref_path).read_text())
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
                                    capture_output=True, text=True).stdout)
    t0 = time.time()
    model = WhisperModel(MODEL, device="cuda", compute_type="int8")
    load_s = time.time() - t0
    results = json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    for name in variants:
        vram = VramPeak()
        vram.start()
        try:
            secs, info, words = run_variant(model, wav, name)
        except Exception as e:  # e.g. CUDA OOM on a big batch
            results[name] = {"error": repr(e)[:300]}
            print(name, "ERROR", e, flush=True)
            continue
        finally:
            vram.stop = True
        res = {"seconds": round(secs, 1), "rtf": round(secs / duration, 4), "language": info.language,
               "vram_peak_mib": vram.peak, "model_load_s": round(load_s, 1), **compare(ref, words)}
        results[name] = res
        print(name, res, flush=True)
        Path(out_path).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
