# M4 spike: pyannote `community-1` on a GTX 1060 (GPU vs CPU)

**Date:** 2026-09-26
**Milestone:** M4 — Speaker diarization (planning sub-phase)
**Status:** Complete. Recommendation in §6.

A *spike* is a short, throwaway experiment that answers a technical
question before a design is committed to. The code here is not product
code and does not ship. It is recorded so that the M4 decision
(`docs/decisions/0008-*.md`) can point at measured numbers instead of
estimates. M3 ran the equivalent spike for faster-whisper; its numbers
live in `docs/decisions/0007-transcription-pipeline.md` §1.

## 1. Question

Can pyannote.audio 4.0 with the `pyannote/speaker-diarization-community-1`
pipeline (spec §2.2, §5) run **on this machine's GPU**? If not, is
**CPU** fast enough for a background job on 30–120 min podcast episodes?

This decides where diarization runs in production, which the project
owner fixed as this machine in M3.

## 2. Why this needs measuring, not assuming

A background research pass (2026-09-26) raised three risks. Each is
listed with how much it could be trusted.

| Risk | Claim | Confidence |
|---|---|---|
| **VRAM** | `community-1` peaked at ~9.5 GB in one reported case, about 6× the ~1.5 GB of `3.1` ([pyannote#1963](https://github.com/pyannote/pyannote-audio/issues/1963)) | One issue report. Batch sizes are tunable, so the real peak here is unknown. |
| **PyTorch vs. Pascal** | pyannote 4.0.x pins `torch==2.8.0`. torch 2.8's cu128/cu129 wheels dropped Pascal (sm_61). The **cu126** build is expected to still include it. | Plausible, not verified. The spike checks `torch.cuda.get_arch_list()` and runs a real GPU op. |
| **Shared GPU** | The worker already keeps Whisper `large-v3-turbo` resident, using ~1.55 GB of the 3 GB (0007). | Measured in M3. |

Other findings from the same pass:

- **WhisperX will not be used.** Its dependency pins conflict with
  faster-whisper 1.2.1 / pyannote 4. Its `assign_word_speakers` is
  small enough to reimplement: for each word, pick the diarization turn
  with the largest time overlap, optionally falling back to the nearest
  turn. This diverges from spec §2.2 and will be recorded in 0008.
- **Use `exclusive_speaker_diarization`.** The 4.x output offers both
  `speaker_diarization` and `exclusive_speaker_diarization`. The
  exclusive one allows no overlapping speech, so each instant has at
  most one speaker, which is what word-to-speaker assignment needs.
- **Silent failures.** The research listed several (missing token,
  license not accepted). Several of those claims looked unreliable. The
  startup healthcheck planned for M4 therefore checks the one thing
  that holds regardless of cause: non-empty, multi-speaker output on a
  known 2-speaker clip.

## 3. Setup

**Hardware**
- GPU: NVIDIA GTX 1060 3GB (Pascal, CC 6.1), driver 580.x. About
  0.5 GB is already used by the desktop.
- CPU: Ryzen 5 2600X (6 cores / 12 threads).
- RAM: 16 GB.
- OS: Linux, Python 3.12.

**Isolation**
- Everything runs in a **separate throwaway venv** in the session
  scratchpad, never in `backend/.venv`. A torch/CUDA install must not
  be able to break the working faster-whisper / ctranslate2 4.8.2 stack.
- The venv is deleted after the spike.

**Versions**
- `torch==2.8.0` and `torchaudio==2.8.0`, installed from
  `https://download.pytorch.org/whl/cu126` (the Pascal-capable build).
- `pyannote.audio==4.0.*`, installed afterwards so that its
  `torch==2.8.0` pin is satisfied by the cu126 build already present.

**Model access**
- The gated `pyannote/speaker-diarization-community-1` pipeline. Its
  weights are CC-BY-4.0.
- The owner accepted its conditions on Hugging Face.
- A read token is stored in `backend/.env` as `HF_TOKEN`. The script
  reads it and never prints it.

**Input**
- The 16 kHz mono WAV that M3 extracted from a real 2-host podcast
  episode (~64 min) uploaded locally by the owner.
- Only a **10-minute window (minute 5 to 15)** is used. It skips any
  intro music bed, and results scale roughly linearly.
- The audio is passed in memory as `{"waveform", "sample_rate"}`.
- The audio is not redistributed. Only timing and memory figures are
  recorded here.

## 4. Method

**Checks**
1. **Install sanity:** `torch.cuda.is_available()`, `get_arch_list()`
   includes a CC 6.x arch, and a matmul runs on the GPU.
2. **Timing and memory:** measured by `diar_spike.py` (below). For each
   run it:
   - loads the pipeline and moves it to the requested device;
   - optionally lowers the segmentation/embedding batch sizes, which is
     the main VRAM lever;
   - polls `nvidia-smi` every 0.3 s in a background thread for the
     peak GPU memory;
   - times one pipeline call on the 10-minute window;
   - reports speed as a multiple of real time, the speakers and turns
     found, and peak VRAM, plus `torch.cuda.max_memory_allocated` on
     GPU.

**Runs**

| Run | Device | Batch size | Purpose |
|---|---|---|---|
| A | cuda | default | Does it fit and run on the GTX 1060? |
| B | cuda | reduced (e.g. 8 → 1) | Only if A runs out of memory: is there a setting that fits? |
| C | cpu | default | Speed of the safe fallback. |

**Sanity signal:** the episode has 2 hosts, so about 2 detected
speakers is a sanity check on output quality, not only speed.

**Decision rule**
- **Runs on GPU, with Whisper co-resident or behind a shared GPU lock:**
  use the GPU.
- **Otherwise:** use CPU if it is at least ~2× real-time (≤ 30 min per
  1 h episode for a background job).
- **Both fail:** revisit, for example with pyannote `3.1` (~1.5 GB).

### Script (`diar_spike.py`, run from the scratchpad)

```python
"""M4 planning spike: pyannote community-1 speed/VRAM on GPU vs CPU.

Usage: pv/bin/python diar_spike.py {cuda|cpu} [batch]
Reads HF_TOKEN from backend/.env (never printed).
"""

import subprocess
import sys
import threading
import time

import soundfile as sf
import torch
from dotenv import dotenv_values
from pyannote.audio import Pipeline

ENV = "/home/pablo/projects/new_youtube_suite/backend/.env"
WAV = "/home/pablo/projects/new_youtube_suite/uploads/8_audio.wav"
SECONDS = 600

device = sys.argv[1]
batch = int(sys.argv[2]) if len(sys.argv) > 2 else None
token = (dotenv_values(ENV).get("HF_TOKEN") or "").strip()

audio, sr = sf.read(WAV, frames=SECONDS * 16000, start=300 * 16000, dtype="float32")
waveform = torch.from_numpy(audio).unsqueeze(0)

peak = [0]
done = False


def poll() -> None:
    while not done:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        peak[0] = max(peak[0], int(out))
        time.sleep(0.3)


t = time.time()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
pipeline.to(torch.device(device))
if batch is not None:
    for attr in ("segmentation_batch_size", "embedding_batch_size"):
        if hasattr(pipeline, attr):
            setattr(pipeline, attr, batch)
print(f"load {time.time() - t:.1f}s  batch={batch}")

th = threading.Thread(target=poll)
th.start()
t = time.time()
try:
    out = pipeline({"waveform": waveform, "sample_rate": sr})
    elapsed = time.time() - t
finally:
    done = True
    th.join()

ann = out.exclusive_speaker_diarization
speakers = sorted(ann.labels())
turns = list(ann.itertracks(yield_label=True))
print(
    f"{device}: {SECONDS}s audio -> {elapsed:.1f}s  speed={SECONDS / elapsed:.1f}x  "
    f"speakers={speakers} turns={len(turns)}  peakGPU(total)={peak[0]}MiB"
)
if device == "cuda":
    print(f"torch max_memory_allocated={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB")
```

## 5. Results

Before run A, the GPU held only the desktop (578 MiB of 3072 MiB), with
no model resident.

| Check / run | Result |
|---|---|
| Installed versions | `torch 2.8.0+cu126`, `pyannote.audio 4.0.7` (pulls `torchcodec 0.16.0`). The venv is 6.3 GB. |
| Arch list / GPU op | `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`. `sm_60` is binary-compatible with the GTX 1060's sm_61. The GPU matmul ran OK. **Pascal works with the cu126 build.** |
| **A: cuda, default batch** | Load 14.5 s. **600 s of audio in 37.1 s, so 16.2× real-time.** It found **2 speakers** (correct: 2-host episode) in 292 turns. **Peak GPU memory was 2676 MiB total**, about 2.1 GB for the process. `torch.cuda.max_memory_allocated` was 1629 MiB. |
| B: cuda, reduced batch | Not needed: A fit. |
| C: cpu | **Aborted by decision after 12 min 11 s**, still unfinished. Load takes ~15 s, so 600 s of audio had not completed in ~716 s: **slower than ~0.84× real-time**, meaning a 1 h episode would take more than ~71 min. CPU was already ruled out once A succeeded, so the run was stopped rather than waiting for an exact figure. This is a lower bound, not a precise measurement. |

### Side findings

- **torchcodec is broken in this setup.** It fails with
  `libnvrtc.so.13: cannot open shared object file`, because torchcodec
  0.16 expects CUDA 13 libraries. As a result, pyannote cannot decode
  files by path. Passing **preloaded audio**
  (`{"waveform", "sample_rate"}`) avoids the problem entirely. That was
  already the plan: the WAV M3 stored is read with `soundfile`.
- **CUDA wheel versions clash with the backend's pins.** torch 2.8.0
  cu126 requires `nvidia-cudnn-cu12==9.10.2.21` and
  `nvidia-cublas-cu12==12.6.4.1`. The backend (M3) currently pins
  9.26.0.51 and 12.9.2.10. Adding pyannote to `backend/.venv` would
  downgrade them, so ctranslate2 4.8.2 must be re-verified on those
  versions (see check D).
- **Memory: pyannote and Whisper cannot both be resident.** Whisper
  `large-v3-turbo` uses ~1.55 GB (0007) and pyannote ~2.1 GB, and the
  desktop takes ~0.55 GB. That totals more than 3 GB. Whatever runs on
  the GPU must hold **one model at a time**. *(Superseded by checks
  D/E below. This estimate assumed pyannote's default embedding batch
  of 32, and most of pyannote's VRAM is batch-dependent working memory,
  not weights.)*

### Checks D + E: Whisper and pyannote in one process

The project owner asked two follow-ups: would both models fit if the
desktop didn't use the GPU, and could they share it sequentially? Both
were measured in a single process, as in the real arq worker, on
torch's CUDA wheels. The script is `swap_spike.py` (below).

**Environment**
- The spike venv plus `faster-whisper==1.2.1` / `ctranslate2==4.8.2`.
- torch's pinned `nvidia-cudnn-cu12==9.10.2.21` and
  `nvidia-cublas-cu12==12.6.4.1` stayed in place.

**D: sequential sharing (offload to RAM between models)**
- **D1, Whisper on GPU with the torch CUDA wheels:** works. 60 s of
  audio took 7.1 s and produced 27 segments. Peak 1931 MiB; 1675 MiB
  once idle and resident.
- **D2, Whisper to RAM** (`unload_model(to_cpu=True)`): 0.33 s. The GPU
  drops back to 651 MiB.
- **D3, pyannote alone:** reload from the HF cache plus
  `to(cuda)` took 2.3 s. With `embedding_batch_size=32` (the default),
  600 s of audio took 35.1 s, i.e. **17.1× real-time**, found 2
  speakers, and **peaked at 2695 MiB**.
- **D4, swap back:** `pyannote.to(cpu)` plus `empty_cache` took 0.05 s;
  `whisper.load_model()` took 0.14 s. Re-running Whisper gave identical
  output (27 segments) with the same peak.

**E: co-residency (Whisper stays on the GPU while pyannote runs)**

| Setup | Time for 600 s | Speed | Speakers | Peak GPU (total, incl. desktop) |
|---|---|---|---|---|
| `embedding_batch_size=8` | 34.4 s | **17.4× real-time** | 2 | **2326 MiB** |
| `embedding_batch_size=1` | 47.7 s | 12.6× real-time | 2 | 1810 MiB |

**What D/E establish**
1. **Both models can stay resident, even with the desktop on the GPU,**
   if pyannote's embedding batch is lowered to 8. The peak is 2326 of
   3072 MiB, leaving ~750 MiB of headroom, and there is **no speed
   penalty** versus batch 32. That is more headroom than sequential
   swapping at batch 32 (2695 MiB).
2. **Sequential swapping works and is nearly free** (under 0.5 s per
   swap), so it stays available as a fallback. It isn't needed with
   batch 8.
3. **ctranslate2 4.8.2 works on torch's cuDNN 9.10 / cuBLAS 12.6.**
   One `backend/.venv` can host both. The backend must then pin torch's
   CUDA wheel versions instead of M3's newer ones, and install torch
   from the cu126 index.
4. **Not measured:** peak VRAM on a full-length episode. Embedding
   batches are fixed-size, so the peak should not grow with audio
   length, but this still needs confirming on a full episode during
   development.

### Script (`swap_spike.py`)

```python
"""Checks D + E: Whisper and pyannote in ONE process, on torch's CUDA wheels."""

import subprocess
import threading
import time

import soundfile as sf
import torch  # imported before ctranslate2 so torch's CUDA libs load first
from dotenv import dotenv_values
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

ENV = "/home/pablo/projects/new_youtube_suite/backend/.env"
WAV = "/home/pablo/projects/new_youtube_suite/uploads/8_audio.wav"
token = (dotenv_values(ENV).get("HF_TOKEN") or "").strip()


def used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return int(out)


class GpuPoll:  # background thread tracking peak nvidia-smi memory.used
    def __init__(self):
        self.peak, self._stop = 0, False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, used())
            time.sleep(0.2)
    def reset(self):
        self.peak = used()
    def stop(self):
        self._stop = True
        self._t.join()


def load_audio(start_s, seconds):
    return sf.read(WAV, frames=seconds * 16000, start=start_s * 16000, dtype="float32")


poll = GpuPoll()
whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
clip60, _ = load_audio(300, 60)
list(whisper.transcribe(clip60, beam_size=5, vad_filter=True, word_timestamps=True)[0])  # D1

whisper.model.unload_model(to_cpu=True)                                                 # D2

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
pipeline.to(torch.device("cuda"))
clip600, sr = load_audio(300, 600)
wave = {"waveform": torch.from_numpy(clip600).unsqueeze(0), "sample_rate": sr}
pipeline(wave)                                                                          # D3

pipeline.to(torch.device("cpu")); torch.cuda.empty_cache()                               # D4
whisper.model.load_model()
list(whisper.transcribe(clip60, beam_size=5, vad_filter=True, word_timestamps=True)[0])

pipeline.to(torch.device("cuda"))                                                       # E
for batch in (8, 1):
    pipeline.embedding_batch_size = batch
    torch.cuda.empty_cache()
    poll.reset()
    pipeline(wave)  # timed; OOM caught and reported in the full script
```

(Timing and print statements are trimmed here. The full script ran from
the session scratchpad and printed only timings, memory and counts,
never transcript text.)

## 6. Conclusion

**Recommendation: GPU for diarization, with both models resident and
pyannote's `embedding_batch_size` set to 8.**

- **Setup:** pyannote.audio 4.0.7 with `community-1`, on
  `torch==2.8.0+cu126` in the same worker process as faster-whisper.
- **Speed and memory:** ~17× real-time, so a 1 h episode diarizes in
  ~3.5 min. The measured peak is ~2.3 GB total with the desktop
  running, leaving ~750 MiB of headroom.
- **Execution:** GPU inference is serialized by one shared lock, so
  Whisper and pyannote never compute at the same time, and their
  inference peaks never stack on top of each other.
- **CPU is ruled out.** It ran below ~0.84× real-time (more than 71 min
  per hour of audio), against more than 17× on the GPU.
- **Fallbacks, both measured:**
  - swapping models between GPU and RAM, under 0.5 s per swap;
  - `embedding_batch_size=1`, at 1.8 GB peak and 12.6× real-time.

Consequences for M4's plan (to be recorded in
`docs/decisions/0008-*.md`):

- **CUDA wheel pins change.** The backend moves to torch's pins
  (`nvidia-cudnn-cu12==9.10.2.21`, `nvidia-cublas-cu12==12.6.4.1`).
  Check D1 confirmed faster-whisper still works on them. Torch is
  installed from the cu126 index.
- **Audio is passed preloaded.** torchcodec cannot load here
  (`libnvrtc.so.13`), so the stored WAV is read with `soundfile`.
- **WhisperX is not a dependency.** The word-to-speaker assignment is
  reimplemented; see §2.
- **To confirm during development:** peak VRAM on a full-length
  episode.

The spike venv (6.3 GB) is deleted after planning.
