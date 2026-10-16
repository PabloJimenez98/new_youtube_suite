"""One lock for all GPU inference in the worker process.

Whisper (M3) and pyannote (M4) both stay resident on the 3 GB GTX 1060,
but running them *concurrently* would stack their inference peaks --
so every GPU-bound handler step runs under this lock. See
docs/decisions/0008-speaker-diarization.md §1. Since M9 the LLM steps
(chapters titles, relevance, show notes) take it too: the CUDA build of
llama.cpp runs their prompt prefill on the GPU (docs/decisions/0014 §2).
"""

import asyncio

gpu_lock = asyncio.Lock()
