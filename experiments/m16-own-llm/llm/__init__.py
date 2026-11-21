"""M16: our own LLM, trained from scratch (docs/decisions/0025-own-llm-m16.md).

- `config`: run configuration (YAML) and its resume hash.
- `model`: the decoder (llama.cpp-compatible graph, 0025 §2).
- `tokenizer`: byte-level BPE + chat template (0025 §4).
- `data`: token shards and the deterministic, resumable mixed loader (0025 §5).
- `prepare_data`: download, tokenizer sample, tokenisation into shards.
- `train`: the trainer and its CLI (start / pause / resume / status, 0025 §7).
"""

import os
from pathlib import Path

# Everything large lives on WSL ext4, never in the repo (0025 §12).
HOME = Path(os.environ.get("M16_HOME", "~/yts-experiments/m16-llm")).expanduser()
