"""Make the CUDA build of llama-cpp-python importable (M9, docs/decisions/0014 §2).

The CUDA build links against `libcudart.so.12` / `libcublas.so.12`. The
venv already ships them in NVIDIA's pip packages (for faster-whisper and
torch), but not on the loader path. Loading them once with RTLD_GLOBAL
before `import llama_cpp` lets its libraries resolve them, without
asking every caller (worker, API, tests) to set LD_LIBRARY_PATH. With a
CPU-only build this is a harmless no-op.
"""

import ctypes
import importlib.util
from pathlib import Path

_LIBS = ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12")
_done = False


def preload() -> None:
    global _done
    if _done:
        return
    _done = True
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for rel in _LIBS:
            path = Path(root) / rel
            if path.is_file():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
