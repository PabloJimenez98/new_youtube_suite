"""Download the models of the examiner stack into docker/models (mounted at backend/models).

    docker compose -f docker/compose.yml [-f docker/compose.gpu.yml] run --rm fetch

From the project's GitHub Release (YTS_RELEASE_URL, default v1.0), checked against SHA256SUMS:
  yts-chapters.tar     chapter boundary BiLSTM + fine-tuned mT5 titles + e5 encoder + fastText LID
  yts-relevance.tar    relevance BiLSTM i2 (+ the frozen episode split)
  yts-own-llm.tar      yts-own-110m-dpo, our 100.7M LLM trained from scratch (GGUF)
  yts-extras.tar       YuNet + PP-OCR ONNX detectors, the diarization healthcheck clip, a demo clip

GPU mode also needs the Qwen3.5-9B GGUF (5.7 GB), which is not ours. It comes from QWEN_GGUF_URL,
which by default (docker/compose.yml) is the file the project used: unsloth/Qwen3.5-9B-GGUF.
Hugging Face models (Whisper, pyannote, xlm-emo-t) are cached on first use, and are downloaded now
unless `--no-prefetch` is given. pyannote's pipeline is gated: HF_TOKEN must be set and its terms
accepted.
"""

import hashlib
import os
import shutil
import sys
import tarfile
from pathlib import Path

import httpx

MODELS = Path("/app/backend/models")
RELEASE = os.environ.get("YTS_RELEASE_URL",
                         "https://github.com/PabloJimenez98/new_youtube_suite/releases/download/v1.0")
ASSETS = ["yts-chapters.tar", "yts-relevance.tar", "yts-own-llm.tar", "yts-extras.tar"]
MT5_BUNDLE = MODELS / "chapters" / "bilstm-e5s-sentence-20261107+mt5s"
QWEN_BUNDLE = MODELS / "chapters" / "bilstm-e5s-sentence-20261107+qwen3.5-9b"
QWEN_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
MODE = os.environ.get("YTS_MODE", "cpu")


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, step = 0, 0
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    log(f"  {dest.name}: {step * 10}% of {total / 1e9:.2f} GB")
    part.rename(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_release() -> None:
    cache = MODELS / ".downloads"
    sums_path = cache / "SHA256SUMS"
    download(f"{RELEASE}/SHA256SUMS", sums_path)
    sums = dict(reversed(line.split()) for line in sums_path.read_text().splitlines() if line.strip())
    for name in ASSETS:
        marker = MODELS / f".installed-{name}"
        if marker.is_file() and marker.read_text() == sums[name]:
            log(f"{name}: already installed")
            continue
        tar_path = cache / name
        if not (tar_path.is_file() and sha256(tar_path) == sums[name]):
            log(f"{name}: downloading")
            download(f"{RELEASE}/{name}", tar_path)
            if sha256(tar_path) != sums[name]:
                sys.exit(f"{name}: checksum mismatch, delete {tar_path} and retry")
        with tarfile.open(tar_path) as tar:
            tar.extractall(MODELS, filter="data")
        marker.write_text(sums[name])
        tar_path.unlink()
        log(f"{name}: installed")


def assemble_qwen_bundle() -> None:
    """GPU mode: the deployed chapter bundle = the mT5 bundle's files + the Qwen GGUF + its manifest."""
    gguf = QWEN_BUNDLE / "llm" / QWEN_FILE
    if not gguf.is_file():
        url = os.environ.get("QWEN_GGUF_URL")
        if not url:
            sys.exit("GPU mode needs the Qwen3.5-9B GGUF: set QWEN_GGUF_URL in docker/.env (see DELIVERY.md)")
        log(f"downloading {QWEN_FILE}")
        download(url, gguf)
    for src in MT5_BUNDLE.rglob("*"):
        rel = src.relative_to(MT5_BUNDLE)
        dst = QWEN_BUNDLE / rel
        if src.is_dir() or rel.name == "manifest.json" or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)  # same files, no extra disk space
        except OSError:
            shutil.copy2(src, dst)
    shutil.copy2(MODELS / "chapters" / "manifest.qwen3.5-9b.json", QWEN_BUNDLE / "manifest.json")
    log(f"assembled {QWEN_BUNDLE.name}")


def prefetch() -> None:
    """Warm the Hugging Face cache so the first job does not wait for downloads."""
    from faster_whisper.utils import download_model
    from huggingface_hub import snapshot_download

    model = os.environ.get("WHISPER_MODEL", "small")
    log(f"Whisper {model}")
    download_model(model)
    log("xlm-emo-t (hook scorer)")
    snapshot_download("MilaNLProc/xlm-emo-t")
    token = os.environ.get("HF_TOKEN") or None
    if token:
        log("pyannote speaker-diarization-community-1")
        snapshot_download("pyannote/speaker-diarization-community-1", token=token)
    else:
        log("HF_TOKEN not set: diarization will be unavailable (see DELIVERY.md)")


if __name__ == "__main__":
    fetch_release()
    if MODE == "gpu":
        assemble_qwen_bundle()
    if "--no-prefetch" not in sys.argv:
        prefetch()
    log(f"done: models for {MODE} mode are in docker/models")
