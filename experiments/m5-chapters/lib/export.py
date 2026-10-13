"""Export a chapter-model bundle for the backend (0009 §6-7).

Layout (read by `backend/app/chapters/pipeline.py`):

    <out>/manifest.json   version, boundary {...}, titles {...}, metrics, run
    <out>/encoder/        frozen sentence encoder (HF save_pretrained)
    <out>/boundary.pt     BiLSTM {state_dict, model_config}   (kind == "bilstm" only)
    <out>/titles/         fine-tuned mT5-small (HF save_pretrained)

`current` next to the versioned directories is a symlink to the bundle
the backend should serve (`CHAPTERS_MODEL_DIR=models/chapters/current`).

`add_llm_titles` derives a v2 bundle (0010 §3) from an exported one: the
same boundary model and mT5 (now the fallback), plus a title LLM and the
LID model for the episode language:

    <out>/llm/<model>.gguf
    <out>/lid.176.ftz
    manifest titles.llm {file, n_ctx, max_words, max_tokens, prompt_version}, titles.lid {file}
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from lib.decoding import DecodeConfig
from lib.encode import ENCODERS
from lib.units import UnitConfig


def export_bundle(
    out_root: Path,
    version: str,
    kind: str,
    encoder_key: str,
    unit_cfg: UnitConfig,
    decode_cfg: DecodeConfig,
    baseline_window: int,
    boundary_ckpt: Path | None,
    titles_dir: Path,
    title_cfg: dict,
    metrics: dict,
    run: str,
    make_current: bool = True,
) -> Path:
    if kind not in ("bilstm", "baseline"):
        raise ValueError(kind)
    out = Path(out_root) / version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    name, prefix = ENCODERS[encoder_key]
    AutoTokenizer.from_pretrained(name).save_pretrained(out / "encoder")
    AutoModel.from_pretrained(name).save_pretrained(out / "encoder")

    if kind == "bilstm":
        ck = torch.load(boundary_ckpt, map_location="cpu")
        torch.save({"state_dict": ck["state_dict"], "model_config": ck["model_config"]}, out / "boundary.pt")

    shutil.copytree(titles_dir, out / "titles")

    manifest = {
        "version": version,
        "boundary": {
            "kind": kind,
            "encoder": name,
            "encoder_prefix": prefix,
            "unit": asdict(unit_cfg),
            "decode": asdict(decode_cfg),
            "baseline_window": baseline_window,
        },
        "titles": title_cfg,
        "metrics": metrics,
        "run": run,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    if make_current:
        link = Path(out_root) / "current"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(version)
    return out


def add_llm_titles(
    src: Path,
    out_root: Path,
    version: str,
    gguf: Path,
    lid: Path,
    llm_cfg: dict,
    metrics: dict,
    run: str,
    make_current: bool = True,
) -> Path:
    src, out = Path(src).resolve(), Path(out_root) / version
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("llm", "lid.176.ftz"))
    (out / "llm").mkdir()
    shutil.copy2(gguf, out / "llm" / Path(gguf).name)
    shutil.copy2(lid, out / "lid.176.ftz")

    manifest = json.loads((out / "manifest.json").read_text())
    manifest["version"] = version
    manifest["titles"]["llm"] = {"file": f"llm/{Path(gguf).name}", **llm_cfg}
    manifest["titles"]["lid"] = {"file": "lid.176.ftz"}
    manifest["titles_metrics"] = metrics
    manifest["derived_from"] = manifest.get("run")
    manifest["run"] = run
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    if make_current:
        link = Path(out_root) / "current"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(version)
    return out
