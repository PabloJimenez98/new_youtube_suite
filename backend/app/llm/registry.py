"""Which GGUF an LLM tool uses: Qwen (the chapter bundle's) or our own model (M16).

See docs/decisions/0025-own-llm-m16.md §11. Every LLM-backed tool (chapter titles,
show notes, relevance) resolves its model through `resolve(choice)`:

- "qwen" -> the GGUF named by the chapter bundle's `titles.llm`, exactly as before M16.
- "own"  -> `LLM_OWN_DIR/manifest.json` ({"file", "n_ctx", "label"}) and its GGUF.

The choice for a run is the job's `params["llm"]`, else the global default stored in
`app_settings` (key `llm_default`), else "qwen" (`app.settings_store`).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

QWEN, OWN = "qwen", "own"
CHOICES = (QWEN, OWN)


class LlmUnavailableError(Exception):
    """The requested LLM isn't configured on this machine."""


@dataclass(frozen=True)
class LlmSpec:
    choice: str
    path: Path
    n_ctx: int
    label: str


def _bundle_titles() -> tuple[Path | None, dict]:
    bundle = get_settings().chapters_model_dir
    manifest = Path(bundle) / "manifest.json" if bundle else None
    if not (manifest and manifest.is_file()):
        return None, {}
    return Path(bundle), json.loads(manifest.read_text()).get("titles", {})


def own_manifest() -> tuple[Path, dict] | None:
    d = get_settings().llm_own_dir
    if d is None:
        return None
    d = Path(d)
    m = d / "manifest.json"
    if not m.is_file():
        return None
    man = json.loads(m.read_text())
    if not (d / man.get("file", "")).is_file():
        return None
    return d, man


def own_available() -> bool:
    return own_manifest() is not None


def own_label() -> str | None:
    om = own_manifest()
    return None if om is None else om[1].get("label") or Path(om[1]["file"]).stem


def resolve(choice: str | None) -> LlmSpec:
    choice = choice or QWEN
    if choice == QWEN:
        bundle, titles = _bundle_titles()
        if bundle is None or "llm" not in titles:
            raise LlmUnavailableError(
                "No Qwen LLM configured -- set CHAPTERS_MODEL_DIR to a chapter bundle with an LLM "
                "(see README, 'Chapter models')")
        path = bundle / titles["llm"]["file"]
        return LlmSpec(QWEN, path, titles["llm"].get("n_ctx", 4096), path.stem)
    if choice == OWN:
        om = own_manifest()
        if om is None:
            raise LlmUnavailableError(
                "Our own LLM isn't installed -- export one into LLM_OWN_DIR (M16, "
                "experiments/m16-own-llm/TRAINING.md)")
        d, man = om
        path = d / man["file"]
        return LlmSpec(OWN, path, int(man.get("n_ctx", 4096)), man.get("label") or path.stem)
    raise LlmUnavailableError(f"Unknown LLM choice {choice!r} (expected one of {CHOICES})")
