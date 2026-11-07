"""Tiny helper to author notebooks from Python lists of cells.

Notebooks are generated from `tools/make_*.py` so their *source* is
reviewable as plain Python in git; the executed .ipynb (with outputs)
is what documents each run.
"""
import os

import nbformat


def build(cells: list[tuple[str, str]], path: str) -> None:
    # A retrain (M5_RUN_TAG, see lib/runs.py) writes its notebooks into
    # M5_NB_DIR so the committed, executed ones keep documenting their run.
    out_dir = os.environ.get("M5_NB_DIR", "").strip()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.basename(path))
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (experiments/.venv)", "language": "python"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(src.strip()) if kind == "md" else nbformat.v4.new_code_cell(src.strip())
        for kind, src in cells
    ]
    nbformat.write(nb, path)
