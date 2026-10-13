"""Tiny helper to author notebooks from Python lists of cells.

Notebooks are generated from `tools/make_*.py` so their *source* is
reviewable as plain Python in git; the executed .ipynb (with outputs)
is what documents each run.
"""
import nbformat


def build(cells: list[tuple[str, str]], path: str) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (experiments/.venv)", "language": "python"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(src.strip()) if kind == "md" else nbformat.v4.new_code_cell(src.strip())
        for kind, src in cells
    ]
    nbformat.write(nb, path)
