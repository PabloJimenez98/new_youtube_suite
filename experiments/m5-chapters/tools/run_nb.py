"""Execute a generated notebook with experiments/m5-chapters as its cwd.

`jupyter nbconvert --execute` runs a notebook from its own directory, so a
notebook written into a run folder (M5_NB_DIR, see tools/nb.py) would lose
its relative paths (`runs/`, `lib`, `../.env`). This runs it from here and
saves the executed copy in place.

Usage (from experiments/m5-chapters/):
    ../.venv/bin/python tools/run_nb.py runs/<tag>-data/01_data.ipynb
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent.parent


def main(path: str) -> None:
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)  # keep the outputs even if a cell failed
        print(f"{path}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
