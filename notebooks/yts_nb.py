"""Shared setup for the delivery notebooks (docs/decisions/0026).

Every notebook starts with `from yts_nb import *`. That import:

- puts `backend/` on `sys.path` and makes it the working directory, so `app.*` imports
  and `backend/.env` load exactly as they do for the API and the worker;
- exposes `session_factory`, the app's own async SQLAlchemy session factory;
- gives `sql()` (a read-only query to a DataFrame), the data paths and a plot style.

The notebooks only read: they never write to the database or to a model bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

import warnings

# Quiet, readable outputs: no download/progress bars, no warnings about optional audio decoders
# that these notebooks never use.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec")
warnings.filterwarnings("ignore", message=r"(?s).*IProgress not found")

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
os.chdir(BACKEND)
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.database import async_session_factory as session_factory  # noqa: E402

settings = get_settings()

# Inputs that live outside the repository (audio excerpts, raw benchmark outputs). They are not
# distributed (0026 §3); the notebooks keep their executed outputs so they can be read without them.
EXPERIMENTS_DATA = Path(os.environ.get("YTS_EXPERIMENTS_DATA", Path.home() / "yts-experiments"))
EXPERIMENTS = REPO / "experiments"

# One palette for every chart: categorical slots, and a neutral grey for references/baselines.
COLORS = ["#2a6fdb", "#e07b39", "#2e9e6b", "#8a5cd1", "#c94f7c", "#5b8a99"]
GREY = "#8c8c8c"

plt.rcParams.update({
    "figure.figsize": (8, 4),
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=COLORS),
    "font.size": 10,
})
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", 30)


async def sql(query: str, **params) -> pd.DataFrame:
    """Run one read-only SELECT through the app's session factory and return a DataFrame."""
    if not query.lstrip().lower().startswith(("select", "with")):
        raise ValueError("sql() is read-only: SELECT / WITH only")
    async with session_factory() as db:
        result = await db.execute(text(query), params)
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def compare(label: str, recomputed: float, reported: float, tol: float = 0.0005) -> str:
    """One line that states whether a recomputed number matches the report (0026 §4)."""
    ok = abs(recomputed - reported) <= tol
    mark = "matches" if ok else "DIFFERS from"
    return f"{label}: recomputed {recomputed:.4f} {mark} the report ({reported:.4f})"


__all__ = [
    "REPO", "BACKEND", "EXPERIMENTS", "EXPERIMENTS_DATA", "COLORS", "GREY",
    "np", "pd", "plt", "settings", "session_factory", "sql", "compare",
]
