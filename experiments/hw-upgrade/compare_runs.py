"""Per-episode, per-tool job seconds: M9 season-1 run vs the RTX 5070 runs.

Usage (any python3):
    python experiments/hw-upgrade/compare_runs.py experiments/hw-upgrade/runA_m9_settings.csv [more.csv ...]

Episodes are matched on the YouTube ID in brackets in the file name.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

M9 = Path(__file__).resolve().parent.parent / "m9-perf" / "season1_run.csv"
TOOLS = ("transcribe", "diarize", "chapters", "relevance", "shownotes", "shorts")


def load(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"\[([\w-]{11})\]", row["episode"])
            if m and row["status"] == "succeeded":
                out[m.group(1)][row["tool"]] = float(row["seconds"])  # last success wins
    return out


def main(paths: list[str]) -> None:
    runs = {"M9": load(M9)} | {Path(p).stem: load(Path(p)) for p in paths}
    episodes = sorted(set().union(*(set(r) for name, r in runs.items() if name != "M9")))
    for tool in TOOLS:
        if not any(tool in r.get(e, {}) for name, r in runs.items() if name != "M9" for e in episodes):
            continue
        print(f"\n{tool}")
        print("episode      " + "".join(f"{n:>22}" for n in runs))
        for e in episodes:
            cells = []
            for name, r in runs.items():
                v = r.get(e, {}).get(tool)
                base = runs["M9"].get(e, {}).get(tool)
                if v is None:
                    cells.append(f"{'-':>22}")
                elif name == "M9" or not base:
                    cells.append(f"{v:>21.1f}s")
                else:
                    cells.append(f"{v:>12.1f}s ({base / v:4.1f}x)")
            print(f"{e}  " + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
