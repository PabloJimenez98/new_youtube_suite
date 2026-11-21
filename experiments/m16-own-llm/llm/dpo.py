"""DPO (0025 §9 step 3): preference pairs -> rows, frozen reference log-probs, the loss.

Pairs JSONL, one per line: {"messages": [...prompt turns...], "chosen": str, "rejected": str, ...}.

    python -m llm.dpo build --out DIR --seq-len 4096 PAIRS.jsonl ...   # tokenize + pad (val via "split")
    python -m llm.dpo ref DIR WEIGHTS.pt                                 # reference log-probs (the SFT model)

Then a run with `objective: dpo` and `data: {kind: dpo, sft_dir: DIR}` trains the policy,
normally with `init_from` = the same SFT weights. Loss (Rafailov et al. 2023):
    -log sigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r))))
with log-probs summed over the response tokens only.

Directory layout: {train,val}_{chosen,rejected}_{tokens,mask}.bin (rows of seq_len+1),
{train,val}_ref.npy (n x 2 float32: ref log-prob of chosen, rejected), index.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from llm.data import DTYPE
from llm.sft_data import encode_example, read_jsonl
from llm.tokenizer import PAD, load_tokenizer


def seq_logprob(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(target) over masked positions, per row. logits (B,T,V), targets/mask (B,T)."""
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def dpo_loss(pc, pr, rc, rr, beta: float):
    """Mean DPO loss and the fraction of pairs where the policy's implicit reward prefers `chosen`."""
    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean()


def _rows(tok, items, row_len, pad):
    ids = np.full((len(items), row_len), pad, dtype=DTYPE)
    mask = np.zeros((len(items), row_len), dtype=np.uint8)
    for i, (a, m) in enumerate(items):
        ids[i, : len(a)] = a
        mask[i, : len(m)] = m
    return ids, mask


def build(files: list[Path], tokenizer_path, out: Path, seq_len: int) -> dict:
    tok = load_tokenizer(tokenizer_path)
    pad = tok.token_to_id(PAD)
    row_len = seq_len + 1
    data = {"train": ([], []), "val": ([], [])}
    dropped = 0
    for f in files:
        for ex in read_jsonl(f):
            c = encode_example(tok, ex["messages"] + [{"role": "assistant", "content": ex["chosen"]}])
            r = encode_example(tok, ex["messages"] + [{"role": "assistant", "content": ex["rejected"]}])
            if max(len(c[0]), len(r[0])) > row_len:
                dropped += 1
                continue
            split = "val" if ex.get("split") == "val" else "train"
            data[split][0].append(c)
            data[split][1].append(r)
    out.mkdir(parents=True, exist_ok=True)
    for split, (cs, rs) in data.items():
        for name, items in (("chosen", cs), ("rejected", rs)):
            ids, mask = _rows(tok, items, row_len, pad)
            ids.tofile(out / f"{split}_{name}_tokens.bin")
            mask.tofile(out / f"{split}_{name}_mask.bin")
    idx = {"seq_len": seq_len, "pairs": len(data["train"][0]), "val_pairs": len(data["val"][0]),
           "dropped_too_long": dropped, "has_ref": False, "tokenizer": str(tokenizer_path)}
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


class DPOLoader:
    """Batches of pairs in a fixed order; the state is one row counter (exact pause/resume)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory).expanduser()
        self.index = json.loads((self.dir / "index.json").read_text())
        if not self.index.get("has_ref"):
            raise ValueError(f"{self.dir}: no reference log-probs yet (python -m llm.dpo ref ...)")
        self.seq_len = self.index["seq_len"]
        self.split = {s: self._load(s) for s in ("train", "val")}
        self.row = 0

    def _load(self, split):
        n = self.index["pairs" if split == "train" else "val_pairs"]
        if n == 0:
            return None
        row_len = self.seq_len + 1
        arr = {}
        for name in ("chosen", "rejected"):
            arr[f"{name}_t"] = np.memmap(self.dir / f"{split}_{name}_tokens.bin", dtype=DTYPE, mode="r").reshape(n, row_len)
            arr[f"{name}_m"] = np.memmap(self.dir / f"{split}_{name}_mask.bin", dtype=np.uint8, mode="r").reshape(n, row_len)
        arr["ref"] = np.load(self.dir / f"{split}_ref.npy")
        return arr

    def _batch(self, arr, idx):
        out = []
        for name in ("chosen", "rejected"):
            t = torch.from_numpy(arr[f"{name}_t"][idx].astype(np.int64))
            m = torch.from_numpy(arr[f"{name}_m"][idx].astype(np.float32))
            out += [t[:, :-1], t[:, 1:], m[:, 1:]]
        ref = torch.from_numpy(arr["ref"][idx])
        return (*out, ref[:, 0], ref[:, 1])

    def next_batch(self, batch: int, seq_len: int):
        if seq_len != self.seq_len:
            raise ValueError(f"config seq_len {seq_len} != DPO data seq_len {self.seq_len}")
        arr = self.split["train"]
        n = len(arr["ref"])
        idx = [(self.row + i) % n for i in range(batch)]
        self.row += batch
        return self._batch(arr, idx)

    def val_batches(self, batch: int, max_pairs: int):
        arr = self.split["val"]
        if arr is None:
            return
        n = min(len(arr["ref"]), max_pairs)
        for i in range(0, n, batch):
            yield self._batch(arr, list(range(i, min(n, i + batch))))

    def state_dict(self):
        return {"row": self.row}

    def load_state_dict(self, d):
        self.row = d["row"]

    def epochs(self):
        return {"dpo": round(self.row / max(1, self.index["pairs"]), 3)}


@torch.no_grad()
def compute_ref(directory: Path, weights: Path, device: str = "cuda", batch: int = 4):
    """Reference log-probs of every pair with the frozen reference model (normally the SFT model)."""
    from llm import config as cfgmod
    from llm.model import Decoder

    ck = torch.load(Path(weights).expanduser(), map_location="cpu", weights_only=False)
    model = Decoder(cfgmod.from_dict(ck["config"]).model)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    idx = json.loads((directory / "index.json").read_text())
    row_len = idx["seq_len"] + 1
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    for split, key in (("train", "pairs"), ("val", "val_pairs")):
        n = idx[key]
        ref = np.zeros((n, 2), dtype=np.float32)
        for j, name in enumerate(("chosen", "rejected")):
            if n == 0:
                break
            t = np.memmap(directory / f"{split}_{name}_tokens.bin", dtype=DTYPE, mode="r").reshape(n, row_len)
            m = np.memmap(directory / f"{split}_{name}_mask.bin", dtype=np.uint8, mode="r").reshape(n, row_len)
            for i in range(0, n, batch):
                tt = torch.from_numpy(t[i : i + batch].astype(np.int64)).to(dev)
                mm = torch.from_numpy(m[i : i + batch].astype(np.float32)).to(dev)
                with ctx:
                    logits = model(tt[:, :-1])
                ref[i : i + batch, j] = seq_logprob(logits, tt[:, 1:], mm[:, 1:]).cpu().numpy()
        np.save(directory / f"{split}_ref.npy", ref)
    idx["has_ref"] = True
    idx["ref_weights"] = str(weights)
    (directory / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def main(argv=None):
    from llm import prepare_data

    ap = argparse.ArgumentParser(prog="python -m llm.dpo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seq-len", type=int, default=4096)
    b.add_argument("--tokenizer", default=str(prepare_data.tokenizer_path()))
    b.add_argument("files", nargs="+")
    r = sub.add_parser("ref")
    r.add_argument("dir")
    r.add_argument("weights")
    r.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build([Path(f).expanduser() for f in a.files], a.tokenizer, Path(a.out).expanduser(), a.seq_len), indent=1))
    else:
        print(json.dumps(compute_ref(Path(a.dir).expanduser(), Path(a.weights), a.device), indent=1))


if __name__ == "__main__":
    main()
