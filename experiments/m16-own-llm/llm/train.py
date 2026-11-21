"""Trainer and CLI: start / pause / resume / status (0025 §7). See TRAINING.md.

    python -m llm.train start configs/stage_a.yaml   # new run, detached
    python -m llm.train pause                        # checkpoint + exit, frees the GPU
    python -m llm.train resume                       # continue the latest run exactly
    python -m llm.train status

`start`/`resume` run in the background (own session, logs in the run dir), so
closing the terminal does not stop training. `--foreground` keeps it attached;
then Ctrl+C pauses the same way `pause` does.

Pausing never waits for a whole optimizer step: the stop request is checked
between micro-batches. If it lands mid-step, the partial gradients are dropped
and the checkpoint records the loader position *at the start of that step*, so
the resumed run redoes it from the same rows: the trajectory is identical to an
uninterrupted run (tests/test_resume.py).

Run directory (`$M16_HOME/runs/<name>/`):
    config.yaml       the run's config (edit only RESUME_MUTABLE fields between pause and resume)
    ckpt/step_XXXXXXXX.pt  full checkpoints (last `keep_last` kept), written atomically
    ckpt/permanent/   model-only weights every `permanent_every_fraction` of the budget
    metrics.jsonl     one line per log/eval event
    samples.md        fixed-prompt generations at every eval
    status.json       state, progress, ETA (what `status` prints)
    train.out         stdout/stderr of the detached process
    PAUSE             present = a pause was requested
    pid               pid of the running trainer
"""

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from llm import HOME
from llm import config as cfgmod
from llm.data import MixedLoader, SFTLoader, val_rows
from llm.model import Decoder
from llm.optim import build_optimizer

PKG_ROOT = Path(__file__).resolve().parent.parent


def runs_root() -> Path:
    return Path(os.environ.get("M16_RUNS", HOME / "runs")).expanduser()


# --------------------------------------------------------------------------- run directory helpers

def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (exited, not yet reaped) still answers kill(0); check /proc on Linux.
    try:
        return Path(f"/proc/{pid}/status").read_text().split("State:")[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def running_pid(run_dir: Path) -> int | None:
    p = run_dir / "pid"
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except ValueError:
        return None
    return pid if pid_alive(pid) else None


def checkpoints(run_dir: Path) -> list[Path]:
    """Complete full checkpoints, oldest first. `*.tmp` files (interrupted writes) are ignored."""
    return sorted((run_dir / "ckpt").glob("step_*.pt"))


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def write_status(run_dir: Path, **kw):
    s = read_status(run_dir)
    s.update(kw, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(run_dir / "status.json")


def resolve_run(name: str | None) -> Path:
    root = runs_root()
    if name:
        d = root / name
        if not d.is_dir():
            sys.exit(f"No run named '{name}' in {root}.")
        return d
    latest = root / "latest"
    if latest.exists():
        return Path(os.path.realpath(latest))
    sys.exit(f"No runs yet in {root}. Start one with: python -m llm.train start <config.yaml>")


# --------------------------------------------------------------------------- schedule

def lr_at(cfg: cfgmod.TrainConfig, tokens: int) -> float:
    """Warmup-stable-decay (0025 §3): linear warmup, flat, linear decay over the last `decay_fraction`."""
    if tokens < cfg.warmup_tokens:
        return cfg.lr * (tokens + 1) / cfg.warmup_tokens
    if cfg.schedule == "cosine":
        frac = min(1.0, (tokens - cfg.warmup_tokens) / max(1, cfg.total_tokens - cfg.warmup_tokens))
        return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))
    decay_start = cfg.total_tokens * (1 - cfg.decay_fraction)
    if tokens < decay_start:
        return cfg.lr
    frac = min(1.0, (tokens - decay_start) / max(1, cfg.total_tokens - decay_start))
    return cfg.lr * (1 - frac * (1 - cfg.min_lr_ratio))


def phase(cfg: cfgmod.TrainConfig, tokens: int) -> tuple[int, int]:
    """(seq_len, micro_batch) at this point of the budget (context extension, 0025 §8)."""
    if cfg.long_from_fraction is not None and tokens >= cfg.long_from_fraction * cfg.total_tokens:
        return cfg.long_seq_len, cfg.long_micro_batch
    return cfg.seq_len, cfg.micro_batch


# --------------------------------------------------------------------------- RNG state

def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# --------------------------------------------------------------------------- trainer

class StopRequested(Exception):
    pass


class Trainer:
    def __init__(self, run_dir: Path, on_step=None):
        self.run_dir = Path(run_dir)
        self.cfg = cfgmod.load(self.run_dir / "config.yaml")
        self.on_step = on_step  # test hook: called after every optimizer step with the trainer
        self.stop = False

    # -- setup

    def _setup(self):
        cfg = self.cfg
        self.dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        if self.dev.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.model = Decoder(cfg.model).to(self.dev)
        self.opt = build_optimizer(self.model, cfg, self.dev.type)
        if cfg.data.kind == "sft":
            self.loader = SFTLoader(cfg.data.sft_dir)
        elif cfg.data.kind == "dpo":
            from llm.dpo import DPOLoader

            self.loader = DPOLoader(cfg.data.sft_dir)
        else:
            self.loader = MixedLoader(cfg.data.root, cfg.data.sources)
        self.step, self.tokens, self.train_seconds = 0, 0, 0.0
        self.next_permanent = cfg.permanent_every_fraction
        self.next_eval = cfg.eval_every_tokens
        if not checkpoints(self.run_dir) and cfg.init_from:
            ck = torch.load(Path(cfg.init_from).expanduser(), map_location="cpu", weights_only=False)
            src = cfgmod.from_dict(ck["config"]).model if "config" in ck else None
            if src is not None and asdict(src) != asdict(cfg.model):
                raise SystemExit(f"init_from model config differs from this run's model config: {cfg.init_from}")
            self.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
            self.log({"event": "init_from", "path": str(cfg.init_from)})
        self._load_latest()
        self.fwd = torch.compile(self.model) if (cfg.compile and self.dev.type == "cuda") else self.model
        self.autocast = (torch.autocast("cuda", dtype=getattr(torch, cfg.dtype)) if self.dev.type == "cuda"
                         else torch.autocast("cpu", enabled=False))
        self._val = None

    def _load_latest(self):
        cks = checkpoints(self.run_dir)
        if not cks:
            return
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        saved = cfgmod.from_dict(ck["config"])
        bad = cfgmod.diff_keys(saved, self.cfg)
        if bad:
            raise SystemExit(
                f"config.yaml no longer matches the checkpoint ({cks[-1].name}); changed: {', '.join(bad)}.\n"
                f"Only these may change between pause and resume: {', '.join(sorted(cfgmod.RESUME_MUTABLE))}.")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.loader.load_state_dict(ck["loader"])
        self.step, self.tokens, self.train_seconds = ck["step"], ck["tokens"], ck["train_seconds"]
        self.next_permanent, self.next_eval = ck["next_permanent"], ck["next_eval"]
        set_rng_state(ck["rng"])
        self.log({"event": "resumed", "from": cks[-1].name, "step": self.step, "tokens": self.tokens})

    # -- checkpointing

    def save(self, loader_state: dict, rng: dict):
        """Atomic: write `*.tmp`, fsync, rename. Older full checkpoints beyond `keep_last` are deleted."""
        ckdir = self.run_dir / "ckpt"
        ckdir.mkdir(exist_ok=True)
        path = ckdir / f"step_{self.step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        state = {
            "model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "loader": loader_state,
            "rng": rng, "step": self.step, "tokens": self.tokens, "train_seconds": self.train_seconds,
            "next_permanent": self.next_permanent, "next_eval": self.next_eval,
            "config": asdict(self.cfg), "config_hash": self.cfg.hash(),
        }
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        for old in checkpoints(self.run_dir)[: -self.cfg.keep_last]:
            old.unlink(missing_ok=True)
        self.last_save = time.monotonic()
        self.log({"event": "checkpoint", "file": path.name, "step": self.step, "tokens": self.tokens})

    def save_permanent(self):
        d = self.run_dir / "ckpt" / "permanent"
        d.mkdir(parents=True, exist_ok=True)
        pct = round(100 * self.tokens / self.cfg.total_tokens)
        tmp = d / f"model_{pct:03d}pct.pt.tmp"
        torch.save({"model": self.model.state_dict(), "tokens": self.tokens, "config": asdict(self.cfg)}, tmp)
        tmp.replace(tmp.with_suffix(""))

    # -- logging

    def log(self, rec: dict):
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- evaluation

    def _dpo(self, batch):
        """(loss, preference accuracy) for one micro-batch of pairs (llm.dpo)."""
        from llm.dpo import dpo_loss, seq_logprob

        xc, yc, mc, xr, yr, mr, rc, rr = batch
        logits = self.fwd(torch.cat([xc, xr]))
        pc, pr = seq_logprob(logits, torch.cat([yc, yr]), torch.cat([mc, mr])).chunk(2)
        return dpo_loss(pc, pr, rc, rr, self.cfg.dpo_beta)

    @torch.no_grad()
    def evaluate(self) -> dict:
        cfg = self.cfg
        if cfg.objective == "dpo":
            self.model.eval()
            losses, accs = [], []
            for b in self.loader.val_batches(cfg.micro_batch, cfg.eval_batches * cfg.micro_batch):
                with self.autocast:
                    loss, acc = self._dpo([t.to(self.dev) for t in b])
                losses.append(loss.item())
                accs.append(acc.item())
            self.model.train()
            return {"val_loss_dpo": sum(losses) / len(losses), "val_pref_acc": sum(accs) / len(accs)} if losses else {}
        if cfg.data.kind == "sft":
            t, m = self.loader.val(cfg.eval_batches * cfg.micro_batch)
            if not len(t):
                return {}
            self.model.eval()
            num = den = 0.0
            for i in range(0, len(t), cfg.micro_batch):
                b, mb = t[i : i + cfg.micro_batch].to(self.dev), m[i : i + cfg.micro_batch, 1:].to(self.dev)
                with self.autocast:
                    loss = self.fwd(b[:, :-1], b[:, 1:], mb).item()
                num += loss * float(mb.sum())
                den += float(mb.sum())
            self.model.train()
            return {"val_loss_sft": num / max(den, 1.0)}
        if self._val is None:
            self._val = {}
            root = Path(cfg.data.root).expanduser()
            for src in cfg.data.sources:
                v = val_rows(root / src, cfg.seq_len, cfg.eval_batches * cfg.micro_batch)
                if v is not None:
                    self._val[src] = v
        self.model.eval()
        out = {}
        for src, (rows, bytes_per_tok) in self._val.items():
            losses = []
            for i in range(0, len(rows), cfg.micro_batch):
                b = rows[i : i + cfg.micro_batch].to(self.dev)
                with self.autocast:
                    losses.append(self.fwd(b[:, :-1], b[:, 1:]).item())
            loss = sum(losses) / len(losses)
            out[f"val_loss_{src}"] = loss
            out[f"val_bpb_{src}"] = loss / math.log(2) / bytes_per_tok
        self.model.train()
        if out:
            bpbs = [v for k, v in out.items() if k.startswith("val_bpb_")]
            out["val_bpb"] = sum(bpbs) / len(bpbs)
        return out

    @torch.no_grad()
    def samples(self):
        cfg = self.cfg
        if not cfg.sample_prompts or not cfg.tokenizer:
            return
        from llm.tokenizer import load_tokenizer

        tok = load_tokenizer(cfg.tokenizer)
        eos = tok.token_to_id("<|endoftext|>")
        self.model.eval()
        lines = [f"\n## step {self.step} · {self.tokens / 1e9:.2f}B tokens\n"]
        for p in cfg.sample_prompts:
            ids = torch.tensor([tok.encode(p).ids], device=self.dev)
            with self.autocast:
                out = self.model.generate(ids, cfg.sample_tokens, eos)
            lines.append(f"- **{p}**{tok.decode(out[0, ids.shape[1]:].tolist())}")
        self.model.train()
        with open(self.run_dir / "samples.md", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # -- main loop

    def _check_stop(self):
        if self.stop or (self.run_dir / "PAUSE").exists():
            raise StopRequested

    def run(self) -> str:
        """Train until the budget is spent ("done") or a pause is requested ("paused")."""
        self._setup()
        cfg = self.cfg
        write_status(self.run_dir, state="running", pid=os.getpid(), name=cfg.name,
                     total_tokens=cfg.total_tokens, step=self.step, tokens=self.tokens)
        self.last_save = time.monotonic()
        ema_tps = None
        self.model.train()
        while self.tokens < cfg.total_tokens:
            seq_len, micro = phase(cfg, self.tokens)
            accum = max(1, cfg.batch_tokens // (micro * seq_len))
            loader_before, rng_before = self.loader.state_dict(), rng_state()
            lr = lr_at(cfg, self.tokens)
            for g in self.opt.param_groups:  # each group scales its own base LR by the WSD factor
                g["lr"] = g["base_lr"] * lr / cfg.lr
            t0 = time.perf_counter()
            try:
                total_loss = 0.0
                for _ in range(accum):
                    self._check_stop()
                    batch = [t.to(self.dev, non_blocking=True) for t in self.loader.next_batch(micro, seq_len)]
                    with self.autocast:
                        if cfg.objective == "dpo":
                            loss = self._dpo(batch)[0] / accum
                        else:
                            loss = self.fwd(*batch) / accum  # (x, y) or (x, y, loss_mask) for SFT
                    loss.backward()
                    total_loss += loss.detach()
            except StopRequested:
                # Mid-step: drop the partial gradients and checkpoint the state before this step.
                self.opt.zero_grad(set_to_none=True)
                self.save(loader_before, rng_before)
                write_status(self.run_dir, state="paused", step=self.step, tokens=self.tokens)
                self.log({"event": "paused", "step": self.step, "tokens": self.tokens})
                return "paused"
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.train_seconds += dt
            step_tokens = accum * micro * seq_len
            self.step += 1
            self.tokens += step_tokens
            tps = step_tokens / dt
            ema_tps = tps if ema_tps is None else 0.9 * ema_tps + 0.1 * tps
            loss_v = float(total_loss)

            if self.step % cfg.log_every_steps == 0 or self.step == 1:
                self.log({"event": "train", "step": self.step, "tokens": self.tokens, "loss": loss_v, "lr": lr,
                          "grad_norm": float(gnorm), "tok_s": tps, "seq_len": seq_len})
                write_status(self.run_dir, state="running", step=self.step, tokens=self.tokens, loss=loss_v,
                             tok_s=ema_tps, eta_hours=(cfg.total_tokens - self.tokens) / ema_tps / 3600,
                             epochs=self.loader.epochs())
            if self.tokens >= self.next_eval:
                ev = self.evaluate()
                self.log({"event": "eval", "step": self.step, "tokens": self.tokens, **ev})
                write_status(self.run_dir, **ev)
                self.samples()
                self.next_eval += cfg.eval_every_tokens
            if self.tokens >= self.next_permanent * cfg.total_tokens:
                self.save_permanent()
                self.next_permanent += cfg.permanent_every_fraction
            if time.monotonic() - self.last_save >= cfg.autosave_minutes * 60:
                self.save(self.loader.state_dict(), rng_state())
            if self.on_step:
                self.on_step(self)

        ev = self.evaluate()
        self.log({"event": "eval", "step": self.step, "tokens": self.tokens, "final": True, **ev})
        self.samples()
        self.save(self.loader.state_dict(), rng_state())
        torch.save({"model": self.model.state_dict(), "config": asdict(cfg)}, self.run_dir / "final_model.pt")
        write_status(self.run_dir, state="done", step=self.step, tokens=self.tokens, **ev)
        self.log({"event": "done", "step": self.step, "tokens": self.tokens})
        return "done"


# --------------------------------------------------------------------------- queue

def queue_path() -> Path:
    return runs_root() / "queue.json"


def read_queue() -> list[dict]:
    try:
        return json.loads(queue_path().read_text())
    except (OSError, ValueError):
        return []


def write_queue(q: list[dict]):
    tmp = queue_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(q, indent=1))
    tmp.replace(queue_path())


def prepare_run(cfg: cfgmod.TrainConfig) -> Path:
    run_dir = runs_root() / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    write_status(run_dir, state="starting", name=cfg.name, total_tokens=cfg.total_tokens, step=0, tokens=0)
    return run_dir


def set_latest(run_dir: Path):
    link = runs_root() / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run_dir.name)


def pop_queue() -> Path | None:
    """The next queued run that isn't finished yet (its directory prepared), or None."""
    q = read_queue()
    while q:
        cfg = cfgmod.from_dict(q.pop(0))
        write_queue(q)
        d = runs_root() / cfg.name
        if read_status(d).get("state") == "done":
            continue
        if not checkpoints(d):
            d = prepare_run(cfg)
        return d
    return None


def _run_one(run_dir: Path) -> str:
    (run_dir / "pid").write_text(str(os.getpid()))
    tr = Trainer(run_dir)

    def handler(signum, _frame):
        tr.stop = True
        print(f"\nSignal {signal.Signals(signum).name}: pausing after the current micro-batch...", flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    try:
        result = tr.run()
        print(f"Trainer finished '{run_dir.name}': {result}", flush=True)
        return result
    except BaseException as e:
        write_status(run_dir, state="crashed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (run_dir / "pid").unlink(missing_ok=True)
        del tr
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch._dynamo.reset()
            torch.cuda.empty_cache()


def run_in_process(run_dir: Path) -> str:
    """Entry point of the trainer process: this run, then any queued ones, until a pause or the end."""
    run_dir = Path(run_dir)
    while True:
        result = _run_one(run_dir)
        if result != "done":
            return result
        nxt = pop_queue()
        if nxt is None:
            return result
        print(f"Queue: starting '{nxt.name}' ({len(read_queue())} more after it)", flush=True)
        set_latest(nxt)
        run_dir = nxt


# --------------------------------------------------------------------------- CLI

def free_vram_gib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return float(out[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def launch(run_dir: Path, foreground: bool, force: bool = False):
    cfg = cfgmod.load(run_dir / "config.yaml")
    pid = running_pid(run_dir)
    if pid:
        sys.exit(f"'{run_dir.name}' is already running (pid {pid}).")
    if cfg.device == "cuda" and not force:
        free = free_vram_gib()
        if free is not None and free < cfg.min_free_vram_gib:
            sys.exit(f"Only {free:.1f} GiB of VRAM free (need {cfg.min_free_vram_gib}). Something else is using "
                     "the GPU: close the game or stop the app worker, then try again (or pass --force).")
    (run_dir / "PAUSE").unlink(missing_ok=True)
    set_latest(run_dir)
    cmd =[sys.executable, "-m", "llm.train", "_run", str(run_dir)]
    if foreground:
        print(f"Training '{run_dir.name}' in the foreground. Ctrl+C pauses (checkpoint, then exit).")
        os.chdir(PKG_ROOT)
        sys.exit(0 if run_in_process(run_dir) in ("done", "paused") else 1)
    with open(run_dir / "train.out", "a") as out:
        p = subprocess.Popen(cmd, cwd=PKG_ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    print(f"Training '{run_dir.name}' in the background (pid {p.pid}).")
    print("  status : python -m llm.train status")
    print("  pause  : python -m llm.train pause")
    print(f"  log    : {run_dir / 'train.out'}")


def cmd_start(a):
    """Start the first config; any further configs are queued and run one after another."""
    cfgs = [cfgmod.load(p) for p in a.config]
    names = [c.name for c in cfgs]
    if len(set(names)) != len(names):
        sys.exit(f"Duplicate run names in: {names}")
    for c in cfgs:
        if checkpoints(runs_root() / c.name):
            sys.exit(f"Run '{c.name}' already has checkpoints. Use: python -m llm.train resume {c.name}")
    if read_queue():
        sys.exit(f"A queue is already pending ({[q['name'] for q in read_queue()]}). Finish it or delete "
                 f"{queue_path()}.")
    if any(running_pid(d) for d in runs_root().iterdir() if d.is_dir() and not d.is_symlink()):
        sys.exit("Another run is training. Pause it first (python -m llm.train pause).")
    write_queue([asdict(c) for c in cfgs[1:]])
    if len(cfgs) > 1:
        print(f"Queued after it: {', '.join(names[1:])}")
    launch(prepare_run(cfgs[0]), a.foreground, a.force)


def cmd_resume(a):
    run_dir = resolve_run(a.run)
    st = read_status(run_dir)
    if st.get("state") == "done":
        nxt = pop_queue()
        if nxt is None:
            sys.exit(f"'{run_dir.name}' already finished and the queue is empty. "
                     "Raise total_tokens in its config.yaml to continue it.")
        print(f"'{run_dir.name}' is finished; continuing the queue with '{nxt.name}'.")
        run_dir = nxt
    launch(run_dir, a.foreground, a.force)


def cmd_pause(a):
    run_dir = resolve_run(a.run)
    pid = running_pid(run_dir)
    if not pid:
        print(f"'{run_dir.name}' is not running; nothing to pause.")
        return
    (run_dir / "PAUSE").touch()
    print(f"Pause requested for '{run_dir.name}' (pid {pid}); waiting for the checkpoint...", flush=True)
    t0 = time.time()
    while pid_alive(pid) and time.time() - t0 < a.timeout:
        time.sleep(1)
    if pid_alive(pid):
        print(f"Still running after {a.timeout}s (a checkpoint may be writing). It will stop on its own; "
              "check with: python -m llm.train status")
    else:
        st = read_status(run_dir)
        print(f"Paused at step {st.get('step')} ({st.get('tokens', 0) / 1e9:.3f}B tokens) after "
              f"{time.time() - t0:.0f}s. GPU is free. Continue with: python -m llm.train resume")


def fmt_status(run_dir: Path) -> str:
    st = read_status(run_dir)
    pid = running_pid(run_dir)
    state = st.get("state", "unknown")
    if state in ("running", "starting") and not pid:
        state = "stopped unexpectedly (crash, reboot or power cut) -> python -m llm.train resume"
    if pid and (run_dir / "PAUSE").exists():
        state = "pausing..."
    total = st.get("total_tokens") or 1
    tokens = st.get("tokens", 0)
    lines = [f"run      : {run_dir.name}  ({run_dir})", f"state    : {state}" + (f"  (pid {pid})" if pid else ""),
             f"progress : step {st.get('step', 0)} · {tokens / 1e9:.3f}B / {total / 1e9:.3f}B tokens "
             f"({100 * tokens / total:.1f} %)"]
    if st.get("tok_s"):
        lines.append(f"speed    : {st['tok_s']:,.0f} tok/s · ETA {st.get('eta_hours', 0):.1f} h of training")
    if "loss" in st:
        lines.append(f"loss     : {st['loss']:.4f}" + (f" · val bpb {st['val_bpb']:.4f}" if "val_bpb" in st else ""))
    if st.get("error"):
        lines.append(f"error    : {st['error']}")
    cks = checkpoints(run_dir)
    lines.append(f"last ckpt: {cks[-1].name if cks else 'none'} · updated {st.get('updated_at', '?')}")
    if (run_dir / "samples.md").exists():
        lines.append(f"samples  : {run_dir / 'samples.md'}")
    q = read_queue()
    if q:
        lines.append(f"queue    : {len(q)} more after this one: {', '.join(c['name'] for c in q)}")
    return "\n".join(lines)


def cmd_status(a):
    if a.run is None and not (runs_root() / "latest").exists():
        print(f"No runs yet in {runs_root()}.")
        return
    print(fmt_status(resolve_run(a.run)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m llm.train", description="M16 trainer (see TRAINING.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="start a new run from a config (several configs = a queue)")
    s.add_argument("config", nargs="+")
    s.add_argument("--foreground", action="store_true")
    s.add_argument("--force", action="store_true", help="skip the free-VRAM check")
    s.set_defaults(fn=cmd_start)
    r = sub.add_parser("resume", help="continue a run (default: the latest) from its last checkpoint")
    r.add_argument("run", nargs="?")
    r.add_argument("--foreground", action="store_true")
    r.add_argument("--force", action="store_true", help="skip the free-VRAM check")
    r.set_defaults(fn=cmd_resume)
    p = sub.add_parser("pause", help="checkpoint and stop a run (default: the latest)")
    p.add_argument("run", nargs="?")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(fn=cmd_pause)
    t = sub.add_parser("status", help="show a run's progress (default: the latest)")
    t.add_argument("run", nargs="?")
    t.set_defaults(fn=cmd_status)
    i = sub.add_parser("_run", help=argparse.SUPPRESS)
    i.add_argument("run_dir")
    i.set_defaults(fn=lambda a: sys.exit(0 if run_in_process(Path(a.run_dir)) in ("done", "paused") else 1))
    a = ap.parse_args(argv)
    runs_root().mkdir(parents=True, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
