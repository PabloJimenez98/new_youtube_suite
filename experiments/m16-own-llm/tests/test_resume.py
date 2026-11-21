"""Pause/resume (the owner's key requirement, 0025 §7): a paused and resumed run is identical to an
uninterrupted one; PAUSE and SIGTERM give a clean checkpointed exit; resume refuses a changed config;
an interrupted checkpoint write never hides the previous checkpoint."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from llm import train
from llm.train import Trainer, checkpoints, read_status
from tests.conftest import tiny_config, write_run

PKG = Path(__file__).resolve().parent.parent


def losses(run_dir):
    out = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "train":
            out[r["step"]] = r["loss"]
    return out


def final_state(run_dir):
    return torch.load(run_dir / "final_model.pt", weights_only=False)["model"]


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_uninterrupted_vs_paused_resumed_identical(shards, runs, optimizer):
    a = write_run(runs, tiny_config("a", shards, optimizer=optimizer))
    assert Trainer(a).run() == "done"

    b = write_run(runs, tiny_config("b", shards, optimizer=optimizer))

    def pause_at_5(tr):
        if tr.step == 5:
            (tr.run_dir / "PAUSE").touch()

    assert Trainer(b, on_step=pause_at_5).run() == "paused"
    assert read_status(b)["state"] == "paused"
    assert read_status(b)["step"] == 5
    (b / "PAUSE").unlink()
    assert Trainer(b).run() == "done"

    sa, sb = final_state(a), final_state(b)
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    la, lb = losses(a), losses(b)
    assert la == lb  # every step's loss, including the ones after the resume


def test_pause_mid_step_redoes_the_step(shards, runs):
    """A stop request between micro-batches drops partial grads and resumes from the step start."""
    a = write_run(runs, tiny_config("a", shards))
    Trainer(a).run()

    b = write_run(runs, tiny_config("b", shards))
    tr = Trainer(b)
    calls = {"n": 0}
    real_next = None

    def stop_on_second_microbatch(batch, seq_len):
        calls["n"] += 1
        # 2 micro-batches per step: call 7 is step 4's first. The stop is seen before its second,
        # so step 4 is abandoned half-way.
        if calls["n"] == 2 * 3 + 1:
            tr.stop = True
        return real_next(batch, seq_len)

    tr._setup()
    real_next = tr.loader.next_batch
    tr.loader.next_batch = stop_on_second_microbatch
    tr._setup = lambda: None  # already set up; keep the patched loader
    assert tr.run() == "paused"
    assert read_status(b)["step"] == 3
    assert Trainer(b).run() == "done"
    for k, v in final_state(a).items():
        assert torch.equal(v, final_state(b)[k]), k


def test_resume_refuses_changed_config(shards, runs):
    b = write_run(runs, tiny_config("b", shards))
    Trainer(b, on_step=lambda tr: (tr.run_dir / "PAUSE").touch() if tr.step == 2 else None).run()
    (b / "PAUSE").unlink()
    write_run(runs, tiny_config("b", shards, lr=1e-2))  # not resume-mutable
    with pytest.raises(SystemExit, match="lr"):
        Trainer(b).run()


def test_resume_allows_mutable_change(shards, runs):
    cfg = tiny_config("b", shards)
    b = write_run(runs, cfg)
    Trainer(b, on_step=lambda tr: (tr.run_dir / "PAUSE").touch() if tr.step == 2 else None).run()
    (b / "PAUSE").unlink()
    write_run(runs, tiny_config("b", shards, total_tokens=cfg.total_tokens * 2, eval_batches=1))
    assert Trainer(b).run() == "done"
    assert read_status(b)["tokens"] == cfg.total_tokens * 2


def test_interrupted_write_keeps_previous_checkpoint(shards, runs):
    b = write_run(runs, tiny_config("b", shards))
    Trainer(b, on_step=lambda tr: (tr.run_dir / "PAUSE").touch() if tr.step == 3 else None).run()
    (b / "PAUSE").unlink()
    good = checkpoints(b)[-1]
    # Simulate a crash mid-write of a later checkpoint: a truncated .tmp next to it.
    (b / "ckpt" / "step_00000009.pt.tmp").write_bytes(b"\x80truncated")
    assert checkpoints(b)[-1] == good
    tr = Trainer(b)
    tr._setup()
    assert tr.step == 3


def test_keep_last_and_permanent(shards, runs):
    b = write_run(runs, tiny_config("b", shards, autosave_minutes=0))  # checkpoint every step
    Trainer(b).run()
    assert len(checkpoints(b)) == 2
    perm = sorted(p.name for p in (b / "ckpt" / "permanent").glob("*.pt"))
    assert perm == ["model_050pct.pt", "model_100pct.pt"]


def test_sigterm_pauses_cleanly(shards, runs):
    b = write_run(runs, tiny_config("b", shards, total_tokens=10**9))
    p = subprocess.Popen([sys.executable, "-m", "llm.train", "_run", str(b)], cwd=PKG,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t0 = time.time()
    while not (b / "metrics.jsonl").exists() or "train" not in (b / "metrics.jsonl").read_text():
        assert p.poll() is None, p.stdout.read().decode()
        assert time.time() - t0 < 120
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 0, p.stdout.read().decode()
    assert read_status(b)["state"] == "paused"
    assert checkpoints(b)
    assert not (b / "pid").exists()


def test_cli_status_and_pause_when_idle(shards, runs, capsys):
    b = write_run(runs, tiny_config("b", shards))
    (runs / "latest").symlink_to("b")
    train.main(["status"])
    assert "run      : b" in capsys.readouterr().out
    train.main(["pause"])
    assert "not running" in capsys.readouterr().out


def test_queue_runs_in_order_and_survives_pause(shards, runs, monkeypatch):
    """`start a b c`: a, then b, then c. A pause in b stops the queue; resume finishes b, then c."""
    from dataclasses import asdict

    cfgs = [tiny_config(n, shards) for n in ("qa", "qb", "qc")]
    train.write_queue([asdict(c) for c in cfgs[1:]])
    first = train.prepare_run(cfgs[0])
    train.set_latest(first)

    real_run = train.Trainer.run

    def run_pausing_qb(self):
        if self.run_dir.name == "qb" and not checkpoints(self.run_dir):
            self.on_step = lambda tr: (tr.run_dir / "PAUSE").touch() if tr.step == 4 else None
        return real_run(self)

    monkeypatch.setattr(train.Trainer, "run", run_pausing_qb)
    assert train.run_in_process(first) == "paused"
    assert read_status(runs / "qa")["state"] == "done"
    assert read_status(runs / "qb")["state"] == "paused"
    assert [q["name"] for q in train.read_queue()] == ["qc"]
    assert Path(os.path.realpath(runs / "latest")).name == "qb"

    (runs / "qb" / "PAUSE").unlink()
    assert train.run_in_process(runs / "qb") == "done"
    assert read_status(runs / "qc")["state"] == "done"
    assert train.read_queue() == []
    assert "queue    :" not in train.fmt_status(runs / "qc")


def test_resume_of_done_run_continues_queue(shards, runs):
    from dataclasses import asdict

    a = write_run(runs, tiny_config("qa", shards))
    train.set_latest(a)
    train.write_status(a, state="done", step=1, tokens=1, total_tokens=1)
    train.write_queue([asdict(tiny_config("qb", shards))])
    assert train.pop_queue() == runs / "qb"
    assert (runs / "qb" / "config.yaml").exists()
    assert train.read_queue() == []


def test_lr_schedules():
    cfg = tiny_config("s", "/nonexistent", total_tokens=1000, warmup_tokens=100, decay_fraction=0.2, lr=1.0)
    assert train.lr_at(cfg, 0) < 0.05
    assert train.lr_at(cfg, 500) == 1.0  # WSD: flat in the middle
    assert train.lr_at(cfg, 900) == pytest.approx(0.5)
    cos = tiny_config("c", "/nonexistent", total_tokens=1000, warmup_tokens=100, lr=1.0, schedule="cosine")
    assert train.lr_at(cos, 550) == pytest.approx(0.5)
    assert train.lr_at(cos, 1000) == pytest.approx(0.0, abs=1e-9)


def test_status_flags_unexpected_stop(shards, runs):
    b = write_run(runs, tiny_config("b", shards))
    train.write_status(b, state="running", step=7, tokens=100, total_tokens=1000)
    assert "stopped unexpectedly" in train.fmt_status(b)
