"""Relevance BiLSTM: model, loss, training, bundle and scorer, with a fake encoder (0024 §3, §5, §6)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.relevance import bilstm
from app.relevance.bilstm import (
    BiLSTMScorer,
    BundleConfig,
    RelevanceBiLSTM,
    RelevanceBundleError,
    cached_embeddings,
    check_bundle,
    distillation_loss,
    embedding_cache_key,
    load_model,
    predict,
    save_bundle,
)
from app.relevance.dataset import Episode
from app.relevance.training import TrainConfig, train_grid, train_one
from app.relevance.windows import Window, Word

DIM = 8


def fake_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic, text-dependent unit vectors."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha1(t.encode()).digest()[:4], "little"))
        v = rng.normal(size=DIM)
        out[i] = v / np.linalg.norm(v)
    return out


def synthetic_episode(upload_id: int, n: int, seed: int) -> Episode:
    """Labels that are a smooth function of the embedding, so a BiLSTM can learn them."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, DIM)).astype(np.float32)
    h = (1 / (1 + np.exp(-emb[:, 0]))).astype(np.float32)
    l = (1 / (1 + np.exp(-emb[:, 1]))).astype(np.float32)  # noqa: E741
    windows = [Window(i, 7.5 * i, 7.5 * i + 15, (Word(7.5 * i, f"w{i}"),)) for i in range(n)]
    return Episode(upload_id, f"ep{upload_id}", "es", windows, h, l, emb)


# ---- model and loss --------------------------------------------------------------


def test_heads_shapes_ranges_and_padding_does_not_leak() -> None:
    torch.manual_seed(0)
    model = RelevanceBiLSTM(emb_dim=DIM, proj=8, hidden=8, head=4).eval()
    a = torch.randn(1, 5, DIM)
    padded = torch.cat([a, torch.randn(1, 3, DIM)], dim=1)
    batch = torch.cat([padded, torch.randn(1, 8, DIM)])
    h, l = model(batch, torch.tensor([5, 8]))  # noqa: E741
    assert h.shape == l.shape == (2, 8)
    assert ((h > 0) & (h < 1)).all() and ((l > 0) & (l < 1)).all()
    h1, l1 = model(a, torch.tensor([5]))
    assert torch.allclose(h[0, :5], h1[0], atol=1e-6) and torch.allclose(l[0, :5], l1[0], atol=1e-6)


def test_smoothness_term_is_zero_on_a_constant_curve() -> None:
    const = torch.full((1, 4), 0.5)
    mask = torch.ones(1, 4)
    assert distillation_loss(const, const, const, const, mask, lam=10.0).item() == pytest.approx(0.0)
    zigzag = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    no_lam = distillation_loss(zigzag, zigzag, zigzag, zigzag, mask, lam=0.0).item()
    with_lam = distillation_loss(zigzag, zigzag, zigzag, zigzag, mask, lam=1.0).item()
    assert no_lam == pytest.approx(0.0) and with_lam == pytest.approx(1.0)  # (0.4+0.6)^2 per step


def test_a_missing_llm_label_drops_out_of_the_llm_term_only() -> None:
    mask = torch.ones(1, 2)
    h_hat = l_hat = torch.tensor([[0.5, 0.5]])
    h = torch.tensor([[0.5, 0.5]])
    l_nan = torch.tensor([[float("nan"), 0.5]])
    assert distillation_loss(h_hat, l_hat, h, l_nan, mask, 0.0).item() == pytest.approx(0.0)
    bad_h = torch.tensor([[0.0, 0.5]])
    assert distillation_loss(h_hat, l_hat, bad_h, l_nan, mask, 0.0).item() > 0


def test_padding_is_masked_out_of_the_loss() -> None:
    h_hat = l_hat = torch.tensor([[0.5, 0.9]])
    target = torch.tensor([[0.5, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert distillation_loss(h_hat, l_hat, target, target, mask, lam=5.0).item() == pytest.approx(0.0)


# ---- training ----------------------------------------------------------------


def test_training_learns_a_learnable_target_and_is_reproducible() -> None:
    train = [synthetic_episode(i, 30, i) for i in range(6)]
    val = [synthetic_episode(100, 30, 100)]
    cfg = TrainConfig(proj=16, hidden=16, head=8, dropout=0.0, lr=1e-2, max_epochs=60, patience=60,
                      lambdas=(0.0,), seeds=(0,))
    run = train_one(train, val, 0.0, 0, cfg)
    assert run.val_spearman > 0.8
    again = train_one(train, val, 0.0, 0, cfg)
    assert again.val_spearman == pytest.approx(run.val_spearman)
    for a, b in zip(run.model.state_dict().values(), again.model.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_grid_picks_the_best_lambda_and_its_median_seed() -> None:
    train = [synthetic_episode(i, 20, i) for i in range(4)]
    val = [synthetic_episode(100, 20, 100)]
    cfg = TrainConfig(proj=8, hidden=8, head=4, max_epochs=3, patience=3, lambdas=(0.0, 0.5), seeds=(0, 1, 2))
    out = train_grid(train, val, cfg)
    assert len(out.runs) == 6
    means = {lam: np.mean([r.val_spearman for r in out.runs if r.lam == lam]) for lam in (0.0, 0.5)}
    assert out.lam == max(means, key=means.get)
    of_lam = sorted(r.val_spearman for r in out.runs if r.lam == out.lam)
    assert out.val_spearman == of_lam[1]
    assert sum(r.model is not None for r in out.runs) == 1
    assert out.summary()["config"]["lambdas"] == [0.0, 0.5]


def test_grid_refuses_episodes_without_embeddings() -> None:
    ep = synthetic_episode(1, 5, 1)
    ep.emb = None
    with pytest.raises(ValueError, match="embeddings"):
        train_grid([ep], [], TrainConfig())


# ---- bundle, cache, scorer -------------------------------------------------------------


def _bundle(tmp_path: Path, label: str = "bilstm-test") -> tuple[Path, RelevanceBiLSTM]:
    torch.manual_seed(1)
    model = RelevanceBiLSTM(emb_dim=DIM, proj=8, hidden=8, head=4).eval()
    enc = tmp_path / "enc-src"
    enc.mkdir(exist_ok=True)
    (enc / "config.json").write_text("{}")
    config = BundleConfig(label=label, encoder_name="fake", encoder_prefix="passage: ", encoder_max_length=128,
                          model_config=model.config, training={"lambda": 0.1})
    save_bundle(tmp_path / label, config, model, {"train": {}}, enc)
    return tmp_path / label, model


def test_bundle_round_trip_gives_identical_outputs(tmp_path: Path) -> None:
    path, model = _bundle(tmp_path)
    config, loaded = load_model(path)
    emb = np.random.default_rng(0).normal(size=(12, DIM)).astype(np.float32)
    for a, b in zip(predict(model, emb), predict(loaded, emb), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert config.label == "bilstm-test" and config.training == {"lambda": 0.1}
    assert (config.combine_heuristic, config.combine_llm) == (0.4, 0.6)
    assert json.loads((path / "config.json").read_text())["encoder"]["dir"] == "encoder"
    check_bundle(path)


def test_missing_or_corrupt_bundles_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(RelevanceBundleError, match="Cannot load"):
        load_model(tmp_path / "nope")
    path, _ = _bundle(tmp_path)
    (path / "weights.pt").write_bytes(b"garbage")
    with pytest.raises(RelevanceBundleError):
        load_model(path)
    other, _ = _bundle(tmp_path, "other")
    cfg = json.loads((other / "config.json").read_text())
    cfg["window"]["block_s"] = 10.0
    (other / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RelevanceBundleError, match="window spec"):
        load_model(other)
    with pytest.raises(RelevanceBundleError, match="missing"):
        bilstm.get_scorer(tmp_path / "absent")


def test_check_bundle_requires_the_encoder(tmp_path: Path) -> None:
    path, _ = _bundle(tmp_path)
    (path / "encoder" / "config.json").unlink()
    with pytest.raises(RelevanceBundleError, match="encoder"):
        check_bundle(path)


def test_embedding_cache_computes_once(tmp_path: Path) -> None:
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return fake_encoder(texts)

    texts = ["hola", "qué tal"]
    key = embedding_cache_key("fake", "p: ", 128, texts)
    a = cached_embeddings(tmp_path, key, texts, enc)
    b = cached_embeddings(tmp_path, key, texts, enc)
    np.testing.assert_array_equal(a, b)
    assert calls == [2]
    assert key != embedding_cache_key("fake", "query: ", 128, texts)
    assert key != embedding_cache_key("fake", "p: ", 128, ["hola", "qué  tal"])


def test_scorer_scores_every_window_with_the_bundle_weights(tmp_path: Path) -> None:
    path, model = _bundle(tmp_path)
    scorer = BiLSTMScorer(path, encoder=fake_encoder)
    words = [Word(float(i), f"palabra{i}") for i in range(60)]
    result = scorer.run(words, 60.0)
    assert [s.window.index for s in result.scores] == list(range(7))  # 8 blocks of 7.5 s -> 7 windows
    h, l = predict(model, fake_encoder([s.window.text for s in result.scores]))  # noqa: E741
    for s, a, b in zip(result.scores, h, l, strict=True):
        assert (s.heuristic, s.llm) == pytest.approx((a, b), abs=1e-6)
        assert s.score == pytest.approx(0.4 * a + 0.6 * b, abs=1e-6)
        assert 0.0 <= s.score <= 1.0


def test_predict_on_an_empty_episode() -> None:
    h, l = predict(RelevanceBiLSTM(emb_dim=DIM), np.zeros((0, DIM), dtype=np.float32))  # noqa: E741
    assert h.shape == l.shape == (0,)


# ---- combine modes (0024 §9, i2) ---------------------------------------------------------


def test_combine_scores_modes() -> None:
    from app.relevance.bilstm import combine_scores

    hh, ll, exact = np.array([0.1, 0.9]), np.array([0.5, 0.5]), np.array([1.0, 0.0])
    h, s = combine_scores("heads", hh, ll, exact)
    np.testing.assert_allclose(h, hh) and np.testing.assert_allclose(s, 0.4 * hh + 0.6 * ll)
    h, s = combine_scores("exact_heuristic", hh, ll, exact)
    np.testing.assert_allclose(h, exact) and np.testing.assert_allclose(s, 0.4 * exact + 0.6 * ll)
    with pytest.raises(ValueError, match="needs"):
        combine_scores("exact_heuristic", hh, ll, None)
    with pytest.raises(ValueError, match="Unknown"):
        combine_scores("nope", hh, ll, exact)


def _exact_bundle(tmp_path: Path, with_lid: bool = True) -> Path:
    import dataclasses

    path, model = _bundle(tmp_path, "exact")
    config, _ = load_model(path)
    (path / "config.json").write_text(json.dumps(dataclasses.replace(config, combine_mode="exact_heuristic").to_json()))
    if with_lid:
        (path / "lid.176.ftz").write_bytes(b"fake")
    return path


def test_exact_mode_round_trips_and_needs_the_lid_model(tmp_path: Path) -> None:
    path = _exact_bundle(tmp_path, with_lid=False)
    assert load_model(path)[0].combine_mode == "exact_heuristic"
    with pytest.raises(RelevanceBundleError, match="lid"):
        check_bundle(path)
    (path / "lid.176.ftz").write_bytes(b"fake")
    check_bundle(path)
    cfg = json.loads((path / "config.json").read_text())
    cfg["combine"]["mode"] = "magic"
    (path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RelevanceBundleError, match="combine mode"):
        load_model(path)


def test_a_pre_mode_bundle_defaults_to_heads(tmp_path: Path) -> None:
    path, _ = _bundle(tmp_path)
    cfg = json.loads((path / "config.json").read_text())
    del cfg["combine"]["mode"]
    (path / "config.json").write_text(json.dumps(cfg))
    assert load_model(path)[0].combine_mode == "heads"


def test_exact_mode_scorer_uses_the_teachers_heuristic_and_the_llm_head(tmp_path: Path) -> None:
    from app.relevance import heuristics
    from app.relevance.bilstm import episode_windows

    path = _exact_bundle(tmp_path)
    scorer = BiLSTMScorer(path, encoder=fake_encoder, lid=lambda text: ("es", 0.99))
    words = [Word(float(i), w) for i, w in enumerate(("hola qué tal eh o sea mira esto es muy importante "
                                                       "porque cambia todo ¿por qué? porque sí claro " * 4).split())]
    result = scorer.run(words, float(len(words)))
    windows = episode_windows(words, float(len(words)))
    exact = heuristics.heuristic_scores(windows, "es")
    _, l = predict(scorer.model, fake_encoder([w.text for w in windows]))  # noqa: E741
    assert result.language == "es"
    for s, e, b in zip(result.scores, exact, l, strict=True):
        assert s.heuristic == pytest.approx(e) and s.llm == pytest.approx(float(b), abs=1e-6)
        assert s.score == pytest.approx(0.4 * e + 0.6 * float(b), abs=1e-6)


def test_validation_can_score_with_the_exact_heuristic() -> None:
    from app.relevance.training import val_spearman

    ep = synthetic_episode(1, 30, 1)
    model = RelevanceBiLSTM(emb_dim=DIM, proj=8, hidden=8, head=4)
    # The target is 0.4*h + 0.6*l; with the exact h, half the score is already right.
    assert val_spearman(model, [ep], "cpu", "exact_heuristic") > val_spearman(model, [ep], "cpu", "heads") - 1e-9
