"""Unit tests for model pipeline steps (model, checkpoint)."""
import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from omegaconf import OmegaConf

from trader_keras.pipeline import Ctx, pipe
from trader_keras.steps.model import checkpoint, model


def _make_ctx(**overrides) -> Ctx:
    """Build a minimal Ctx with small dummy data and config."""
    cfg = OmegaConf.create({
        "backbone": {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "data": {"lookback": 10},
        "train": {
            "horizons": [1, 5],
            "loss": "gaussian_nll",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "clip_grad_norm": 1.0,
        },
    })
    n_samples, lookback, n_features = 8, 10, 6
    ctx: Ctx = {
        "cfg": cfg,
        "x_train": np.random.randn(n_samples, lookback, n_features).astype("float32"),
    }
    ctx.update(overrides)
    return ctx


# ── model step ──────────────────────────────────────────────────────


def test_model_step_returns_ctx_with_model():
    ctx = model(_make_ctx())
    assert "model" in ctx


def test_model_step_output_shape_gaussian():
    ctx = model(_make_ctx())
    m = ctx["model"]
    x = np.random.randn(4, 10, 6).astype("float32")
    y = m(x, training=False)
    # 2 horizons, gaussian_nll → n_out=2
    assert y.shape == (4, 2, 2), f"Expected (4,2,2), got {y.shape}"


def test_model_step_output_shape_mse():
    ctx = _make_ctx()
    ctx["cfg"].train.loss = "mse"
    ctx = model(ctx)
    m = ctx["model"]
    x = np.random.randn(4, 10, 6).astype("float32")
    y = m(x, training=False)
    assert y.shape == (4, 2, 1), f"Expected (4,2,1), got {y.shape}"


def test_model_step_is_compiled():
    """Model should be compiled (has optimizer after compile)."""
    ctx = model(_make_ctx())
    m = ctx["model"]
    assert m.optimizer is not None


def test_model_step_optimizer_lr():
    ctx = model(_make_ctx())
    m = ctx["model"]
    lr = float(m.optimizer.learning_rate)
    assert abs(lr - 1e-3) < 1e-7


def test_model_step_can_train_one_batch():
    """Smoke test: one gradient step should produce finite loss."""
    ctx = model(_make_ctx())
    m = ctx["model"]
    x = np.random.randn(4, 10, 6).astype("float32")
    y = np.random.randn(4, 2).astype("float32")
    history = m.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert all(np.isfinite(l) for l in history.history["loss"])


# ── checkpoint step ─────────────────────────────────────────────────


def test_checkpoint_noop_when_no_config():
    """checkpoint is a no-op when backbone has no checkpoint field."""
    ctx = model(_make_ctx())
    weights_before = ctx["model"].get_weights()
    ctx = checkpoint(ctx)
    weights_after = ctx["model"].get_weights()
    for wb, wa in zip(weights_before, weights_after):
        np.testing.assert_array_equal(wb, wa)


def test_checkpoint_noop_when_empty_string():
    ctx = _make_ctx()
    ctx["cfg"].backbone.checkpoint = ""
    ctx = model(ctx)
    ctx = checkpoint(ctx)
    assert ctx["model"].trainable is True


def test_checkpoint_noop_when_null():
    ctx = _make_ctx()
    ctx["cfg"].backbone.checkpoint = None
    ctx = model(ctx)
    ctx = checkpoint(ctx)
    assert ctx["model"].trainable is True


def test_checkpoint_freeze_sets_trainable_false(tmp_path):
    """When freeze=True and a valid checkpoint, model.trainable → False."""
    ctx = model(_make_ctx())
    ckpt = str(tmp_path / "weights.weights.h5")
    ctx["model"].save_weights(ckpt)
    ctx["cfg"].backbone.checkpoint = ckpt
    ctx["cfg"].backbone.freeze = True
    ctx = checkpoint(ctx)
    assert ctx["model"].trainable is False


def test_checkpoint_loads_weights(tmp_path):
    """Weights should change after loading a different checkpoint."""
    ctx = model(_make_ctx())
    m = ctx["model"]
    ckpt = str(tmp_path / "weights.weights.h5")
    m.save_weights(ckpt)

    ctx2 = model(_make_ctx())
    m2 = ctx2["model"]
    differs = any(
        not np.allclose(w1, w2)
        for w1, w2 in zip(m.get_weights(), m2.get_weights())
    )
    assert differs, "Fresh models should have different random weights"

    ctx2["cfg"].backbone.checkpoint = ckpt
    ctx2 = checkpoint(ctx2)
    for w1, w2 in zip(m.get_weights(), ctx2["model"].get_weights()):
        np.testing.assert_allclose(w1, w2, rtol=1e-5)


# ── pipe integration ────────────────────────────────────────────────


def test_pipe_model_then_checkpoint():
    """model → checkpoint should work as a pipeline."""
    ctx = pipe(_make_ctx(), model, checkpoint)
    assert "model" in ctx
    assert ctx["model"].trainable is True
