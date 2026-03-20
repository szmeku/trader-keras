"""Unit tests for Keras model and loss functions."""
import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest

from trader_keras.config import Stage1Config
from trader_keras.models.gru import build_gru_model, gaussian_nll_loss, mse_loss


def _cfg(**kw) -> Stage1Config:
    defaults = dict(hidden_size=16, num_layers=1, lookback=10, horizons=[1, 5], dropout=0.0)
    return Stage1Config(**{**defaults, **kw})


def test_model_output_shape_probabilistic():
    cfg = _cfg(probabilistic=True)
    model = build_gru_model(n_features=8, n_horizons=2, cfg=cfg)
    x = np.random.randn(4, 10, 8).astype("float32")
    y = model(x, training=False)
    assert y.shape == (4, 2, 2), f"Expected (4,2,2), got {y.shape}"


def test_model_output_shape_mse():
    cfg = _cfg(probabilistic=False)
    model = build_gru_model(n_features=8, n_horizons=2, cfg=cfg)
    x = np.random.randn(4, 10, 8).astype("float32")
    y = model(x, training=False)
    assert y.shape == (4, 2, 1), f"Expected (4,2,1), got {y.shape}"


def test_gaussian_nll_loss_finite():
    import keras
    y_true = np.random.randn(8, 2).astype("float32")
    y_pred = np.random.randn(8, 2, 2).astype("float32")
    loss = gaussian_nll_loss(y_true, y_pred)
    val = float(keras.ops.convert_to_numpy(loss))
    assert np.isfinite(val), f"NLL loss is not finite: {val}"


def test_mse_loss_finite():
    import keras
    y_true = np.random.randn(8, 2).astype("float32")
    y_pred = np.random.randn(8, 2, 1).astype("float32")
    loss = mse_loss(y_true, y_pred)
    val = float(keras.ops.convert_to_numpy(loss))
    assert np.isfinite(val), f"MSE loss is not finite: {val}"


def test_model_compile_and_step():
    """Quick gradient step to verify training works end-to-end."""
    import keras

    cfg = _cfg(probabilistic=True, lr=1e-3)
    model = build_gru_model(n_features=8, n_horizons=2, cfg=cfg)
    from trader_keras.models.gru import gaussian_nll_loss

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
        loss=gaussian_nll_loss,
    )
    x = np.random.randn(16, 10, 8).astype("float32")
    y = np.random.randn(16, 2).astype("float32")
    history = model.fit(x, y, epochs=2, batch_size=8, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses), f"Training produced non-finite losses: {losses}"
