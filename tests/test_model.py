"""Unit tests for loss functions."""
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

from trader_keras.models.gru import LOSSES


def test_gaussian_nll_loss_finite():
    y_true = np.random.randn(8, 2).astype("float32")
    y_pred = np.random.randn(8, 2, 2).astype("float32")
    loss = LOSSES["gaussian_nll"](y_true, y_pred)
    val = float(keras.ops.convert_to_numpy(loss))
    assert np.isfinite(val), f"NLL loss is not finite: {val}"


def test_mse_loss_finite():
    y_true = np.random.randn(8, 2).astype("float32")
    y_pred = np.random.randn(8, 2, 1).astype("float32")
    loss = LOSSES["mse"](y_true, y_pred)
    val = float(keras.ops.convert_to_numpy(loss))
    assert np.isfinite(val), f"MSE loss is not finite: {val}"
