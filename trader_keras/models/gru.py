"""GRU predictor — multi-horizon forecaster in Keras 3.

Architecture:
    Input: (batch, lookback, n_features)
    → GRU(hidden_size, num_layers, dropout)
    → per-horizon Dense heads
    Output: (batch, n_horizons, n_out)
"""
from __future__ import annotations

import keras
from keras import layers, ops


LOSSES: dict[str, callable] = {}


def _register_loss(fn):
    LOSSES[fn.__name__] = fn
    return fn


def build_gru_model(n_features: int, n_horizons: int, cfg) -> keras.Model:
    """Build GRU stack → per-horizon output heads."""
    n_out = LOSSES[cfg.loss].n_out
    inputs = keras.Input(shape=(cfg.lookback, n_features), name="sequence")

    x = inputs
    for i in range(cfg.num_layers):
        x = layers.GRU(cfg.hidden_size, return_sequences=(i < cfg.num_layers - 1),
                       dropout=cfg.dropout, recurrent_dropout=0.0, name=f"gru_{i}")(x)

    heads = [layers.Dense(n_out, name=f"head_h{i}")(x) for i in range(n_horizons)]
    # Stack heads: (batch, n_horizons, n_out) — works for any n_horizons
    stacked = layers.Concatenate(axis=-1)(heads) if len(heads) > 1 else heads[0]
    outputs = layers.Reshape((n_horizons, n_out))(stacked)

    return keras.Model(inputs=inputs, outputs=outputs, name="gru_forecaster")


@_register_loss
def gaussian_nll(y_true, y_pred):
    """Gaussian NLL: penalizes both wrong mu and wrong sigma."""
    mu, log_sigma = y_pred[..., 0], y_pred[..., 1]
    sigma = ops.softplus(log_sigma) + 1e-6
    return ops.mean(ops.square(y_true - mu) / ops.square(sigma) + 2.0 * ops.log(sigma))

gaussian_nll.n_out = 2


@_register_loss
def mse(y_true, y_pred):
    return ops.mean(ops.square(y_true - y_pred[..., 0]))

mse.n_out = 1
