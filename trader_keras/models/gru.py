"""GRU predictor in Keras 3 — probabilistic multi-horizon forecaster.

Architecture:
    Input: (batch, lookback, n_features)
    → GRU(hidden_size, num_layers, dropout)
    → final hidden state: (batch, hidden_size)
    → per-horizon heads: Linear(hidden_size, 2) → (mu, log_sigma)

Loss: Gaussian NLL = (r - mu)^2 / sigma^2 + log(sigma^2)
      summed across all horizons.
"""
from __future__ import annotations

import keras
from keras import layers, ops

from ..config import Stage1Config


def _gru_stack(hidden_size: int, num_layers: int, dropout: float) -> list[layers.Layer]:
    """Build a stack of GRU layers, returning final hidden state."""
    gru_layers = []
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        gru_layers.append(
            layers.GRU(
                hidden_size,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=0.0,
                name=f"gru_{i}",
            )
        )
    return gru_layers


def build_gru_model(
    n_features: int,
    n_horizons: int,
    cfg: Stage1Config,
) -> keras.Model:
    """Build and return the GRU forecasting model.

    Outputs shape: (batch, n_horizons, 2) — (mu, log_sigma) per horizon.
    In non-probabilistic mode outputs (batch, n_horizons, 1) — just mu.
    """
    inputs = keras.Input(shape=(cfg.lookback, n_features), name="sequence")

    x = inputs
    for gru_layer in _gru_stack(cfg.hidden_size, cfg.num_layers, cfg.dropout):
        x = gru_layer(x)  # final layer returns (batch, hidden_size)

    # Per-horizon output heads
    head_outputs = []
    n_out = 2 if cfg.probabilistic else 1
    for h_idx in range(n_horizons):
        out = layers.Dense(n_out, name=f"head_h{h_idx}")(x)
        head_outputs.append(out)  # each: (batch, n_out)

    # Stack along horizons: (batch, n_horizons, n_out)
    if n_horizons == 1:
        outputs = layers.Reshape((1, n_out))(head_outputs[0])
    else:
        # Concatenate then reshape: each head_out is (batch, n_out)
        # → concat to (batch, n_horizons * n_out) → reshape to (batch, n_horizons, n_out)
        outputs = layers.Concatenate(axis=-1)(head_outputs)  # (batch, n_horizons * n_out)
        outputs = layers.Reshape((n_horizons, n_out))(outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name="gru_forecaster")


def gaussian_nll_loss(
    y_true: object,
    y_pred: object,
    magnitude_alpha: float = 0.0,
) -> object:
    """Gaussian NLL loss for probabilistic forecasting.

    Args:
        y_true: (batch, n_horizons) — actual log returns
        y_pred: (batch, n_horizons, 2) — (mu, log_sigma) per horizon
        magnitude_alpha: weight exponent; 0 = uniform weighting

    Returns:
        Scalar mean loss.
    """
    mu = y_pred[..., 0]         # (batch, n_horizons)
    log_sigma = y_pred[..., 1]  # (batch, n_horizons)
    sigma = ops.softplus(log_sigma) + 1e-6  # ensure sigma > 0

    # NLL = (r - mu)^2 / sigma^2 + log(sigma^2)
    nll = ops.square(y_true - mu) / ops.square(sigma) + 2.0 * ops.log(sigma)

    if magnitude_alpha > 0.0:
        weights = ops.power(ops.abs(y_true) + 1e-8, magnitude_alpha)
        nll = nll * weights

    return ops.mean(nll)


def mse_loss(y_true: object, y_pred: object) -> object:
    """MSE loss for point-estimate forecasting.

    Args:
        y_true: (batch, n_horizons)
        y_pred: (batch, n_horizons, 1)
    """
    return ops.mean(ops.square(y_true - y_pred[..., 0]))
