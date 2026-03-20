"""Stage 1 training loop using Keras 3 + JAX backend."""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, Stage1Config
from .data.loader import load_dataset
from .logger.base import BaseLogger
from .logger.factory import create_logger
from .models.gru import build_gru_model, gaussian_nll_loss, mse_loss

logger = logging.getLogger(__name__)

_ARTIFACTS = Path("artifacts")


class _MetricsCallback:
    """Keras-compatible callback that forwards epoch metrics to our BaseLogger."""

    def __init__(self, base_logger: BaseLogger, val_exists: bool) -> None:
        self._log = base_logger
        self._val_exists = val_exists
        self._step = 0
        self._t0 = time.time()

    # Called by our manual loop
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        metrics: dict[str, Any] = {"epoch": epoch}
        train_loss = logs.get("loss", float("nan"))
        metrics["stage1/train_loss"] = train_loss

        if self._val_exists and "val_loss" in logs:
            val_loss = logs["val_loss"]
            metrics["stage1/val_loss"] = val_loss
            if train_loss > 0:
                metrics["stage1/overfit_ratio"] = val_loss / train_loss

        elapsed = time.time() - self._t0
        metrics["stage1/elapsed_s"] = elapsed
        self._log.log(metrics, step=epoch)
        self._step = epoch + 1


def _build_loss_fn(cfg: Stage1Config):
    if cfg.probabilistic:
        def loss_fn(y_true, y_pred):
            return gaussian_nll_loss(y_true, y_pred, cfg.magnitude_alpha)
    else:
        loss_fn = mse_loss
    return loss_fn


def _config_to_dict(cfg: Config) -> dict[str, Any]:
    return {
        "stage1": dataclasses.asdict(cfg.stage1),
        "data": dataclasses.asdict(cfg.data),
        "logging": dataclasses.asdict(cfg.logging),
    }


def train(cfg: Config) -> Path:
    """Full Stage 1 training run.

    Returns:
        Path to saved model artifact.
    """
    import keras  # set backend before importing keras
    import keras.callbacks as kc

    # Set JAX as backend (must be done before first keras import per run)
    os.environ.setdefault("KERAS_BACKEND", "jax")

    s1 = cfg.stage1
    if s1.seed is not None:
        keras.utils.set_random_seed(s1.seed)

    run_config = _config_to_dict(cfg)
    log = create_logger(cfg.logging, run_config)

    logger.info("Loading data…")
    x_train, y_train, x_val, y_val, feature_cols = load_dataset(cfg.data, s1)
    n_features = x_train.shape[2]
    n_horizons = len(s1.horizons)
    val_exists = len(x_val) > 0

    logger.info(
        "Data: train=%d  val=%d  features=%d  horizons=%d",
        len(x_train), len(x_val), n_features, n_horizons,
    )

    model = build_gru_model(n_features, n_horizons, s1)
    model.summary(print_fn=logger.info)

    loss_fn = _build_loss_fn(s1)

    # Keras 3 optimizer (backend-agnostic)
    optimizer = keras.optimizers.AdamW(
        learning_rate=s1.lr,
        weight_decay=s1.weight_decay,
        clipnorm=s1.clip_grad_norm,
    )

    model.compile(optimizer=optimizer, loss=loss_fn)

    # Callbacks
    callbacks: list = [
        kc.EarlyStopping(
            monitor="val_loss" if val_exists else "loss",
            patience=s1.patience,
            restore_best_weights=True,
            verbose=0,
        ),
        kc.ReduceLROnPlateau(
            monitor="val_loss" if val_exists else "loss",
            factor=0.5,
            patience=max(3, s1.patience // 3),
            min_lr=s1.lr * 1e-3,
            verbose=0,
        ),
    ]

    val_data = (x_val, y_val) if val_exists else None

    metrics_cb = _MetricsCallback(log, val_exists)

    class _ForwardCallback(kc.Callback):
        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            metrics_cb.on_epoch_end(epoch, logs or {})

    callbacks.append(_ForwardCallback())

    _ARTIFACTS.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = _ARTIFACTS / f"gru_predictor_{ts}.keras"

    callbacks.append(
        kc.ModelCheckpoint(
            str(model_path),
            monitor="val_loss" if val_exists else "loss",
            save_best_only=True,
            verbose=0,
        )
    )

    logger.info("Training…")
    model.fit(
        x_train,
        y_train,
        epochs=s1.epochs,
        batch_size=s1.batch_size,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,  # time-series: never shuffle
    )

    log.finish()
    logger.info("Model saved to %s", model_path)
    return model_path
