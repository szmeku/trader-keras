"""Stage 1 training loop using Keras 3 + JAX backend."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf

from .data.loader import load_dataset
from .models.gru import LOSSES, build_gru_model

logger = logging.getLogger(__name__)

_ARTIFACTS = Path("artifacts")


def train(cfg: DictConfig) -> Path:
    """Full Stage 1 training run. Returns path to saved model."""
    import keras
    import keras.callbacks as kc

    os.environ.setdefault("KERAS_BACKEND", "jax")

    s1 = cfg.stage1
    keras.utils.set_random_seed(s1.seed)

    # W&B — always init; disable via WANDB_MODE=disabled
    api_key = os.environ.get("WANDB_API_TOKEN") or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=False)
    wandb.init(
        project=cfg.wandb.project, tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
    )

    logger.info("Loading data...")
    x_train, y_train, x_val, y_val, feature_cols = load_dataset(cfg.data, s1)
    n_features = x_train.shape[2]
    n_horizons = len(s1.horizons)

    logger.info(
        "Data: train=%d  val=%d  features=%d  horizons=%d",
        len(x_train), len(x_val), n_features, n_horizons,
    )

    model = build_gru_model(n_features, n_horizons, s1)
    model.summary(print_fn=logger.info)

    loss_fn = LOSSES[s1.loss]
    optimizer = keras.optimizers.AdamW(
        learning_rate=s1.lr, weight_decay=s1.weight_decay, clipnorm=s1.clip_grad_norm,
    )
    model.compile(optimizer=optimizer, loss=loss_fn)

    val_data = (x_val, y_val) if len(x_val) else None
    monitor = "val_loss" if val_data else "loss"

    _ARTIFACTS.mkdir(exist_ok=True)
    model_path = _ARTIFACTS / f"gru_predictor_{time.strftime('%Y%m%d_%H%M%S')}.keras"

    class _EpochLogger(kc.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._t0 = time.time()
            self._epoch_t0 = 0.0

        def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
            self._epoch_t0 = time.time()

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            logs = logs or {}
            epoch_time = time.time() - self._epoch_t0
            sps = len(x_train) / epoch_time if epoch_time > 0 else 0
            metrics = {
                "epoch": epoch,
                "stage1/train_loss": logs.get("loss", float("nan")),
                "stage1/val_loss": logs.get("val_loss", float("nan")),
                "stage1/samples_per_sec": sps,
                "stage1/elapsed_s": time.time() - self._t0,
            }
            logger.info(
                "Epoch %d — loss=%.6f  val_loss=%.6f  samples/sec=%.0f",
                epoch, metrics["stage1/train_loss"],
                metrics["stage1/val_loss"], sps,
            )
            wandb.log(metrics, step=epoch)

    callbacks = [
        kc.EarlyStopping(monitor=monitor, patience=s1.patience, restore_best_weights=True, verbose=0),
        kc.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(3, s1.patience // 3),
                             min_lr=s1.lr * 1e-3, verbose=0),
        kc.ModelCheckpoint(str(model_path), monitor=monitor, save_best_only=True, verbose=0),
        _EpochLogger(),
    ]

    logger.info("Training...")
    model.fit(x_train, y_train, epochs=s1.epochs, batch_size=s1.batch_size,
              validation_data=val_data, callbacks=callbacks, verbose=0, shuffle=False)

    wandb.finish()
    logger.info("Model saved to %s", model_path)
    return model_path
