"""Training pipeline steps: fit_supervised, save."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import keras
import keras.callbacks as kc
import wandb
from omegaconf import OmegaConf

from ..pipeline import Ctx, step

logger = logging.getLogger(__name__)

_ARTIFACTS = Path("artifacts")


class _WandbLogger(kc.Callback):
    """Logs per-epoch metrics to wandb (keras verbose=2 handles console)."""

    def __init__(self, n_train: int) -> None:
        super().__init__()
        self._n_train = n_train
        self._t0 = time.time()
        self._epoch_t0 = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._epoch_t0 = time.time()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        dt = time.time() - self._epoch_t0
        wandb.log({
            "epoch": epoch,
            "train_loss": (logs or {}).get("loss"),
            "val_loss": (logs or {}).get("val_loss"),
            "samples_per_sec": self._n_train / dt if dt > 0 else 0,
            "elapsed_s": time.time() - self._t0,
        }, step=epoch)


@step
def fit_supervised(ctx: Ctx) -> Ctx:
    """Train model with keras model.fit(), EarlyStopping, checkpointing."""
    cfg = ctx["cfg"]
    tr = cfg.train
    model = ctx["model"]
    x_train, y_train = ctx["x_train"], ctx["y_train"]
    x_val, y_val = ctx["x_val"], ctx["y_val"]

    wandb.init(
        project=cfg.wandb.project, tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
    )

    n_params = model.count_params()
    n_train = len(x_train)
    n_features = x_train.shape[-1]
    memo_ratio = n_train / n_params if n_params > 0 else float("inf")
    info_ratio = (n_train * n_features) / n_params if n_params > 0 else float("inf")
    logger.info(
        "samples=%d, features=%d, params=%d | memo_ratio=%.2f, info_ratio=%.2f",
        n_train, n_features, n_params, memo_ratio, info_ratio,
    )
    wandb.summary["n_params"] = n_params
    wandb.summary["n_train"] = n_train
    wandb.summary["memo_ratio"] = memo_ratio
    wandb.summary["info_ratio"] = info_ratio

    val_data = (x_val, y_val) if len(x_val) else None
    monitor = "val_loss" if val_data else "loss"

    _ARTIFACTS.mkdir(exist_ok=True)
    model_path = _ARTIFACTS / f"gru_predictor_{time.strftime('%Y%m%d_%H%M%S')}.keras"

    callbacks = [
        kc.EarlyStopping(
            monitor=monitor, patience=tr.patience,
            restore_best_weights=True, verbose=0,
        ),
        kc.ReduceLROnPlateau(
            monitor=monitor, factor=0.5,
            patience=max(3, tr.patience // 3),
            min_lr=tr.lr * 1e-3, verbose=0,
        ),
        kc.ModelCheckpoint(
            str(model_path), monitor=monitor,
            save_best_only=True, verbose=0,
        ),
        _WandbLogger(n_train=len(x_train)),
    ]

    model.fit(
        x_train, y_train, epochs=tr.epochs, batch_size=tr.batch_size,
        validation_data=val_data, callbacks=callbacks, verbose=2, shuffle=True,
    )

    ctx["model"] = model
    ctx["model_path"] = model_path
    return ctx


@step
def save(ctx: Ctx) -> Ctx:
    """Finalize training run: call wandb.finish()."""
    model_path = ctx["model_path"]
    wandb.finish()
    logger.info("Training complete. Model at %s", model_path)
    return ctx
