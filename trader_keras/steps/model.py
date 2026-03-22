"""Pipeline steps for model building and checkpoint loading."""
from __future__ import annotations

import logging

import keras
from omegaconf import OmegaConf

from ..models.gru import LOSSES, build_gru_model
from ..pipeline import Ctx, step

logger = logging.getLogger(__name__)


@step
def model(ctx: Ctx) -> Ctx:
    """Build and compile a GRU model from ctx config and training data shape."""
    cfg = ctx["cfg"]
    bb = cfg.backbone
    tr = cfg.train

    n_features = ctx["x_train"].shape[2]
    n_horizons = len(tr.horizons)

    m = build_gru_model(
        n_features=n_features, n_horizons=n_horizons, cfg=bb,
        loss_name=tr.loss, lookback=cfg.data.lookback,
    )
    m.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=tr.lr,
            weight_decay=tr.weight_decay,
            clipnorm=tr.clip_grad_norm,
        ),
        loss=LOSSES[tr.loss],
    )
    logger.info("Compiled GRU model: %d params", m.count_params())

    ctx["model"] = m
    return ctx


@step
def checkpoint(ctx: Ctx) -> Ctx:
    """Load weights from checkpoint and optionally freeze the model.

    No-op when backbone has no 'checkpoint' field or it's empty/null.
    """
    cfg = ctx["cfg"]
    bb = cfg.backbone

    ckpt_path = OmegaConf.select(bb, "checkpoint", default=None)
    if not ckpt_path:
        return ctx

    m = ctx["model"]
    m.load_weights(ckpt_path)
    logger.info("Loaded weights from %s", ckpt_path)

    freeze = OmegaConf.select(bb, "freeze", default=False)
    if freeze:
        m.trainable = False
        logger.info("Model frozen (trainable=False)")

    return ctx
