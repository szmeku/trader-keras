"""Data pipeline steps: load, featurize, window."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.features import FEATURE_COLS
from ..data.features import create_features, create_targets
from ..pipeline import Ctx, step

logger = logging.getLogger(__name__)


@step
def load(ctx: Ctx) -> Ctx:
    """Load parquet files matching pattern, concat into ctx['bars']."""
    cfg = ctx["cfg"]
    data_dir = Path(cfg.data.data_dir).expanduser()
    paths = sorted(glob.glob(str(data_dir / cfg.data.pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{cfg.data.pattern}' in {data_dir}"
        )

    frames = [_load_parquet(p, cfg.data.load_limit) for p in paths]
    ctx["bars"] = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d bars from %d files", len(ctx["bars"]), len(paths))
    return ctx


@step
def featurize(ctx: Ctx) -> Ctx:
    """Compute log-ratio features, drop NaN rows."""
    ctx["bars"] = create_features(ctx["bars"])
    logger.info("Featurized: %d rows, cols %s", len(ctx["bars"]), FEATURE_COLS)
    return ctx


@step
def window(ctx: Ctx) -> Ctx:
    """Create targets, sliding windows, train/val split."""
    cfg = ctx["cfg"]
    df, target_cols = create_targets(ctx["bars"], cfg.train.horizons)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    x_seqs, y_seqs = _make_sequences(
        df, feature_cols, target_cols,
        lookback=cfg.data.lookback, stride=cfg.data.stride,
    )

    split = int(len(x_seqs) * cfg.train.train_ratio)
    ctx["x_train"] = x_seqs[:split]
    ctx["y_train"] = y_seqs[:split]
    ctx["x_val"] = x_seqs[split:]
    ctx["y_val"] = y_seqs[split:]
    ctx["feature_cols"] = feature_cols
    logger.info("Windows: train=%d, val=%d", split, len(x_seqs) - split)
    return ctx


def _load_parquet(path: str, load_limit: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.iloc[:load_limit or None]


def _make_sequences(
    df: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
    lookback: int, stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of `lookback` over the feature array."""
    x_data = df[feature_cols].values.astype(np.float32)
    y_data = df[target_cols].values.astype(np.float32)
    n = len(x_data)
    starts = range(0, n - lookback + 1, stride)
    x_seqs = np.stack([x_data[i : i + lookback] for i in starts])
    y_seqs = np.stack([y_data[i + lookback - 1] for i in starts])
    return x_seqs, y_seqs
