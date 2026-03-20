"""Data loading: parquet → features → numpy sequences."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig, Stage1Config
from .features import create_features, create_targets

logger = logging.getLogger(__name__)


def _load_parquet(path: str, load_limit: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.iloc[:load_limit or None]


def load_dataset(
    data_cfg: DataConfig,
    stage1_cfg: Stage1Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Full pipeline: parquet → features → targets → train/val splits."""
    data_dir = Path(data_cfg.data_dir).expanduser()
    paths = sorted(glob.glob(str(data_dir / data_cfg.pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{data_cfg.pattern}' in {data_dir}")

    all_x, all_y = [], []
    feature_cols: list[str] | None = None

    for path in paths:
        logger.info("Loading %s", path)
        bars = _load_parquet(path, data_cfg.load_limit)
        df = create_features(bars)
        df, target_cols = create_targets(df, stage1_cfg.horizons)

        if feature_cols is None:
            from ..constants import FEATURE_COLS
            feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        x_arr, y_arr = _make_sequences(df, feature_cols, target_cols, stage1_cfg)
        all_x.append(x_arr)
        all_y.append(y_arr)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    split = int(len(x) * stage1_cfg.train_ratio)
    return x[:split], y[:split], x[split:], y[split:], feature_cols or []


def _make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    cfg: Stage1Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of `lookback` over the feature array."""
    x_data = df[feature_cols].values.astype(np.float32)
    y_data = df[target_cols].values.astype(np.float32)

    n = len(x_data)
    starts = range(0, n - cfg.lookback + 1, cfg.stride)
    x_seqs = np.stack([x_data[i : i + cfg.lookback] for i in starts])
    y_seqs = np.stack([y_data[i + cfg.lookback - 1] for i in starts])
    return x_seqs, y_seqs
