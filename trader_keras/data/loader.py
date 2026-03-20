"""Data loading: parquet → OHLCV bars → features → numpy sequences."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig, Stage1Config
from ..constants import DATA_DIR
from .features import create_features, create_targets, select_features
from .resampler import parse_bar_seconds_from_name, reaggregate_bars, resample_to_bars

logger = logging.getLogger(__name__)

# Columns expected in raw trade data parquets
_TRADE_COLS = {"timestamp", "price", "amount", "side"}
# Columns expected in pre-aggregated bar parquets
_BAR_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


def _load_parquet(path: str, load_limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if load_limit:
        df = df.iloc[:load_limit]
    return df


def _is_bar_data(df: pd.DataFrame) -> bool:
    return _BAR_COLS.issubset(df.columns) and not _TRADE_COLS.issubset(df.columns)


def load_bars(path: str, bar_seconds: int, load_limit: int | None = None) -> pd.DataFrame:
    """Load a single parquet file → OHLCV bars at bar_seconds resolution."""
    df = _load_parquet(path, load_limit)
    filename = Path(path).name

    if _is_bar_data(df):
        source_sec = parse_bar_seconds_from_name(filename) or bar_seconds
        if source_sec != bar_seconds:
            df = reaggregate_bars(df, source_sec, bar_seconds)
        return df
    else:
        return resample_to_bars(df, bar_seconds)


def load_dataset(
    data_cfg: DataConfig,
    stage1_cfg: Stage1Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Full pipeline: parquet → bars → features → targets → train/val splits.

    Returns:
        x_train, y_train, x_val, y_val, feature_cols
        Shapes:
          x_*: (N, lookback, n_features)
          y_*: (N, n_horizons, 1) — or (N, n_horizons, 2) if probabilistic (targets only, no sigma)
    """
    data_dir = Path(data_cfg.data_dir).expanduser()
    pattern = data_cfg.pattern
    paths = sorted(glob.glob(str(data_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    all_x, all_y = [], []
    feature_cols: list[str] | None = None

    for path in paths:
        logger.info("Loading %s", path)
        bars = load_bars(path, stage1_cfg.bar_seconds, data_cfg.load_limit)
        df = create_features(bars, lookback=stage1_cfg.lookback)
        df, target_cols = create_targets(df, stage1_cfg.horizons)

        if feature_cols is None:
            feature_cols = select_features(df, stage1_cfg.features)

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
    y_data = df[target_cols].values.astype(np.float32)  # (T, n_horizons)

    n = len(x_data)
    lookback = cfg.lookback
    stride = cfg.stride

    starts = range(0, n - lookback + 1, stride)
    x_seqs = np.stack([x_data[i : i + lookback] for i in starts])  # (N, L, F)
    y_seqs = np.stack([y_data[i + lookback - 1] for i in starts])  # (N, H)

    return x_seqs, y_seqs
