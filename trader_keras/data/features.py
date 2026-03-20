"""Feature engineering — 6 log-ratio features, matching trader-t5x."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import FEATURE_COLS


def create_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute log-ratio features with NO future leakage (all use prev bar)."""
    df = bars.copy()
    eps = 1e-8
    prev = df[["open", "high", "low", "close", "volume"]].shift(1)
    df["log_open"] = np.log(df["open"] / (prev["open"] + eps) + eps)
    df["log_high"] = np.log(df["high"] / (prev["high"] + eps) + eps)
    df["log_low"] = np.log(df["low"] / (prev["low"] + eps) + eps)
    df["log_close"] = np.log(df["close"] / (prev["close"] + eps) + eps)
    df["log_volume"] = np.log(df["volume"] / (prev["volume"] + eps) + eps).clip(-10, 10)
    df["log_spread"] = (df.get("spread", 0.0) / (df["close"] + eps)).clip(0, 0.1)
    return df.dropna(subset=FEATURE_COLS)


def create_targets(
    df: pd.DataFrame, horizons: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Create log-return targets for each horizon."""
    target_cols = []
    for h in horizons:
        col = f"target_h{h}"
        df[col] = np.log(df["close"].shift(-h) / df["close"])
        target_cols.append(col)
    return df.dropna(subset=target_cols), target_cols
