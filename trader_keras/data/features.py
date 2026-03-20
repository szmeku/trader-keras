"""Feature engineering — pure pandas/numpy, no Numba dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import FEATURE_COLS, OPTIONAL_FEATURE_COLS


def create_features(bars: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute technical features with NO future leakage (all shift(1) applied).

    Args:
        bars: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume, buy_ratio
        lookback: rolling window size for volatility / volume_ma

    Returns:
        DataFrame with feature columns appended and NaN rows dropped.
    """
    df = bars.copy()
    closes = df["close"]

    # Returns — shift(1) applied after to prevent leakage
    df["returns"] = closes.pct_change()
    df["log_returns"] = np.log(closes / closes.shift(1))

    # Rolling stats (computed on non-shifted data, then shifted)
    df["volatility"] = df["returns"].rolling(lookback, min_periods=2).std().shift(1)
    df["volume_ma"] = df["volume"].rolling(lookback, min_periods=1).mean().shift(1)
    df["volume_ratio"] = np.where(
        df["volume_ma"] > 0, df["volume"] / df["volume_ma"], 0.0
    )

    df["log_volume"] = np.log1p(df["volume"])
    df["log_volatility"] = np.log1p(df["volatility"])
    df["log_volume_ma"] = np.log1p(df["volume_ma"])

    # Vol regime: position within rolling min-max of volatility
    vol = df["volatility"]
    vol_min = vol.rolling(lookback, min_periods=10).min().shift(1)
    vol_max = vol.rolling(lookback, min_periods=10).max().shift(1)
    df["vol_regime"] = (vol - vol_min) / (vol_max - vol_min + 1e-10)

    # Price acceleration: second derivative of log returns
    df["price_accel"] = df["log_returns"].diff().shift(1)

    # Time of day in seconds (0–86399)
    ts = df["timestamp"]
    df["time_in_day_counter"] = (
        ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    ).astype(float)

    # Optional features (present only if source columns exist)
    if "spread" in df.columns:
        df["log_spread"] = np.log1p(df["spread"])

    return df.dropna()


def select_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> list[str]:
    """Return the active feature column list, adding optionals that are present."""
    base = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    present_optionals = [c for c in OPTIONAL_FEATURE_COLS if c in df.columns]
    return base + [c for c in present_optionals if c not in base]


def create_targets(
    df: pd.DataFrame, horizons: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    """Create log-return targets for each horizon (bars ahead), no leakage.

    Returns:
        df with target columns added, and list of target column names.
        Rows where any target is NaN are dropped.
    """
    target_cols = []
    for h in horizons:
        col = f"target_h{h}"
        df[col] = np.log(df["close"].shift(-h) / df["close"])
        target_cols.append(col)
    return df.dropna(subset=target_cols), target_cols
