"""Trade data → OHLCV bar resampling (pure numpy/pandas, no Numba)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_bar_seconds_from_name(pattern: str) -> int | None:
    """Extract bar_seconds from filename like '*_1m.parquet' → 60."""
    m = re.search(r"_(\d+)([smhd])\.parquet", pattern)
    return int(m.group(1)) * _UNIT_TO_SECONDS[m.group(2)] if m else None


def _to_epoch_seconds(ts_series: pd.Series) -> np.ndarray:
    """Convert a datetime64 Series to integer seconds since epoch (pandas-version agnostic)."""
    # Use numpy timedelta from epoch to avoid pandas version differences (ns vs us)
    epoch = np.datetime64(0, "s")
    return (ts_series.values.astype("datetime64[s]") - epoch).astype(np.int64)


def resample_to_bars(df: pd.DataFrame, bar_seconds: int = 1) -> pd.DataFrame:
    """Resample raw trade data (timestamp, price, amount, side) to OHLCV bars."""
    ts = _to_epoch_seconds(df["timestamp"])
    start = int(ts.min())
    bar_idx = (ts - start) // bar_seconds

    prices = df["price"].values
    amounts = df["amount"].values
    sides = df["side"].values.astype(bool)

    n_bins = int(bar_idx.max()) + 1
    opens = np.full(n_bins, np.nan)
    closes = np.full(n_bins, np.nan)
    highs = np.full(n_bins, np.nan)
    lows = np.full(n_bins, np.nan)
    volumes = np.zeros(n_bins)
    buy_counts = np.zeros(n_bins, dtype=np.int64)
    trade_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(len(prices)):
        b = bar_idx[i]
        p = prices[i]
        if np.isnan(opens[b]):
            opens[b] = p
        closes[b] = p
        highs[b] = p if np.isnan(highs[b]) else max(highs[b], p)
        lows[b] = p if np.isnan(lows[b]) else min(lows[b], p)
        volumes[b] += amounts[i]
        if sides[i]:
            buy_counts[b] += 1
        trade_counts[b] += 1

    valid = ~np.isnan(opens)
    timestamps = pd.to_datetime(
        np.arange(n_bins)[valid] * bar_seconds + start, unit="s"
    )
    buy_ratio = np.where(
        trade_counts[valid] > 0, buy_counts[valid] / trade_counts[valid], 0.5
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens[valid],
            "high": highs[valid],
            "low": lows[valid],
            "close": closes[valid],
            "volume": volumes[valid],
            "buy_ratio": buy_ratio,
        }
    )


def reaggregate_bars(
    bars: pd.DataFrame, source_seconds: int, target_seconds: int
) -> pd.DataFrame:
    """Re-aggregate pre-made OHLCV bars to coarser resolution."""
    if target_seconds == source_seconds:
        return bars
    if target_seconds % source_seconds != 0:
        raise ValueError(
            f"target_seconds ({target_seconds}) must be multiple of source_seconds ({source_seconds})"
        )
    ts = _to_epoch_seconds(bars["timestamp"])
    start = int(ts.min())
    group = (ts - start) // target_seconds

    agg = (
        bars.assign(_group=group)
        .groupby("_group", sort=True)
        .agg(
            timestamp=("timestamp", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )

    # Volume-weighted buy_ratio
    if "buy_ratio" in bars.columns:
        vw_num = (bars["buy_ratio"] * bars["volume"]).values
        vw_num_grp = pd.Series(vw_num).groupby(group).sum().values
        vol_grp = agg["volume"].values
        agg["buy_ratio"] = np.where(vol_grp > 0, vw_num_grp / vol_grp, 0.5)

    return agg
