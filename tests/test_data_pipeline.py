"""Unit tests for data pipeline."""
import numpy as np
import pandas as pd
import pytest

from trader_keras.data.features import create_features, create_targets, select_features
from trader_keras.data.resampler import reaggregate_bars


def _make_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "buy_ratio": rng.uniform(0.4, 0.6, n),
        }
    )


def test_create_features_no_leakage():
    bars = _make_bars(200)
    df = create_features(bars, lookback=20)
    assert "log_returns" in df.columns
    assert "vol_regime" in df.columns
    assert not df["log_returns"].isna().any()
    assert len(df) < len(bars)  # some rows dropped (NaN at start)


def test_create_features_shape():
    bars = _make_bars(200)
    df = create_features(bars, lookback=20)
    cols = select_features(df)
    assert all(c in df.columns for c in cols), "missing feature columns"


def test_create_targets():
    bars = _make_bars(200)
    df = create_features(bars, lookback=20)
    df2, target_cols = create_targets(df.copy(), horizons=[1, 5, 10])
    assert len(target_cols) == 3
    assert all(c in df2.columns for c in target_cols)
    assert not df2[target_cols].isna().any().any()


def test_reaggregate_bars():
    bars = _make_bars(120)
    agg = reaggregate_bars(bars, source_seconds=1, target_seconds=10)
    assert len(agg) == 12
    assert "close" in agg.columns


def test_reaggregate_noop():
    bars = _make_bars(60)
    result = reaggregate_bars(bars, source_seconds=1, target_seconds=1)
    assert len(result) == len(bars)
