"""Unit tests for data pipeline."""
import numpy as np
import pandas as pd

from trader_keras.constants import FEATURE_COLS
from trader_keras.data.features import create_features, create_targets


def _make_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "spread": rng.uniform(0.0001, 0.001, n),
        }
    )


def test_create_features_no_leakage():
    bars = _make_bars(200)
    df = create_features(bars)
    for col in FEATURE_COLS:
        assert col in df.columns
    assert not df[FEATURE_COLS].isna().any().any()
    assert len(df) < len(bars)


def test_create_features_shape():
    bars = _make_bars(200)
    df = create_features(bars)
    assert all(c in df.columns for c in FEATURE_COLS)


def test_create_targets():
    bars = _make_bars(200)
    df = create_features(bars)
    df2, target_cols = create_targets(df.copy(), horizons=[1, 5, 10])
    assert len(target_cols) == 3
    assert all(c in df2.columns for c in target_cols)
    assert not df2[target_cols].isna().any().any()
