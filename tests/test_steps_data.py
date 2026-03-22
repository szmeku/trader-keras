"""Unit tests for data pipeline steps (load, featurize, window)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from trader_keras.data.features import FEATURE_COLS
from trader_keras.pipeline import Ctx
from trader_keras.steps.data import featurize, load, window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 200) -> pd.DataFrame:
    """Synthetic OHLCV bars — deterministic via seed."""
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


def _make_cfg(**overrides) -> OmegaConf:
    """Build a minimal test config matching Config dataclass."""
    base = {
        "data": {"pattern": "test_*.parquet", "data_dir": "/tmp", "load_limit": 0,
                 "lookback": 10, "stride": 1},
        "train": {
            "horizons": [1, 5],
            "train_ratio": 0.8,
        },
    }
    return OmegaConf.create({**base, **overrides})


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_loads_parquet_into_bars(self, tmp_path):
        bars = _make_bars(50)
        bars.to_parquet(tmp_path / "test_a.parquet", index=False)

        cfg = _make_cfg(data={"pattern": "test_*.parquet",
                              "data_dir": str(tmp_path), "load_limit": 0})
        ctx: Ctx = {"cfg": cfg}
        ctx = load(ctx)

        assert "bars" in ctx
        assert len(ctx["bars"]) == 50

    def test_concats_multiple_files(self, tmp_path):
        for name in ("test_a.parquet", "test_b.parquet"):
            _make_bars(30).to_parquet(tmp_path / name, index=False)

        cfg = _make_cfg(data={"pattern": "test_*.parquet",
                              "data_dir": str(tmp_path), "load_limit": 0})
        ctx: Ctx = {"cfg": cfg}
        ctx = load(ctx)
        assert len(ctx["bars"]) == 60

    def test_load_limit_truncates(self, tmp_path):
        _make_bars(100).to_parquet(tmp_path / "test_a.parquet", index=False)

        cfg = _make_cfg(data={"pattern": "test_*.parquet",
                              "data_dir": str(tmp_path), "load_limit": 25})
        ctx: Ctx = {"cfg": cfg}
        ctx = load(ctx)
        assert len(ctx["bars"]) == 25

    def test_no_files_raises(self, tmp_path):
        cfg = _make_cfg(data={"pattern": "nope_*.parquet",
                              "data_dir": str(tmp_path), "load_limit": 0})
        ctx: Ctx = {"cfg": cfg}
        with pytest.raises(FileNotFoundError):
            load(ctx)


# ---------------------------------------------------------------------------
# featurize
# ---------------------------------------------------------------------------

class TestFeaturize:
    def test_adds_feature_columns(self):
        cfg = _make_cfg()
        ctx: Ctx = {"cfg": cfg, "bars": _make_bars(100)}
        ctx = featurize(ctx)

        for col in FEATURE_COLS:
            assert col in ctx["bars"].columns

    def test_drops_nan_rows(self):
        bars = _make_bars(100)
        cfg = _make_cfg()
        ctx: Ctx = {"cfg": cfg, "bars": bars}
        ctx = featurize(ctx)
        assert not ctx["bars"][FEATURE_COLS].isna().any().any()
        assert len(ctx["bars"]) < 100  # first row dropped (shift)


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------

class TestWindow:
    def _featurized_ctx(self, n: int = 200) -> Ctx:
        """Build a ctx with featurized bars ready for windowing."""
        cfg = _make_cfg()
        ctx: Ctx = {"cfg": cfg, "bars": _make_bars(n)}
        return featurize(ctx)

    def test_produces_train_val_arrays(self):
        ctx = self._featurized_ctx()
        ctx = window(ctx)

        for key in ("x_train", "y_train", "x_val", "y_val", "feature_cols"):
            assert key in ctx, f"missing key: {key}"

    def test_x_shape_lookback(self):
        ctx = self._featurized_ctx()
        ctx = window(ctx)

        lookback = ctx["cfg"].data.lookback
        n_features = len(ctx["feature_cols"])
        assert ctx["x_train"].shape[1] == lookback
        assert ctx["x_train"].shape[2] == n_features

    def test_y_shape_matches_horizons(self):
        ctx = self._featurized_ctx()
        ctx = window(ctx)
        n_horizons = len(ctx["cfg"].train.horizons)
        assert ctx["y_train"].shape[1] == n_horizons

    def test_train_val_split_ratio(self):
        ctx = self._featurized_ctx(300)
        ctx = window(ctx)

        total = len(ctx["x_train"]) + len(ctx["x_val"])
        train_frac = len(ctx["x_train"]) / total
        assert 0.75 <= train_frac <= 0.85  # ~0.8

    def test_stride_reduces_samples(self):
        ctx1 = self._featurized_ctx(200)
        ctx1["cfg"] = OmegaConf.merge(ctx1["cfg"], {"data": {"stride": 1}})
        ctx1 = window(ctx1)
        n1 = len(ctx1["x_train"]) + len(ctx1["x_val"])

        ctx2 = self._featurized_ctx(200)
        ctx2["cfg"] = OmegaConf.merge(ctx2["cfg"], {"data": {"stride": 5}})
        ctx2 = window(ctx2)
        n2 = len(ctx2["x_train"]) + len(ctx2["x_val"])

        assert n2 < n1

    def test_dtype_is_float32(self):
        ctx = self._featurized_ctx()
        ctx = window(ctx)
        assert ctx["x_train"].dtype == np.float32
        assert ctx["y_train"].dtype == np.float32
