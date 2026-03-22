"""Integration tests for env pipeline step."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from trader_keras.pipeline import Ctx

_DATA_DIR = Path("~/projects/data").expanduser()
_HAS_DATA = (
    _DATA_DIR.exists()
    and any(_DATA_DIR.glob("icmarkets_xauusd_*.parquet"))
    and (_DATA_DIR / "instrument_specs.json").exists()
)

pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="No XAUUSD data in ~/projects/data")


def _make_cfg(**overrides) -> OmegaConf:
    """Minimal env config — short date range for speed."""
    base = {
        "env": {
            "symbol": "XAUUSD",
            "date_start": "2024-06-01",
            "date_end": "2024-07-01",
            "lookback": 30,
            "balance": 10_000.0,
            "leverage": 20.0,
        },
    }
    return OmegaConf.create({**base, **overrides})


class TestEnvStep:
    def test_creates_trading_env(self):
        from trader_keras.steps.env import env

        ctx: Ctx = {"cfg": _make_cfg()}
        ctx = env(ctx)

        assert "env" in ctx

    def test_env_is_trading_env_instance(self):
        from icmarkets_env import TradingEnv
        from trader_keras.steps.env import env

        ctx: Ctx = {"cfg": _make_cfg()}
        ctx = env(ctx)

        assert isinstance(ctx["env"], TradingEnv)

    def test_obs_dim_and_action_types(self):
        from trader_keras.steps.env import env

        ctx: Ctx = {"cfg": _make_cfg()}
        ctx = env(ctx)

        trading_env = ctx["env"]
        assert trading_env.obs_dim > 0
        assert trading_env.n_action_types == 6

    def test_n_bars_positive(self):
        from trader_keras.steps.env import env

        ctx: Ctx = {"cfg": _make_cfg()}
        ctx = env(ctx)

        assert ctx["env"].n_bars > 0

    def test_reset_and_step(self):
        from trader_keras.steps.env import env

        ctx: Ctx = {"cfg": _make_cfg()}
        ctx = env(ctx)

        trading_env = ctx["env"]
        obs = trading_env.reset()
        assert obs.shape == (trading_env.obs_dim,)

        obs2, reward, done, info = trading_env.step(0, 0.0, 0.0)
        assert obs2.shape == (trading_env.obs_dim,)

    def test_step_registered_in_pipeline(self):
        from trader_keras.pipeline import STEPS
        from trader_keras.steps.env import env  # noqa: F401 — triggers registration

        assert "env" in STEPS
