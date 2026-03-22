"""Environment pipeline step: wrap icmarkets_env."""
from __future__ import annotations

import logging

from icmarkets_env import TradingEnv

from ..pipeline import Ctx, step

logger = logging.getLogger(__name__)


@step
def env(ctx: Ctx) -> Ctx:
    """Create TradingEnv from config and add to context."""
    cfg = ctx["cfg"]
    e = cfg.env
    trading_env = TradingEnv.from_symbol(
        e.symbol, e.date_start, e.date_end,
        lookback=e.lookback, balance=e.balance, leverage=e.leverage,
    )
    logger.info("Env: %s %s→%s, %d bars", e.symbol, e.date_start, e.date_end, trading_env.n_bars)
    ctx["env"] = trading_env
    return ctx
