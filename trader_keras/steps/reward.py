"""Reward functions for RL training.

Each function takes (raw_reward, close_info, balance) and returns a scalar reward.
Built before JIT so string dispatch doesn't affect compilation.
"""
from __future__ import annotations

import jax.numpy as jnp


def build_reward_fn(cfg):
    """Build a reward function from config. Returns a JAX-compatible function."""
    reward_type = cfg.type

    if reward_type == "close_only":
        def reward_fn(raw_reward, close_info, balance):
            return jnp.where(
                close_info.has_close, close_info.realized_pnl / balance, 0.0,
            )

    elif reward_type == "equity":
        def reward_fn(raw_reward, close_info, balance):
            return raw_reward / balance

    elif reward_type == "pbrs":
        alpha = cfg.alpha

        def reward_fn(raw_reward, close_info, balance):
            close_reward = jnp.where(
                close_info.has_close, close_info.realized_pnl / balance, 0.0,
            )
            return raw_reward / balance + alpha * close_reward

    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    return reward_fn
