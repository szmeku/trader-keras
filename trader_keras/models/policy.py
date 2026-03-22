"""Actor-critic MLP for trading env mixed action space.

Architecture:
    Input: (batch, obs_dim)  — flat observation vector
    → Shared MLP (2x hidden, ReLU)
    Outputs:
        action_logits: (batch, 6) — categorical logits for action_type
        p0_params: (batch, 2) — mean + log_std for p0 (Gaussian, clipped to [0,1])
        p1_params: (batch, 2) — mean + log_std for p1 (Gaussian, clipped to [-1,1])
        value: (batch, 1) — state value estimate
"""
from __future__ import annotations

import keras
from keras import layers

N_ACTION_TYPES = 6


def build_policy_model(obs_dim: int, cfg) -> keras.Model:
    """Build actor-critic MLP for the trading environment."""
    inputs = keras.Input(shape=(obs_dim,), name="observation")

    x = layers.Dense(cfg.hidden_size, activation="relu", name="shared_0")(inputs)
    x = layers.Dense(cfg.hidden_size, activation="relu", name="shared_1")(x)

    action_logits = layers.Dense(N_ACTION_TYPES, name="action_logits")(x)
    p0_params = layers.Dense(2, name="p0_params")(x)
    p1_params = layers.Dense(2, name="p1_params")(x)
    value = layers.Dense(1, name="value")(x)

    return keras.Model(
        inputs=inputs,
        outputs=[action_logits, p0_params, p1_params, value],
        name="actor_critic",
    )
