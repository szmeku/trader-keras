"""Actor-critic GRU for trading env mixed action space.

Architecture:
    Inputs: (batch, obs_dim) observation + (batch, hidden_size) GRU hidden state
    -> GRU (single step, streaming)
    Outputs:
        action_logits: (batch, 6) — categorical logits for action_type
        p0_params: (batch, 2) — mean + log_std for p0 (Gaussian, clipped to [0,1])
        p1_params: (batch, 2) — mean + log_std for p1 (Gaussian, clipped to [-1,1])
        value: (batch, 1) — state value estimate
        new_hidden: (batch, hidden_size) — updated GRU hidden state
"""
from __future__ import annotations

import keras
from keras import layers

N_ACTION_TYPES = 6


def build_policy_model(obs_dim: int, cfg) -> keras.Model:
    """Build actor-critic GRU for the trading environment."""
    obs_input = keras.Input(shape=(obs_dim,), name="observation")
    h_input = keras.Input(shape=(cfg.hidden_size,), name="hidden_state")

    x = layers.Reshape((1, obs_dim))(obs_input)
    gru = layers.GRU(cfg.hidden_size, return_state=True, name="gru")
    x, new_h = gru(x, initial_state=h_input)

    action_logits = layers.Dense(N_ACTION_TYPES, name="action_logits")(x)
    p0_params = layers.Dense(2, name="p0_params")(x)
    p1_params = layers.Dense(2, name="p1_params")(x)
    value = layers.Dense(1, name="value")(x)

    return keras.Model(
        inputs=[obs_input, h_input],
        outputs=[action_logits, p0_params, p1_params, value, new_h],
        name="actor_critic",
    )
