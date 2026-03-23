"""Actor-critic GRU for trading env mixed action space.

Architecture:
    Inputs: (batch, obs_dim) observation + (batch, num_layers, hidden_size) GRU hidden state
    -> stacked GRU (single step, streaming)
    Outputs:
        action_logits: (batch, 6) — categorical logits for action_type
        p0_params: (batch, 2) — Beta distribution params for p0 ∈ [0,1]
        p1_params: (batch, 2) — Beta distribution params for p1 ∈ [-1,1] (shifted)
        value: (batch, 1) — state value estimate
        new_hidden: (batch, num_layers, hidden_size) — updated GRU hidden state
"""
from __future__ import annotations

import keras
from keras import layers

N_ACTION_TYPES = 6


def build_policy_model(obs_dim: int, cfg) -> keras.Model:
    """Build actor-critic GRU for the trading environment.

    Hidden state shape: (batch, num_layers, hidden_size).
    """
    num_layers = cfg.num_layers
    hidden_size = cfg.hidden_size

    obs_input = keras.Input(shape=(obs_dim,), name="observation")
    h_input = keras.Input(shape=(num_layers, hidden_size), name="hidden_state")

    x = layers.Reshape((1, obs_dim))(obs_input)
    new_hiddens = []
    for i in range(num_layers):
        h_i = layers.Lambda(lambda h, idx=i: h[:, idx, :], name=f"h_slice_{i}")(h_input)
        gru = layers.GRU(hidden_size, return_state=True, name=f"gru_{i}")
        x, new_h = gru(x, initial_state=h_i)
        new_hiddens.append(new_h)
        if i < num_layers - 1:
            x = layers.Reshape((1, hidden_size))(x)

    new_h_cat = layers.Concatenate(axis=-1)(new_hiddens) if num_layers > 1 else new_hiddens[0]
    new_h_out = layers.Reshape((num_layers, hidden_size), name="hidden_out")(new_h_cat)

    action_logits = layers.Dense(N_ACTION_TYPES, name="action_logits")(x)
    p0_params = layers.Dense(2, name="p0_params")(x)
    p1_params = layers.Dense(2, name="p1_params")(x)
    value = layers.Dense(1, name="value")(x)

    return keras.Model(
        inputs=[obs_input, h_input],
        outputs=[action_logits, p0_params, p1_params, value, new_h_out],
        name="actor_critic",
    )
