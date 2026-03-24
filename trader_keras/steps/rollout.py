"""JIT-compiled rollout collection via lax.scan.

Wires policy forward pass + JAX action sampling + env step into a single
compiled loop. ~1000x faster than the Python for-loop equivalent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from icmarkets_env.action import decode_action
from icmarkets_env.env import reset, step as env_step


def beta_params(raw_a, raw_b):
    """Raw network output → Beta(alpha, beta) with alpha, beta > 1."""
    return jax.nn.softplus(raw_a) + 1.0, jax.nn.softplus(raw_b) + 1.0


def beta_log_prob(x, alpha, beta):
    """Beta distribution log PDF."""
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jax.scipy.stats.beta.logpdf(x, alpha, beta)


def _sample_action_jax(key, logits, p0_params, p1_params):
    """Sample action: categorical + 2 Beta distributions. Returns action, log_prob."""
    k1, k2, k3 = jax.random.split(key, 3)

    # Categorical
    action_type = jax.random.categorical(k1, logits)

    # Beta for p0 ∈ [0, 1]
    p0_a, p0_b = beta_params(p0_params[0], p0_params[1])
    p0 = jnp.clip(jax.random.beta(k2, p0_a, p0_b), 1e-6, 1.0 - 1e-6)

    # Shifted Beta for p1 ∈ [-1, 1]: sample unit ∈ [0,1], then 2*unit - 1
    p1_a, p1_b = beta_params(p1_params[0], p1_params[1])
    p1_unit = jnp.clip(jax.random.beta(k3, p1_a, p1_b), 1e-6, 1.0 - 1e-6)
    p1 = 2.0 * p1_unit - 1.0

    # Log probability
    cat_lp = jax.nn.log_softmax(logits)[action_type]
    p0_lp = beta_log_prob(p0, p0_a, p0_b)
    p1_lp = beta_log_prob(p1_unit, p1_a, p1_b) - jnp.log(2.0)
    log_prob = cat_lp + p0_lp + p1_lp

    return action_type, p0, p1, log_prob


def build_collect_rollout(policy, env_params, lookback, balance, reward_fn=None):
    """Build a JIT-compiled rollout function for the given policy and env.

    Args:
        reward_fn: (raw_reward, close_info, balance) -> scalar. Built from config
                   before JIT so string dispatch doesn't affect compilation.
    """
    spec = env_params.spec
    if reward_fn is None:
        from .reward import build_reward_fn
        from types import SimpleNamespace
        reward_fn = build_reward_fn(SimpleNamespace(type="pbrs", alpha=1.0))

    def _collect(rng_key, trainable_vars, non_trainable_vars,
                 init_state, init_obs, bar_feats, init_hidden, start_idx, n_steps):

        def scan_body(carry, _step_offset):
            state, obs, hidden, step_idx, rng = carry

            # Policy forward (stateless)
            outputs, _ = policy.stateless_call(
                trainable_vars, non_trainable_vars,
                [obs[jnp.newaxis], hidden[jnp.newaxis]], training=False,
            )
            logits, p0_params, p1_params, value, new_hidden = outputs
            logits = logits[0]
            p0_params = p0_params[0]
            p1_params = p1_params[0]
            value_f = value[0, 0]
            new_hidden = new_hidden[0]

            # Sample action
            rng, sample_key = jax.random.split(rng)
            action_type, p0, p1, log_prob = _sample_action_jax(
                sample_key, logits, p0_params, p1_params,
            )

            # Decode to env action
            safe_idx = jnp.clip(step_idx, 0, env_params.closes.shape[0] - 1)
            bid = env_params.closes[safe_idx]
            spread = env_params.spreads[safe_idx]
            ask = bid + spread * spec.point
            action = decode_action(
                jnp.int32(action_type),
                jnp.array([p0, p1]),
                balance=state.balance, bid=bid, ask=ask,
                spec=spec,
            )

            # Env step
            new_obs, new_state, raw_reward, done, close_info = env_step(
                state, action, env_params, bar_feats,
                step_idx, lookback=lookback, balance=balance,
            )
            reward = reward_fn(raw_reward, close_info, balance)

            # Auto-reset on done
            reset_obs, reset_state, _ = reset(env_params, lookback=lookback, balance=balance)
            final_state = jax.tree.map(
                lambda r, n: jnp.where(done, r, n), reset_state, new_state,
            )
            final_obs = jnp.where(done, reset_obs, new_obs)
            final_hidden = jnp.where(done, jnp.zeros_like(hidden), new_hidden)
            next_idx = jnp.where(done, jnp.int32(0), step_idx + 1)

            transition = {
                "obs": obs,
                "hidden": hidden,
                "action_type": jnp.int32(action_type),
                "p0": jnp.float32(p0),
                "p1": jnp.float32(p1),
                "reward": jnp.float32(reward),
                "done": jnp.float32(done),
                "log_prob": jnp.float32(log_prob),
                "value": jnp.float32(value_f),
            }
            return (final_state, final_obs, final_hidden, next_idx, rng), transition

        indices = jnp.arange(n_steps)
        (final_state, final_obs, final_hidden, final_idx, _), transitions = jax.lax.scan(
            scan_body, (init_state, init_obs, init_hidden, start_idx, rng_key), indices,
        )
        return transitions, (final_state, final_obs, final_hidden, final_idx)

    return jax.jit(_collect, static_argnums=(8,))
