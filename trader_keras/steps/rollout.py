"""JIT-compiled rollout collection via lax.scan.

Wires policy forward pass + JAX action sampling + env step into a single
compiled loop. ~1000x faster than the Python for-loop equivalent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from icmarkets_env.action import decode_action
from icmarkets_env.env import reset, step as env_step


def _sample_action_jax(key, logits, p0_params, p1_params):
    """Sample action in JAX (categorical + 2 gaussians). Returns action, log_prob."""
    k1, k2, k3 = jax.random.split(key, 3)

    # Categorical
    action_type = jax.random.categorical(k1, logits)

    # Gaussian p0
    p0_mean, p0_log_std = p0_params[0], p0_params[1]
    p0_std = jnp.exp(jnp.clip(p0_log_std, -5.0, 2.0))
    p0_raw = p0_mean + p0_std * jax.random.normal(k2)
    p0 = jnp.clip(p0_raw, 0.0, 1.0)

    # Gaussian p1
    p1_mean, p1_log_std = p1_params[0], p1_params[1]
    p1_std = jnp.exp(jnp.clip(p1_log_std, -5.0, 2.0))
    p1_raw = p1_mean + p1_std * jax.random.normal(k3)
    p1 = jnp.clip(p1_raw, -1.0, 1.0)

    # Log probability
    log_probs_cat = jax.nn.log_softmax(logits)
    cat_lp = log_probs_cat[action_type]
    p0_lp = -0.5 * ((p0_raw - p0_mean) / p0_std) ** 2 - jnp.log(p0_std) - 0.5 * jnp.log(2 * jnp.pi)
    p1_lp = -0.5 * ((p1_raw - p1_mean) / p1_std) ** 2 - jnp.log(p1_std) - 0.5 * jnp.log(2 * jnp.pi)
    log_prob = cat_lp + p0_lp + p1_lp

    return action_type, p0, p1, log_prob


def build_collect_rollout(policy, env_params, lookback, balance):
    """Build a JIT-compiled rollout function for the given policy and env.

    Returns a function: (rng_key, trainable_vars, non_trainable_vars, init_state,
                          init_obs, bar_feats, init_hidden, start_idx, n_steps)
                      -> (transitions, (final_state, final_obs, final_hidden, final_idx))
    """
    spec = env_params.spec

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
            # Close-only reward: normalized PnL on trade close, else 0
            reward = jnp.where(close_info.has_close, close_info.pnl / balance, 0.0)

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
