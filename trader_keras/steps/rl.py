"""RL pipeline step: PPO training via lax.scan over TBPTT chunks.

The inner chunk loop (collect → GAE → PPO grad step) runs entirely on-device
as a single lax.scan, eliminating Python round-trips between chunks.
"""
from __future__ import annotations

import logging

import jax
from jax import numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf

from icmarkets_env.env import reset

from ..models.policy import build_policy_model
from ..pipeline import Ctx, step
from .ppo_loss import ppo_loss_from_outputs
from .reward import build_reward_fn
from .rollout import build_collect_rollout

logger = logging.getLogger(__name__)


@jax.jit
def _compute_gae(
    rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray,
    gamma: float, gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GAE via reverse lax.scan — fully on-device."""
    next_values = jnp.concatenate([values[1:], jnp.zeros(1)])
    non_terminals = 1.0 - dones
    deltas = rewards + gamma * next_values * non_terminals - values

    def scan_fn(last_gae, t):
        delta = deltas[t]
        nt = non_terminals[t]
        gae = delta + gamma * gae_lambda * nt * last_gae
        return gae, gae

    indices = jnp.arange(rewards.shape[0] - 1, -1, -1)
    _, advantages_rev = jax.lax.scan(scan_fn, jnp.float32(0.0), indices)
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def _build_chunk_step(policy, collect_fn, tx, non_trainable, bar_feats, rl_cfg):
    """Build lax.scan body: collect → GAE → PPO update, all on-device."""
    chunk = rl_cfg.tbptt_chunk
    gamma, gae_lambda = rl_cfg.gamma, rl_cfg.gae_lambda
    clip_eps, val_c, ent_c = rl_cfg.clip_epsilon, rl_cfg.value_coeff, rl_cfg.entropy_coeff

    def chunk_step(carry, _):
        trainable, opt_state, state, obs, hidden, step_idx, rng = carry
        rng, rollout_key = jax.random.split(rng)

        transitions, (state, obs, hidden, step_idx) = collect_fn(
            rollout_key, trainable, non_trainable,
            state, obs, bar_feats, hidden, step_idx, chunk,
        )

        advantages, returns = _compute_gae(
            transitions["reward"], transitions["value"], transitions["done"],
            gamma, gae_lambda,
        )

        h0 = transitions["hidden"][0]

        def loss_fn(params):
            def fwd(h, obs_t):
                out, _ = policy.stateless_call(
                    params, non_trainable,
                    [obs_t[jnp.newaxis], h[jnp.newaxis]], training=True,
                )
                logits, p0_p, p1_p, val, new_h = out
                return new_h[0], (logits[0], p0_p[0], p1_p[0], val[0])

            _, (logits, p0_p, p1_p, vals) = jax.lax.scan(fwd, h0, transitions["obs"])
            return ppo_loss_from_outputs(
                logits, p0_p, p1_p, vals,
                {
                    "action_types": transitions["action_type"],
                    "p0s": transitions["p0"],
                    "p1s": transitions["p1"],
                    "old_log_probs": transitions["log_prob"],
                    "advantages": advantages,
                    "returns": returns,
                },
                clip_eps, val_c, ent_c,
            )

        loss, grads = jax.value_and_grad(loss_fn)(trainable)
        updates, opt_state = tx.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)

        new_carry = (trainable, opt_state, state, obs, hidden, step_idx, rng)
        metrics = {
            "reward": transitions["reward"],
            "action_type": transitions["action_type"],
            "loss": loss,
        }
        return new_carry, metrics

    return chunk_step


@step
def fit_rl(ctx: Ctx) -> Ctx:
    """PPO training: each epoch = full sequential pass through the env."""
    cfg = ctx["cfg"]
    rl = cfg.rl
    env = ctx["env"]

    policy = build_policy_model(env.obs_dim, cfg.backbone)

    # Optax optimizer — functional, lax.scan-compatible
    components = []
    if rl.clip_grad_norm > 0:
        components.append(optax.clip_by_global_norm(rl.clip_grad_norm))
    components.append(optax.adamw(learning_rate=rl.lr))
    tx = optax.chain(*components)

    lookback = cfg.env.lookback
    balance = cfg.env.balance
    params = env.params
    n_steps = env.n_bars
    hidden_shape = (cfg.backbone.num_layers, cfg.backbone.hidden_size)

    reward_fn = build_reward_fn(rl.reward)
    logger.info("Reward: type=%s", rl.reward.type)
    collect = build_collect_rollout(
        policy, params, lookback=lookback, balance=balance, reward_fn=reward_fn,
    )

    wandb.init(
        project=cfg.wandb.project, tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
    )

    n_params = policy.count_params()
    obs_dim = env.obs_dim
    memo_ratio = n_steps / n_params if n_params > 0 else float("inf")
    info_ratio = (n_steps * obs_dim) / n_params if n_params > 0 else float("inf")
    logger.info(
        "steps=%d, obs_dim=%d, params=%d | memo_ratio=%.2f, info_ratio=%.2f",
        n_steps, obs_dim, n_params, memo_ratio, info_ratio,
    )
    wandb.summary.update({
        "n_params": n_params, "n_train": n_steps,
        "memo_ratio": memo_ratio, "info_ratio": info_ratio,
    })

    chunk = rl.tbptt_chunk
    n_chunks = n_steps // chunk
    act_names = ["HOLD", "LIM_BUY", "LIM_SELL", "STP_BUY", "STP_SELL", "CANCEL"]

    # Extract params once; they live as JAX arrays from here on
    trainable = [v.value for v in policy.trainable_variables]
    non_trainable = [v.value for v in policy.non_trainable_variables]
    opt_state = tx.init(trainable)

    # bar_feats is constant across epochs — compute once
    _, _, bar_feats = reset(params, lookback=lookback, balance=balance)

    chunk_step = _build_chunk_step(policy, collect, tx, non_trainable, bar_feats, rl)

    @jax.jit
    def run_epoch(trainable, opt_state, state, obs, hidden, step_idx, rng):
        carry = (trainable, opt_state, state, obs, hidden, step_idx, rng)
        final_carry, metrics = jax.lax.scan(chunk_step, carry, jnp.arange(n_chunks))
        return final_carry[0], final_carry[1], final_carry[-1], metrics

    rng = jax.random.PRNGKey(cfg.backbone.seed)
    for epoch in range(rl.n_epochs):
        obs, state, _ = reset(params, lookback=lookback, balance=balance)
        hidden = jnp.zeros(hidden_shape)

        trainable, opt_state, rng, metrics = run_epoch(
            trainable, opt_state, state, obs, hidden, jnp.int32(0), rng,
        )

        # Transfer to host once per epoch
        all_rewards = np.asarray(metrics["reward"].reshape(-1))
        all_acts = np.asarray(metrics["action_type"].reshape(-1))
        mean_reward = float(all_rewards.mean())
        mean_loss = float(np.mean(np.asarray(metrics["loss"])))
        epoch_closes = int((all_rewards != 0).sum())
        act_counts = " ".join(f"{act_names[i]}={int((all_acts==i).sum())}" for i in range(6))

        total_pnl = float(all_rewards.sum()) * balance
        wandb.log({
            "rl/epoch": epoch, "rl/mean_reward": mean_reward,
            "rl/loss": mean_loss, "rl/total_pnl": total_pnl,
        })
        logger.info("Epoch %d — reward=%f loss=%.4f pnl=$%.2f | %s | closes=%d",
                    epoch, mean_reward, mean_loss, total_pnl, act_counts, epoch_closes)

    # Write optimized params back to Keras model for save step
    for var, val in zip(policy.trainable_variables, trainable):
        var.assign(val)

    ctx["model"] = policy
    return ctx
