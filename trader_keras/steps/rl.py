"""RL pipeline step: PPO training loop with JIT-compiled rollouts."""
from __future__ import annotations

import logging
from pathlib import Path

import jax
from jax import numpy as jnp
import keras
import numpy as np
import wandb
from omegaconf import OmegaConf

from icmarkets_env.env import reset

from ..models.policy import build_policy_model
from ..pipeline import Ctx, step
from .ppo_loss import ppo_loss_from_outputs
from .rollout import build_collect_rollout

logger = logging.getLogger(__name__)


def _compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
    gamma: float, gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


@step
def fit_rl(ctx: Ctx) -> Ctx:
    """PPO training loop with JIT-compiled rollouts."""
    cfg = ctx["cfg"]
    rl = cfg.rl
    env = ctx["env"]

    policy = build_policy_model(env.obs_dim, cfg.backbone)
    optimizer = keras.optimizers.AdamW(learning_rate=rl.lr)

    lookback = cfg.env.lookback
    balance = cfg.env.balance
    params = env._params

    obs, state, bar_feats = reset(params, lookback=lookback, balance=balance)
    collect = build_collect_rollout(policy, params, lookback=lookback, balance=balance)

    wandb.init(
        project=cfg.wandb.project, tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
    )

    rng = jax.random.PRNGKey(cfg.backbone.seed)
    total_steps = 0
    while total_steps < rl.total_timesteps:
        rng, rollout_key = jax.random.split(rng)
        trainable = [v.value for v in policy.trainable_variables]
        non_trainable = [v.value for v in policy.non_trainable_variables]

        transitions, state = collect(
            rollout_key, trainable, non_trainable,
            state, obs, bar_feats, jnp.int32(0), rl.rollout_steps,
        )

        # Convert to numpy for GAE + training
        rollout = {k: np.asarray(v) for k, v in transitions.items()}
        advantages, returns = _compute_gae(
            rollout["reward"], rollout["value"], rollout["done"],
            rl.gamma, rl.gae_lambda,
        )
        mean_loss = _train_on_rollout(policy, optimizer, rollout, advantages, returns, rl)
        total_steps += rl.rollout_steps

        wandb.log({
            "rl/total_steps": total_steps,
            "rl/mean_reward": float(rollout["reward"].mean()),
            "rl/loss": mean_loss,
        })
        logger.info("Steps %d — reward=%.4f loss=%.4f", total_steps, rollout["reward"].mean(), mean_loss)

    model_path = Path("outputs") / "policy.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(model_path))

    ctx["model"] = policy
    ctx["model_path"] = model_path
    return ctx


def _train_on_rollout(
    policy: keras.Model, optimizer: keras.Optimizer,
    rollout: dict, advantages: np.ndarray, returns: np.ndarray,
    rl_cfg,
) -> float:
    """Run PPO epochs + minibatch updates on one rollout. Returns mean loss."""
    n = len(advantages)
    indices = np.arange(n)
    losses: list[float] = []

    for _ in range(rl_cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, rl_cfg.batch_size):
            idx = indices[start : start + rl_cfg.batch_size]
            batch = {
                "obs": rollout["obs"][idx],
                "action_types": rollout["action_type"][idx],
                "p0s": rollout["p0"][idx],
                "p1s": rollout["p1"][idx],
                "old_log_probs": rollout["log_prob"][idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
            }
            losses.append(_update_step(policy, optimizer, batch, rl_cfg))
    return float(np.mean(losses))


def _clip_grads(grads: list, max_norm: float) -> list:
    """Global norm gradient clipping."""
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads))
    clip_coef = jnp.minimum(max_norm / (total_norm + 1e-6), 1.0)
    return [g * clip_coef for g in grads]


def _update_step(
    policy: keras.Model, optimizer: keras.Optimizer,
    batch: dict, rl_cfg,
) -> float:
    """Single gradient step via stateless_call + jax.grad. Returns loss."""
    trainable = policy.trainable_variables
    non_trainable = policy.non_trainable_variables

    def loss_fn(trainable_vals, non_trainable_vals):
        outputs, _ = policy.stateless_call(
            trainable_vals, non_trainable_vals, batch["obs"], training=True,
        )
        logits, p0_params, p1_params, values = outputs
        return ppo_loss_from_outputs(
            logits, p0_params, p1_params, values, batch,
            rl_cfg.clip_epsilon, rl_cfg.value_coeff, rl_cfg.entropy_coeff,
        )

    loss, grads = jax.value_and_grad(loss_fn)(
        [v.value for v in trainable],
        [v.value for v in non_trainable],
    )
    if rl_cfg.clip_grad_norm > 0:
        grads = _clip_grads(grads, rl_cfg.clip_grad_norm)
    optimizer.apply(grads, trainable)
    return float(loss)
