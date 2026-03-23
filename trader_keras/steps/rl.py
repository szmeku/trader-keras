"""RL pipeline step: PPO training with BPTT through full sequence."""
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
    """PPO training: each epoch = full sequential pass through the env."""
    cfg = ctx["cfg"]
    rl = cfg.rl
    env = ctx["env"]

    policy = build_policy_model(env.obs_dim, cfg.backbone)
    optimizer = keras.optimizers.AdamW(learning_rate=rl.lr)

    lookback = cfg.env.lookback
    balance = cfg.env.balance
    params = env.params
    n_steps = env.n_bars
    hidden_size = cfg.backbone.hidden_size
    num_layers = cfg.backbone.num_layers
    hidden_shape = (num_layers, hidden_size)

    collect = build_collect_rollout(policy, params, lookback=lookback, balance=balance)

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
    wandb.summary["n_params"] = n_params
    wandb.summary["n_train"] = n_steps
    wandb.summary["memo_ratio"] = memo_ratio
    wandb.summary["info_ratio"] = info_ratio

    rng = jax.random.PRNGKey(cfg.backbone.seed)
    for epoch in range(rl.n_epochs):
        obs, state, bar_feats = reset(params, lookback=lookback, balance=balance)
        hidden = jnp.zeros(hidden_shape)

        rng, rollout_key = jax.random.split(rng)
        trainable = [v.value for v in policy.trainable_variables]
        non_trainable = [v.value for v in policy.non_trainable_variables]

        transitions, _ = collect(
            rollout_key, trainable, non_trainable,
            state, obs, bar_feats, hidden, jnp.int32(0), n_steps,
        )

        rollout = {k: np.asarray(v) for k, v in transitions.items()}

        # Diagnostics: what is the agent actually doing?
        acts = rollout["action_type"]
        act_names = ["HOLD", "LIM_BUY", "LIM_SELL", "STP_BUY", "STP_SELL", "CANCEL"]
        act_counts = " ".join(f"{act_names[i]}={int((acts==i).sum())}" for i in range(6))
        n_closes = int((rollout["reward"] != 0).sum())
        n_nonzero_reward = float(rollout["reward"][rollout["reward"] != 0].sum()) if n_closes else 0.0
        advantages, returns = _compute_gae(
            rollout["reward"], rollout["value"], rollout["done"],
            rl.gamma, rl.gae_lambda,
        )

        loss = _train_bptt(policy, optimizer, rollout, advantages, returns, rl, hidden_shape)

        wandb.log({
            "rl/epoch": epoch,
            "rl/mean_reward": float(rollout["reward"].mean()),
            "rl/loss": loss,
        })
        logger.info("Epoch %d — reward=%.4f loss=%.4f | %s | closes=%d",
                    epoch, rollout["reward"].mean(), loss, act_counts, n_closes)

    model_path = Path("outputs") / "policy.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(model_path))

    ctx["model"] = policy
    ctx["model_path"] = model_path
    return ctx


def _train_bptt(
    policy: keras.Model, optimizer: keras.Optimizer,
    rollout: dict, advantages: np.ndarray, returns: np.ndarray,
    rl_cfg, hidden_shape: tuple[int, ...],
) -> float:
    """Single BPTT gradient step: re-run GRU forward, backprop through time."""
    trainable = policy.trainable_variables
    non_trainable = policy.non_trainable_variables

    obs_seq = jnp.array(rollout["obs"])
    batch = {
        "action_types": jnp.array(rollout["action_type"]),
        "p0s": jnp.array(rollout["p0"]),
        "p1s": jnp.array(rollout["p1"]),
        "old_log_probs": jnp.array(rollout["log_prob"]),
        "advantages": jnp.array(advantages),
        "returns": jnp.array(returns),
    }

    def loss_fn(trainable_vals, non_trainable_vals):
        def scan_step(hidden, obs_t):
            outputs, _ = policy.stateless_call(
                trainable_vals, non_trainable_vals,
                [obs_t[jnp.newaxis], hidden[jnp.newaxis]], training=True,
            )
            logits, p0_params, p1_params, value, new_hidden = outputs
            return new_hidden[0], (logits[0], p0_params[0], p1_params[0], value[0])

        init_hidden = jnp.zeros(hidden_shape)
        _, (logits, p0_params, p1_params, values) = jax.lax.scan(
            scan_step, init_hidden, obs_seq,
        )
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


def _clip_grads(grads: list, max_norm: float) -> list:
    """Global norm gradient clipping."""
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads))
    clip_coef = jnp.minimum(max_norm / (total_norm + 1e-6), 1.0)
    return [g * clip_coef for g in grads]
