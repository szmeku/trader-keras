"""RL pipeline step: PPO training loop with actor-critic policy."""
from __future__ import annotations

import logging

import jax
import keras
import numpy as np
import wandb
from omegaconf import OmegaConf

from ..models.policy import build_policy_model
from ..pipeline import Ctx, step
from .ppo_loss import ppo_loss_from_outputs

logger = logging.getLogger(__name__)


def _sample_action(
    policy_output: tuple,
) -> tuple[int, float, float, float, float]:
    """Sample action from policy outputs (single observation)."""
    logits, p0_params, p1_params, value = policy_output
    logits = np.asarray(logits[0])
    p0_params = np.asarray(p0_params[0])
    p1_params = np.asarray(p1_params[0])
    value_f = float(np.asarray(value[0, 0]))

    # Categorical sample for action_type
    probs = _softmax(logits)
    action_type = int(np.random.choice(len(probs), p=probs))

    # Gaussian sample for p0 (clipped to [0, 1])
    p0_mean, p0_log_std = float(p0_params[0]), float(p0_params[1])
    p0_std = np.exp(np.clip(p0_log_std, -5.0, 2.0))
    p0_raw = np.random.normal(p0_mean, p0_std)
    p0 = float(np.clip(p0_raw, 0.0, 1.0))

    # Gaussian sample for p1 (clipped to [-1, 1])
    p1_mean, p1_log_std = float(p1_params[0]), float(p1_params[1])
    p1_std = np.exp(np.clip(p1_log_std, -5.0, 2.0))
    p1_raw = np.random.normal(p1_mean, p1_std)
    p1 = float(np.clip(p1_raw, -1.0, 1.0))

    # Combined log probability
    log_prob = (
        np.log(probs[action_type] + 1e-8)
        + _gaussian_log_prob(p0_raw, p0_mean, p0_std)
        + _gaussian_log_prob(p1_raw, p1_mean, p1_std)
    )
    return action_type, p0, p1, float(log_prob), value_f


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _gaussian_log_prob(x: float, mean: float, std: float) -> float:
    return float(-0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi))


def _collect_rollout(
    env, policy: keras.Model, rollout_steps: int,
) -> dict[str, np.ndarray]:
    """Step through env, collect transitions for PPO."""
    n = rollout_steps
    bufs = {
        "obs": np.empty((n, env.obs_dim), np.float32),
        "action_types": np.empty(n, np.int32),
        "p0s": np.empty(n, np.float32), "p1s": np.empty(n, np.float32),
        "rewards": np.empty(n, np.float32), "dones": np.empty(n, np.float32),
        "log_probs": np.empty(n, np.float32), "values": np.empty(n, np.float32),
    }
    obs = env.reset()
    for t in range(n):
        bufs["obs"][t] = obs
        outputs = policy(obs[np.newaxis], training=False)
        action_type, p0, p1, log_prob, value = _sample_action(outputs)
        obs, reward, done, _ = env.step(action_type, p0, p1)
        bufs["action_types"][t] = action_type
        bufs["p0s"][t], bufs["p1s"][t] = p0, p1
        bufs["rewards"][t], bufs["dones"][t] = reward, float(done)
        bufs["log_probs"][t], bufs["values"][t] = log_prob, value
        if done:
            obs = env.reset()
    return bufs


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
    """PPO training loop."""
    cfg = ctx["cfg"]
    rl = cfg.rl
    env = ctx["env"]

    policy = build_policy_model(env.obs_dim, cfg.backbone)
    optimizer = keras.optimizers.AdamW(learning_rate=rl.lr)

    wandb.init(
        project=cfg.wandb.project, tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
    )

    total_steps = 0
    while total_steps < rl.total_timesteps:
        rollout = _collect_rollout(env, policy, rl.rollout_steps)
        advantages, returns = _compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rl.gamma, rl.gae_lambda,
        )
        _train_on_rollout(policy, optimizer, rollout, advantages, returns, rl)
        total_steps += rl.rollout_steps

        wandb.log({
            "rl/total_steps": total_steps,
            "rl/mean_reward": float(rollout["rewards"].mean()),
        })
        logger.info("Steps %d — mean_reward=%.4f", total_steps, rollout["rewards"].mean())

    wandb.finish()
    ctx["model"] = policy
    return ctx


def _train_on_rollout(
    policy: keras.Model, optimizer: keras.Optimizer,
    rollout: dict, advantages: np.ndarray, returns: np.ndarray,
    rl_cfg,
) -> None:
    """Run PPO epochs + minibatch updates on one rollout."""
    n = len(advantages)
    indices = np.arange(n)

    for _ in range(rl_cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, rl_cfg.batch_size):
            idx = indices[start : start + rl_cfg.batch_size]
            batch = {
                "obs": rollout["obs"][idx],
                "action_types": rollout["action_types"][idx],
                "p0s": rollout["p0s"][idx],
                "p1s": rollout["p1s"][idx],
                "old_log_probs": rollout["log_probs"][idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
            }
            _update_step(policy, optimizer, batch, rl_cfg)


def _update_step(
    policy: keras.Model, optimizer: keras.Optimizer,
    batch: dict, rl_cfg,
) -> None:
    """Single gradient step via stateless_call + jax.grad."""
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

    grads = jax.grad(loss_fn)(
        [v.value for v in trainable],
        [v.value for v in non_trainable],
    )
    optimizer.apply(grads, trainable)
