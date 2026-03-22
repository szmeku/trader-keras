"""PPO loss computation and Gaussian log-probability helpers."""
from __future__ import annotations

import numpy as np
from keras import ops


def ppo_loss_from_outputs(
    logits, p0_params, p1_params, values, batch: dict,
    clip_eps: float, value_coeff: float, entropy_coeff: float,
) -> float:
    """Compute clipped PPO loss from model outputs."""
    values = ops.squeeze(values, axis=-1)

    # Categorical log probs
    log_probs_cat = ops.log_softmax(logits)
    action_log_probs = ops.take_along_axis(
        log_probs_cat, ops.cast(batch["action_types"][:, None], "int32"), axis=1,
    )
    action_log_probs = ops.squeeze(action_log_probs, axis=-1)

    # Gaussian log probs for p0 and p1
    p0_lp = gaussian_log_prob(batch["p0s"], p0_params[:, 0], p0_params[:, 1])
    p1_lp = gaussian_log_prob(batch["p1s"], p1_params[:, 0], p1_params[:, 1])

    new_log_probs = action_log_probs + p0_lp + p1_lp

    # PPO clipped surrogate
    ratio = ops.exp(new_log_probs - batch["old_log_probs"])
    adv = batch["advantages"]
    adv = (adv - ops.mean(adv)) / (ops.std(adv) + 1e-8)
    surr1 = ratio * adv
    surr2 = ops.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -ops.mean(ops.minimum(surr1, surr2))

    # Value loss
    value_loss = ops.mean(ops.square(values - batch["returns"]))

    # Entropy bonus (categorical + Gaussian)
    probs = ops.softmax(logits)
    cat_entropy = -ops.mean(ops.sum(probs * log_probs_cat, axis=-1))

    # Gaussian entropy: 0.5 * ln(2*pi*e*std^2) = 0.5 + ln(std) + 0.5*ln(2*pi)
    p0_std = ops.exp(ops.clip(p0_params[:, 1], -5.0, 2.0))
    p1_std = ops.exp(ops.clip(p1_params[:, 1], -5.0, 2.0))
    gauss_entropy = ops.mean(
        0.5 + ops.log(p0_std) + 0.5 * np.log(2 * np.pi)
        + 0.5 + ops.log(p1_std) + 0.5 * np.log(2 * np.pi)
    )
    entropy = cat_entropy + gauss_entropy

    return policy_loss + value_coeff * value_loss - entropy_coeff * entropy


def gaussian_log_prob(x, mean, log_std):
    """Gaussian log-probability. Takes log_std (clipped to [-5, 2])."""
    std = ops.exp(ops.clip(log_std, -5.0, 2.0))
    return -0.5 * ops.square((x - mean) / std) - ops.log(std) - 0.5 * np.log(2 * np.pi)
