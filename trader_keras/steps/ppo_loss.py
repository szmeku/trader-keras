"""PPO loss computation with Beta-distribution continuous actions."""
from __future__ import annotations

import jax.scipy.special
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

    # Beta log probs for p0 ∈ [0,1] and p1 ∈ [-1,1]
    p0_a, p0_b = _beta_params(p0_params)
    p1_a, p1_b = _beta_params(p1_params)

    p0_lp = _beta_log_prob(batch["p0s"], p0_a, p0_b)
    p1_unit = (batch["p1s"] + 1.0) / 2.0  # map [-1,1] → [0,1]
    p1_lp = _beta_log_prob(p1_unit, p1_a, p1_b) - ops.log(2.0)

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

    # Entropy: categorical + Beta
    probs = ops.softmax(logits)
    cat_entropy = -ops.mean(ops.sum(probs * log_probs_cat, axis=-1))
    beta_ent = ops.mean(_beta_entropy(p0_a, p0_b) + _beta_entropy(p1_a, p1_b) + ops.log(2.0))
    entropy = cat_entropy + beta_ent

    return policy_loss + value_coeff * value_loss - entropy_coeff * entropy


def _beta_params(raw):
    """Raw network output → Beta(alpha, beta) with alpha, beta > 1."""
    return ops.softplus(raw[:, 0]) + 1.0, ops.softplus(raw[:, 1]) + 1.0


def _betaln(a, b):
    """Log Beta function: log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
    lgamma = jax.scipy.special.gammaln
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _beta_log_prob(x, alpha, beta):
    """Beta distribution log PDF."""
    x = ops.clip(x, 1e-6, 1.0 - 1e-6)
    return (alpha - 1) * ops.log(x) + (beta - 1) * ops.log(1 - x) - _betaln(alpha, beta)


def _beta_entropy(alpha, beta):
    """Differential entropy of Beta(alpha, beta)."""
    digamma = jax.scipy.special.digamma
    return (
        _betaln(alpha, beta)
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (alpha + beta - 2) * digamma(alpha + beta)
    )
