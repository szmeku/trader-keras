# RL Implementation Issues

Found during code review 2026-03-22.

## 1. Missing gradient clipping
- `_update_step` in `steps/rl.py` doesn't clip gradients
- Config has `clip_grad_norm` but it's unused in RL pipeline
- Risk: gradient explosion during training

## 2. Missing Gaussian entropy in PPO loss
- `ppo_loss.py` only computes entropy for categorical action head
- Gaussian entropy for p0/p1 is ignored → less exploration on continuous params
- Standard PPO includes entropy for all action distributions

## 3. Duplicate gaussian log prob
- `rl.py:_gaussian_log_prob` (scalar, line 59) and `ppo_loss.py:_gaussian_log_prob_batch` (batched, line 46)
- Same formula, two implementations → maintenance risk
- Unify into one in `ppo_loss.py`, use from both places

## 4. `_sample_action` repetition
- Gaussian sampling pattern repeated for p0 and p1 (8 lines each)
- Extract `_sample_gaussian(params, clip_lo, clip_hi)` helper

## 5. Loss not returned from `_update_step`
- No way to log per-step loss for debugging
- Return loss value, log in `fit_rl`
