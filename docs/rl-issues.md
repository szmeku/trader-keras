# RL Implementation Issues

Found during code review 2026-03-22. Updated 2026-03-23.

## 1. ~~Missing gradient clipping~~ FIXED
- Gradient clipping now works in `_train_bptt` via `_clip_grads`.

## 2. ~~Missing Gaussian entropy in PPO loss~~ FIXED
- `ppo_loss.py` now computes Beta entropy for p0/p1 alongside categorical.

## 3. ~~Duplicate gaussian log prob~~ FIXED
- Switched to Beta distributions. Single implementation in `ppo_loss.py`,
  separate one in `rollout.py` for JIT rollout (different call signature).

## 4. ~~`_sample_action` repetition~~ FIXED
- Refactored into `_sample_action_jax` in `rollout.py`.

## 5. ~~Loss not returned from `_update_step`~~ FIXED
- `_train_bptt` returns loss, logged in `fit_rl`.
