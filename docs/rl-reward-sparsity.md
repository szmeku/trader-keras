## RL Reward Sparsity Problem

**Date:** 2026-03-23

### Symptom

`rl_small` config: reward stuck at -0.0003 across all 20 epochs. Loss changes (gradients flow), but policy behavior doesn't improve.

### Root cause

Close-only reward (`rollout.py:102`) requires a two-step sequence to fire:
1. Place pending order (limit/stop) → market triggers it → position opens
2. Place opposite-side pending order → market triggers it → netting closes position → reward

A random/early policy almost never completes this in 2868 bars. ~99%+ of rewards are exactly 0, so GAE advantages are near-zero and gradients carry no useful signal.

### Constraints

- Only 1 pending order slot in the env (new order replaces old)
- No market orders — all 6 action types are: HOLD, LIMIT_BUY, LIMIT_SELL, STOP_BUY, STOP_SELL, CANCEL
- Limit prices = `ref_price * (1 + p1 * 0.01)` — order may never trigger if price doesn't reach level

### Attempted fixes

| # | Change | Result |
|---|--------|--------|
| 1 | Increase `entropy_coeff` (0.01 → ?) | pending |

### Possible future approaches

- **Reward shaping:** small bonus for having a position open (risk: hold-farming)
- **Curriculum:** start with dense equity-change reward, anneal to close-only
- **Env change:** add market order / close-position action type
