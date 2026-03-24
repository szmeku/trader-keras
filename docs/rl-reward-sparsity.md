## RL Reward Sparsity Problem

**Date:** 2026-03-23

### Symptom

`rl_small` config: reward stuck at -0.0003 across all 20 epochs with close-only reward. Loss changes (gradients flow), but policy behavior doesn't improve.

### Root cause

Close-only reward requires a two-step pending-order sequence to fire. Credit assignment gap of 10-100+ steps between the critical actions (placing orders) and the reward (position close). GAE can't reliably attribute reward to the right actions.

### Solution: configurable reward functions

Three reward types in `rl.reward.type`:

| Type | Formula | Use case |
|------|---------|----------|
| `close_only` | `realized_pnl/balance * has_close` | Pure trading PnL, very sparse |
| `equity` | `equity_change/balance` | Dense signal, no close incentive |
| `pbrs` | `equity_change/balance + alpha * close_pnl` | PBRS (Ng 1999), proven policy-invariant |

PBRS with `alpha=1` is theoretically optimal (preserves optimal policy). Higher alpha (e.g. 10) learns faster but is asset-dependent.

### Results

| Reward | Epochs to positive reward | Notes |
|--------|--------------------------|-------|
| close_only | never (20 epochs) | stuck at -0.0003 |
| pbrs alpha=1 | ~6 | slower bootstrap, reaches +0.04 by epoch 8 |
| pbrs alpha=10 | ~3 | fastest, reaches +0.018 by epoch 3 |

### Other fixes applied

- **Truncated BPTT** (`tbptt_chunk=256`): 11 gradient updates per epoch instead of 1
- **realized_pnl**: `CloseInfo.realized_pnl` captures commissions + swaps (not just price PnL)
- **Reduced entropy_coeff**: agent was pinned to uniform randomness by entropy bonus
