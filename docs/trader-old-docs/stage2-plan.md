# Stage 2 RL Pipeline — Rewrite Plan

**Date:** 2026-03-14
**Status:** Planning

---

## Goal

Two pipelines, one codebase:
- **Stage 1** = supervised training/eval for predictive models (GRU, Transformer, etc.)
- **Stage 2** = RL training through `TradingEnv` (Gymnasium)

Stage 2 can use Stage 1 models in 4 ways:
1. **From scratch** — no Stage 1 model, agent learns from raw env obs
2. **Feature extractor** — freeze Stage 1 GRU, use hidden states as extra obs features
3. **Fine-tune** — LoRA or unfreeze last layers, train end-to-end with RL loss
4. **Prediction features** — Stage 1 as black box, feed (μ, σ, direction) as extra obs

---

## Current State

### Reusable (keep as-is)

| Module | Lines | Notes |
|--------|-------|-------|
| `rl/agent.py` | 129 | PPO agent, pluggable policy/loss, device-agnostic |
| `rl/policies.py` | 83 | MLP + Transformer actor-critic, registry |
| `rl/losses.py` | 78 | PPO + REINFORCE |
| `rl/rewards.py` | 61 | PnL + risk-adjusted shaping |
| `rl/rollout_buffer.py` | 88 | GAE, pre-allocated GPU tensors |
| `rl/obs_normalizer.py` | 33 | Welford's running stats |
| `icmarkets_env/core.py` | 362 | TradingSim — MT5-fidelity engine |
| `icmarkets_env/env.py` | 138 | Gymnasium wrapper (single env) |

### Shared with Stage 1

| Resource | Location | Used by |
|----------|----------|---------|
| Data loading | `data/loader.py` | Both stages |
| Instrument specs | `icmarkets_env/instruments.py` | Stage 2 + tools |
| Types (Action, Bar, etc.) | `icmarkets_env/core.py` | Stage 2 + tools |
| Config system | `config.py` | Both stages |
| Seed init | `trainer/utils.py` (`init_seeds`) | Both stages |
| W&B logging | `logger/` | Both stages |
| Model definitions | `models/gru.py` | Stage 2 imports for feature extraction |
| Risk metrics | `eval/simp_metrics.py` (`_risk_metrics`) | Stage 1 eval; stage 2 backtest will reuse |

### Broken (needs rewrite)

| Module | Problem |
|--------|---------|
| `trainer/stage2.py` | References deleted `envs_legacy` |
| `backtest.py` | Same |

---

## Architecture

```
Stage 1 (supervised)              Stage 2 (RL)
═══════════════════              ═════════════

GRU/Transformer model    ──────► Feature extractor (frozen/LoRA/predictions)
trained on price data            ↓
outputs: predictions,            Agent (policy network)
hidden states                    ↓
                                 TradingEnv (Gymnasium)
                                 ↓
                                 TradingSim (MT5 rules)
```

### Pipeline Flow
```
1. Load data (parquet → DataFrame)
2. Create TradingEnv from DataFrame
3. Optionally load Stage 1 model for feature extraction
4. Create RL agent (policy + loss + reward shaper)
5. Training loop:
   - Collect rollouts: obs → agent.act() → env.step()
   - Compute GAE (rollout_buffer)
   - PPO update (agent.update())
   - Validate periodically
6. Save checkpoint
```

---

## Design Decisions

### Action Space
Use the full env action space from day one: `Dict(type=Discrete(6), params=Box(2,))`.
- 6 order types: HOLD, LIMIT_BUY, LIMIT_SELL, STOP_BUY, STOP_SELL, CANCEL
- 2 continuous params: volume_frac (0-1), price_offset_pct (-1 to 1)
- PPO handles mixed discrete+continuous natively — no simplification needed
- Oracle with future knowledge should learn the full space easily

### Observation Space
Env provides 21 floats (account + market + position + pending + time).

With Stage 1 features, agent sees: `21 + N` where N depends on integration mode:
- Raw predictions: N = num_horizons (e.g., 5)
- Hidden states: N = hidden_dim (e.g., 64-128)
- Both: N = num_horizons + hidden_dim

Feature augmentation happens **agent-side** — env always returns the same 21-dim obs.

### Vectorization
TradingSim is Python — single env is slow for millions of steps.

**Start with single env** (DQN is sample-efficient). Scale later:
1. Gymnasium `AsyncVectorEnv` (easy, N processes)
2. Numba-JIT sim core (10-100x, must maintain MT5 parity)

---

## Oracle Agent (first milestone)

**What:** Regular RL agent that cheats — peeks at future bars agent-side.

**Why:** Validates the entire pipeline end-to-end. If oracle can't profit, something's wrong with env/reward/action space.

**How:**
1. Oracle agent gets full dataset at construction
2. Each step: receives obs from env + looks up next N bars internally
3. Concatenates [obs, future_bars] → feeds to policy network
4. Train DQN/PPO until convergence
5. Result = benchmark P&L (theoretical ceiling)

**Metric:** Real agents measured as `agent_pnl / oracle_pnl` (% of ceiling).

---

## Implementation Order

| Step | What | Depends on |
|------|------|------------|
| 1 | Oracle agent (future-leak through env) | Nothing |
| 2 | Training loop (single env, PPO, full action space) | Step 1 |
| 3 | Train oracle → validate benchmark P&L | Step 2 |
| 4 | Stage 1 integration (GRU features → obs) | Step 2 |
| 5 | Backtest/eval rewrite (TradingEnv) | Step 2 |
| 6 | Vectorization (when speed matters) | Step 2 |

---

## Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| Create | `crypto_trader/rl/oracle_agent.py` | Future-leak agent for benchmark |
| Rewrite | `crypto_trader/trainer/stage2.py` | New training loop |
| Rewrite | `crypto_trader/backtest.py` | Use TradingEnv |
| Modify | `crypto_trader/rl/policies.py` | Policy for augmented obs (env + features) |
| Extract | `crypto_trader/eval/risk_metrics.py` | Sharpe/Sortino/Calmar shared functions |
| Keep | All other `rl/` modules | Reuse as-is |

---

## Decisions Made

1. **PPO for everything** — handles full Dict action space natively. DQN would require discretization.
2. **Future bars for oracle** — generous window (e.g., 20-50 bars). More = stronger signal, obs size is trivial.
3. **Risk metrics extraction** — do now (shared by stage1 eval + stage2 backtest).
4. **Full action space from day one** — no Discrete(3) simplification. PPO handles it.
