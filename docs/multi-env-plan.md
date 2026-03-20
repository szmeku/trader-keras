# Multi-Env Parallelization Plan

**Date:** 2026-03-16
**Status:** Phase 1 + 2 complete (MultiTradingEnv built + tested)
**Goal:** Speed up stage2 training/eval by running multiple TradingEnvs in parallel

---

## Problem

Single TradingEnv = CPU bottleneck. Pure Python sim does ~1000 steps/sec. GPU sits at 16% utilization, 2% memory. Rollout collection dominates epoch time.

Current: 2048 rollout steps * 1-5ms/step = 2-10s per epoch, serial.

---

## Resource Utilization Model

The workload has two distinct parts that map to different hardware:

| Work | Where | Why |
|------|-------|-----|
| `env.step()` — TradingSim tick-by-tick sim | **CPU** | Pure Python, no tensor ops, per-env independent |
| Policy inference (`policy(obs_batch)`) | **GPU** | Batched NN forward pass, parallelizes well |
| PPO update (forward + backward + optimizer) | **GPU** | Standard NN training, batched from rollout buffer |
| GAE computation | **CPU** | Simple numpy loop over rewards/values |

### CPU ↔ GPU Pipeline

With N parallel envs, each rollout step looks like:

```
CPU (N cores)              GPU
─────────────              ───
N envs step in parallel
    ↓
N obs collected
    ↓ transfer (N × obs_dim × 4 bytes)
                           policy(obs_batch)  ← batch_size=N, much better util
    ↓ transfer (N actions back)
N envs step with actions
    ... repeat rollout_steps/N times ...

                           PPO update on full buffer ← already batched
```

### N = num CPU cores

`N` should match available CPU cores (minus 1 for the main process coordinating GPU work):

| Machine | Cores | N envs | Rollout batch |
|---------|-------|--------|---------------|
| GTX 1050 Ti desktop | 4-8 | 3-7 | policy sees 3-7 obs per forward |
| VastAI RTX 4090 | 16-32 | 15-31 | policy sees 15-31 obs per forward |

This turns GPU inference from batch_size=1 (current) to batch_size=N. For a 203K-param MixedPolicy this won't saturate the GPU, but it eliminates the "GPU idle waiting for single env step" problem.

### Why This Matters

Current single-env timeline (one rollout step):
```
[env.step 1-5ms][─── GPU idle ───][policy ~0.1ms][─── CPU idle ───]
```

With N envs (pipelined):
```
CPU: [env1.step][env2.step]...[envN.step]  (parallel across cores)
GPU:                                        [policy(N obs)] → actions
CPU: [env1.step][env2.step]...[envN.step]  (next round)
```

CPU and GPU work overlaps. GPU processes a real batch instead of 1 obs. Both resources stay busy.

---

## Constraint

**TradingEnv is the source of truth.** We never bypass it or use internals directly. The multi-env wrapper must be a layer on top — same `env.step()`, same `env.reset()`, same Gymnasium interface. Validated against single-env for identical results.

---

## Two Operating Modes

### Mode 1: Non-continuous (`continuous=False`, default)

Data split into N exclusive date ranges. No overlap. Each env resets to start of its own range when done. No state carryover between envs.

- **Used for:** Training and eval (same flag for both)
- **Consolidation:** None — envs are fully independent
- **On done:** Env resets to start of its range, keeps cycling

### Mode 2: Continuous (`continuous=True`)

Data split into N consecutive date ranges with 1-bar overlap at boundaries. Sequential consolidation step carries state (balance, position, pending, prev_equity) at boundaries.

- **Used for:** Training and eval (same flag for both)
- **Consolidation:** Required — state carryover at each range boundary
- **Equivalence test:** `sum(pnl_multi_env) == pnl_single_env` (within float tolerance)

### AsyncVectorEnv Role (future)

Gymnasium's `AsyncVectorEnv` handles subprocess spawning, IPC, and batched `step()` / `reset()`. Can be used as transport layer for both modes — we own the consolidation logic.

---

## Approach: Date-Range Split + Consolidation (Mode 2)

Split data into N consecutive ranges. Each env gets one range. All run in parallel. Consolidation fixes boundary conflicts as a sequential post-processing step.

```
Full data:  |-------- env1 --------|-------- env2 --------|-------- env3 --------|
Bars:       0                    2666                    5333                   8000
```

### Why This Works

- Each env is a real TradingEnv instance — no API changes
- Parallel execution: all envs step independently
- Only consolidation is sequential (and fast — just boundary checks)

### Critical: Overlap-by-1 at Boundaries

`TradingEnv.reset()` consumes bar[0] for initial observation (no trading). Split envs must **overlap by 1 bar** so the last-traded bar of env_N becomes env_N+1's initial-obs bar:

```
Env1: bars[0 .. S+1)   actions[0 .. S)   trades on bars[1..S]
Env2: bars[S .. E+1)    actions[S .. E)   initial obs from bar[S], trades on bars[S+1..E]
```

Without overlap, action[S] would map to the wrong bar. Proven by tests in `tests/test_multi_env.py` (2/3/5/10-way splits, uneven splits, all pass with `atol=1e-8`).

### Boundary Conflict

When env_N ends with an open position but env_N+1 started flat:

**Detection:** After all envs finish, check if `env_N.final_position != FLAT` at each boundary.

**Resolution — short replay:**
1. Take env_N's final state (position type, volume, entry price, balance, margin)
2. Replay env_N+1's first K bars through TradingEnv with that carried-over state
3. Stitch: use replayed trajectory for bars [boundary, boundary+K], keep env_N+1's original trajectory for remaining bars

**K selection:** Replay until the carried-over position closes naturally (agent action or stop-out). In practice, most positions close within 10-50 bars.

### The Domino Problem

If replay changes early trades in env_N+1, the cascade could invalidate the entire segment. Worst case: all of env_N+1 needs replaying (back to single-env speed).

**Why it's rare in practice:**
- Most positions are short-lived (< 50 bars typically)
- Boundary hitting mid-position requires the position to span exactly the cut point
- With N envs and M positions, probability = `M * avg_hold_duration / total_bars`

**Mitigation options:**
1. **Accept approximation for training** — PPO is already stochastic. Slight trajectory error at boundaries is noise in the gradient estimate. Only need exact consolidation for evaluation/backtest.
2. **Buffer zones** — overlap segments by K bars. Env1: bars 0-2766, env2: bars 2566-5433. Use env1's trajectory for 0-2666, env2's for 2667+. The 100-bar overlap lets env2 warm up with realistic position state.
3. **Replay from boundary** — when conflict detected, fully replay the affected segment with carried-over state. Sequential for that segment but parallel for the rest.

**Recommendation:** For training (Mode 1), use random offsets — no boundary problem. For eval/backtest (Mode 2), use option 3 (exact replay).

---

## Ideas for Later

### Random Offset (from RL community)
Each env gets the full dataset. On reset, picks a random starting bar. No boundary problem at all. Standard approach in FinRL, ElegantRL. Simple but doesn't give exact full-range evaluation. Good for training diversity. Needs shared data optimization (1 copy) to avoid N copies of data in memory.

### Gymnasium AsyncVectorEnv (subprocess parallelism)
N subprocess workers, each with a TradingEnv. Easy to implement (Gymnasium built-in). Gains: N parallel env steps across CPU cores. Cost: IPC overhead, subprocess memory copies. Estimate: 4-8x with 8 workers on typical machine. Can be used in both modes — we own the consolidation step.

### Numba JIT TradingSim (from [[good-ideas-from-legacy-env]])
Rewrite `TradingSim.step()` inner loops in Numba. Single process, no IPC. Legacy benchmarks: 10-50x speedup. Must re-validate MT5 parity (1138 test trades available). Multi-kernel architecture: separate step/build_obs/reset kernels. Auto-dispatch serial vs parallel based on num_envs threshold.

### GPU Env (from legacy benchmarks)
Branchless CUDA env. Only wins at >= 1024 parallel envs. GTX 1050 Ti too tight (4GB VRAM for data + model). RTX 4090: GPU+4096 envs optimal. Highest ceiling but highest effort.

---

## Current Progress

### Done ✓

| Phase | What | Status |
|-------|------|--------|
| 1 | `MultiTradingEnv` wrapper (`icmarkets_env/multi_env.py`) | ✓ Built |
| 2 | Equivalence tests vs single-env (continuous mode) | ✓ 16 tests passing |
| 2 | Real XAUUSD validation (1000 bars, 2/4/8/20-way splits) | ✓ Passing at atol=1e-8 |
| 2 | State carryover at boundaries (balance, position, pending, prev_equity) | ✓ Implemented |
| 3 | `continuous` flag: exclusive ranges (default) vs 1-bar overlap | ✓ Both modes |
| 4 | `num_envs=-1` auto-detect from CPU count | ✓ In config + MultiTradingEnv |
| 5 | Stage2 integration: per-env buffers, batched inference, per-env GAE | ✓ Tested |
| 5 | `Stage2Config.num_envs` / `Stage2Config.continuous` | ✓ Added |

### Next

| Phase | What | Effort |
|-------|------|--------|
| 6 | Fix `test_sim_vs_mt5.py` → test through TradingEnv not TradingSim | 0.5 day |
| 7 | (Later) AsyncVectorEnv for true subprocess parallelism | 1-2 days |
| 8 | (Later) Numba JIT for sim core | 1 week |

### Auto-detection

```python
import os
num_envs = cfg.num_envs  # from config, -1 = auto
if num_envs <= 0:
    num_envs = max(1, os.cpu_count() - 1)  # reserve 1 core for main/GPU
```

`rollout_steps` stays the same total — each env contributes `rollout_steps // num_envs` steps per epoch. Buffer fills N times faster.

---

## Open Questions

- How many envs before IPC overhead dominates? (benchmark at 2, 4, 8, 16)
- Does training signal quality degrade with boundary approximation? (A/B test: single-env vs multi-env training, compare final reward)
- Should consolidation replay use the same policy or a cached policy snapshot?
- Does `AsyncVectorEnv` (subprocess per env) or `SyncVectorEnv` (thread pool) work better for our Python sim? Subprocess avoids GIL but has IPC cost.
