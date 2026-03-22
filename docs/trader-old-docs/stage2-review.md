# Stage 2 / Sim / Env — Deep Review

## Architecture Overview

**Three simulation implementations exist:**

1. **`core.py` — `TradingSim`** (Python, OOP): The "reference" sim. MT5 OHLC 4-tick model, pending orders, stopout, rollover, swap, netting. Used by `tools/validate.py` for MT5 matching.

2. **`numba_sim.py` — `numba_step()`** (Numba JIT): Flat-array reimplementation of TradingSim. State is `float64[12]`, spec is `float64[7]`. Used by `TradingEnv.step()` and `MultiTradingEnv.step_arrays()`.

3. **`env.py` — `TradingEnv`** (Gymnasium wrapper): Calls `numba_step()` under the hood. Single-env Gymnasium interface.

4. **`multi_env.py` — `MultiTradingEnv`**: Wraps N envs. Has two paths: `step_arrays()` (batched Numba) and `step()` (dict-based compat wrapper). Also creates N `TradingEnv` objects for backward-compat (`self.envs`).

**Training flow** (`stage2.py`):
- `stage2_train()` → `MultiTradingEnv` (random_start) → `_collect_rollout()` → `_gae_and_merge()` → `_ppo_update()`
- Policy: `MixedPolicy` (Categorical type + Normal params) → `ActionDecoder` → raw (type, vol, price) → `step_arrays()`
- Oracle: `OracleObsAugmenter` appends future OHLC to obs (agent-side cheat)

---

## Issues

### 1. DUAL SIM: `core.py` TradingSim is dead weight for RL
- [ ] TODO

`TradingSim` (core.py) is ~250 lines of pure-Python sim. `numba_sim.py` is a parallel ~330-line Numba reimplementation with identical logic. **Both must be kept in sync manually.** TradingSim is only used by:
- `tools/validate.py` (MT5 validation)
- `tools/strategies/` (SMA crossover etc.)
- Tests (`test_sim_freeze.py` characterizes TradingSim behavior)

The RL pipeline (env.py, multi_env.py, stage2.py) never touches TradingSim. **Risk: if someone fixes a bug in one, the other diverges silently.**

---

### 2. `MultiTradingEnv` creates N `TradingEnv` objects it mostly doesn't use
- [ ] TODO

Line 115: `self.envs = self._create_envs()` — creates N full `TradingEnv` objects (each with their own Gymnasium spaces, Numba state arrays, etc.). These are only used by:
- `run_sequential()` (eval, split mode)
- `step()` (dict-based compat — syncs state back, line 267-271)

In the training hot path (`step_arrays()`), these `TradingEnv` objects are **never touched**. Wasted memory and complexity.

---

### 3. `RolloutBuffer` (per-env, torch) is dead code
- [ ] TODO

`rollout_buffer.py` has two classes:
- `VectorizedRolloutBuffer` — used by `stage2_train()` (the actual training path)
- `RolloutBuffer` — **never used** by anything in the codebase. Old per-env buffer, superseded. ~100 lines of dead code.

---

### 4. `MultiOracleAugmenter` is dead code
- [ ] TODO

`oracle_agent.py` lines 93-139: `MultiOracleAugmenter` was for the old split-mode multi-env (each env has different data slice). Since training switched to `random_start=True` (all envs share full data), only `OracleObsAugmenter` is used. **~50 lines of dead code.**

---

### 5. `OracleObsAugmenter.augment()` vs `augment_batch()` inconsistency
- [ ] TODO — **BUG (train/eval mismatch)**

`augment()` (single-env, line 73): recomputes normalization from raw every call.
`augment_batch()` (line 83): uses pre-computed `_future_flat`.

So `augment()` gives slightly different numerical results from `augment_batch()` because `augment()` doesn't use the pre-computed cache. The eval path (`evaluate_rollout`) uses `augment()` while training uses `augment_batch()`. **Train/eval mismatch.**

---

### 6. Unused policies, losses, rewards (registries to nowhere)
- [ ] TODO

- `MLPPolicy`, `TransformerPolicy` — never used, incompatible with Dict action space
- `REINFORCELoss` — never used
- `PnLReward`, `RiskAdjustedReward` — never used (rewards come from env directly, no shaping)
- `POLICY_REGISTRY`, `LOSS_REGISTRY`, `REWARD_REGISTRY`, `create_policy()`, `create_loss()`, `create_reward()` — all never called

The `__init__.py` exports all of them. ~120 lines of dead code + import overhead. The docstring acknowledges: "not currently used but kept for potential future use."

---

### 7. `BaseLoss.compute()` signature mismatch
- [ ] TODO

`BaseLoss.compute()` takes `actions: torch.Tensor` (singular tensor).
`PPOLoss.compute()` actually receives `actions: dict` (from MixedPolicy).
The abstract method signature was never updated when MixedPolicy was introduced. `REINFORCELoss` still expects the old `torch.Tensor` actions.

---

### 8. Observation space has 2 wasted dimensions
- [ ] TODO

`numba_obs()` lines 296-297: `obs[19] = 0.0  # dow placeholder`, `obs[20] = 0.0`. Always zero — day-of-week was never implemented. The policy sees two features that are always 0.0.

Same in `core.py` `obs()`: `dow = 0.0  # placeholder`.

---

### 9. `TradingEnv.step()` returns empty info dict
- [ ] TODO

Line 170: `return self._obs(), float(reward), done, False, {}` — always `{}`. The env could return useful info (equity, balance, margin, stopped_out) like `TradingSim.step()` does. Currently `TradingSim.step()` computes this info but `TradingEnv.step()` discards it.

---

### 10. `evaluate_rollout` vs `evaluate_rollout_parallel` divergence
- [ ] TODO

Two eval functions with different behavior:
- `evaluate_rollout()` — single env, sequential, uses `augment()` (no pre-compute)
- `evaluate_rollout_parallel()` — multi env, split mode, uses `augment_batch()`

They also differ in how they collect trades (single env reads `env.last_close_info`, parallel reads `multi._close_infos` directly). **The parallel version produces a stitched result from split envs — each env chunk starts with no position.** Semantic difference from running one env end-to-end.

---

### 11. `_step_idxs` is accessed as internal state all over
- [ ] TODO

`stage2.py` line 62: `step_idxs = multi_env._step_idxs` — training code directly accesses private state of `MultiTradingEnv`. Same in `rollout_eval.py` line 176. Fragile coupling.

---

### 12. `PPO_ENTROPY_COEFF` default inconsistency
- [ ] TODO — **BUG (confusing defaults)**

`constants.py`: `PPO_ENTROPY_COEFF = 0.05`
`Stage2Config`: `entropy_coeff: float = 0.01`

The constant and the config default disagree. `PPOLoss.__init__` defaults to the constant (0.05), but `stage2_train` constructs PPOLoss with `cfg.entropy_coeff` (0.01). If someone creates PPOLoss without specifying `entropy_coeff`, they get 0.05 — 5x the training default.

---

### 14. `step_batch` is serial despite N envs
- [ ] TODO

`numba_sim.py` line 326: `step_batch` uses plain `range(n_envs)` loop, not `prange`. Comment says "prange overhead dominates for N<500" but training can use N=2048 where `prange` would help. Function is `@njit(cache=True)` without `parallel=True`.

Same for `obs_batch` (line 302).

---

### 15. `RL_FEATURE_COLS` and `prepare_features()` are unused
- [ ] TODO

23 feature columns defined in `constants.py` and `prepare_features()` uses them. **Nothing in the codebase calls `prepare_features()`.** Legacy from old envs that used engineered features. Current TradingEnv obs is a fixed 21-float vector from raw state. Dead code.

---

### 16. `min_episode_bars` not enforced in split mode
- [ ] TODO

`MultiTradingEnv.__init__` accepts `min_episode_bars` but only uses it in `_random_start_idx()`. In split mode, chunk size is `n_bars // num_envs` — could be < 64 bars with many envs. The `MIN_EPISODE_BARS=64` cap exists in `stage2_train` but **not in MultiTradingEnv itself**. Direct construction bypasses the cap.

---

### 18. Training metrics not namespaced — confusing in W&B
- [x] DONE

**Two problems (both fixed):**

**A. Training reward inflated by N envs.** `_collect_rollout`: `total_reward += rewards.sum()` summed rewards from all N envs. Changed to `rewards.mean()` — now comparable across runs with different `num_envs`.

**B. Missing `train_` prefix.** Stage 1 uses `stage1/train_loss`, `stage1/val_loss` (flat, descriptive key). Stage 2 now follows the same convention:
- `stage2/train_reward`, `stage2/train_best_reward` — training rollout (per-env average)
- `stage2/train_pg_loss`, `stage2/train_v_loss`, `stage2/train_entropy`, `stage2/train_type_entropy`, `stage2/train_params_entropy` — PPO update stats
- `stage2/eval/*` — post-training eval (unchanged)

---

### 19. Post-training eval silently drops positions at chunk boundaries
- [ ] TODO — **BUG (eval inaccuracy)**

`evaluate_rollout_parallel` splits data into N disjoint chunks (split mode). Each chunk starts with **fresh balance and no position**. If the policy had a position open at the end of chunk k, that position is silently abandoned (not closed, just gone). The stitched equity curve pretends it's one continuous run but:
- Open positions at boundaries are lost (no close P&L recorded)
- Each chunk's starting balance is independent — chunk 1 doesn't inherit chunk 0's balance changes

The stitched result is `balance + cumsum(all_rewards)`, which at least sums the rewards correctly, but the trade list is incomplete (missing trades that span boundaries).

---

### 20. No raw trade log saved from training or eval
- [ ] TODO — **MISSING FEATURE**

Only summary metrics (`pnl`, `sharpe`, `n_trades`) are logged to W&B. No per-trade log (entry/exit times, prices, P&L) is saved anywhere. Cannot inspect individual trades, verify correctness, or do post-hoc analysis without re-running.

---

### 17. `continuous` mode is poorly integrated
- [ ] TODO

`MultiTradingEnv` has a `continuous` flag that adds 1-bar overlap between splits. But `run_sequential()` doesn't handle overlap — it uses `_action_range()` which recalculates splits without considering overlap. Training always uses `random_start=True`, making `continuous` irrelevant. Barely used, undertested.

---

## Simplification Opportunities

| What | Action | Lines saved |
|------|--------|-------------|
| `RolloutBuffer` class | Delete (replaced by `VectorizedRolloutBuffer`) | ~100 |
| `MultiOracleAugmenter` | Delete (superseded by single `OracleObsAugmenter`) | ~50 |
| `MLPPolicy`, `TransformerPolicy`, `BasePolicy` | Delete (incompatible with Dict actions) | ~50 |
| `REINFORCELoss`, `BaseLoss` | Delete (only PPOLoss used) | ~30 |
| `PnLReward`, `RiskAdjustedReward`, `BaseReward`, `rewards.py` | Delete entire file | ~60 |
| All 3 registries + `create_*` funcs | Delete (direct construction used everywhere) | ~30 |
| `RL_FEATURE_COLS`, `prepare_features()` | Delete (obs is 21-float from raw state) | ~15 |
| obs[19:21] dow placeholder | Remove from obs or implement | ~0 (obs_dim 21→19) |
| `MultiTradingEnv.envs` (N TradingEnv objects) | Create lazily, only for `run_sequential` | memory |

**Total: ~335 lines of dead code across rl/ + constants.py**

---

## Test Coverage Map

### Test files related to stage2/sim/env

| File | Lines | What it tests |
|------|-------|---------------|
| `test_sim.py` | ~435 | `TradingSim` (core.py only): bid/ask, tick sequence, position open/close, commission, margin, stopout, swap, obs, rollover, netting, delay_bars, volume clamping, `TradingEnv` reset/step/truncate |
| `test_sim_freeze.py` | ~588 | **Characterization tests** pinning exact numerical values: P&L math, reward=equity delta, info dict fields, obs vector indices, equity tracking across trades, netting edge cases, short stopout, rollover boundaries, swap on short, commission formula, parametric decoder, `from_dataframe` factory. All test `TradingSim`/`TradingEnv` (never numba_sim directly) |
| `test_numba_sim.py` | ~210 | **Numba↔Python equivalence**: runs same bars+actions through both `TradingSim` and `numba_step()`, asserts rewards match to 1e-10. Covers: random actions, all holds, limit buy/sell cycle, stop orders, cancel, netting reversal, rollover, swap, final balance. Also tests on real XAUUSD data |
| `test_sim_vs_mt5.py` | ~285 | **Python sim vs MT5 Strategy Tester**: runs SMA crossover through `TradingSim` (via `run_strategy()`), compares against saved MT5 snapshot. Zero-tolerance: exact trade count, prices, P&L, commission, volume, timing. Only tests `TradingSim`, never numba_sim or TradingEnv |
| `test_multi_env.py` | ~301 | `MultiTradingEnv` split-mode equivalence: single env on full data === multi-env `run_sequential()`. Tests 1/2/3/5/10 envs on synthetic data, 2/4/8/20 envs on real XAUUSD. Also: API tests (reset/step batch shapes), boundary state carryover. All use `continuous=True` |
| `test_stage2.py` | ~276 | `MixedPolicy` shapes/gradients/registry, `PPOLoss` with MixedPolicy, `OracleObsAugmenter` shapes/padding/normalization, `RolloutBuffer` (the dead one!), end-to-end env loop with MixedPolicy + oracle. `Stage2Config` from_dict |
| `test_rollout_eval.py` | ~300 | `evaluate_rollout`, `rollout_metrics`, `stage2_eval` round-trip, `log_rollout_curves`. Tests metric keys, equity tracking, oracle mode, baselines, action distribution, checkpoint save/load |

### Hacks and problems in tests

**1. `test_sim_freeze.py` — swap test calls private API with wrong signature**
- `TestSwap.test_swap_charged_at_hour_23` (line 179): calls `sim._check_swap(bar, bar.close, bar.spread)` — but `TradingSim._check_swap()` only takes `(bar)`, not `(bar, close, spread)`. This test is either **broken** (wrong number of args) or `_check_swap` was modified after the test was written. **Also**, `TestSwapShort` (line 411) calls `sim._check_swap(bar, bar.close, bar.spread)` with same wrong signature. And both reference `sim.swap_rate` which doesn't exist on TradingSim (it has `swap_cost`). These tests either fail or `_check_swap` was changed to accept `(bar, bid, spread)` at some point — **need to verify if these tests actually pass**.

**2. `test_sim_freeze.py` — `TestTradingEnvReward.test_reward_sum_matches_equity` relies on empty info dict**
- Line 491: `final_equity = info.get("equity", 10_000.0)` — since `TradingEnv.step()` always returns `{}`, this always falls back to 10_000.0. The assertion `abs(total_reward - (final_equity - 10_000.0)) < 1.0` becomes `abs(total_reward - 0) < 1.0`, which **only passes because the policy uses `buy={"type":1, "params":[0.5, 0.0]}`** and params[1]=0.0 means price_offset=0 which puts the limit order right at reference price. This means tiny volume → tiny reward → sum stays close to 0 → assertion passes. **This is a hack** — the test doesn't verify equity is correct, it just verifies reward is small. (This is the exact bug documented in status.md "Equity curve always flat at initial_balance".)

**3. `test_stage2.py` — tests the dead `RolloutBuffer`, not the live `VectorizedRolloutBuffer`**
- `TestRolloutBufferMixed` (lines 157-186): imports and tests `RolloutBuffer` (the one never used by the training pipeline). The actual `VectorizedRolloutBuffer` used by `stage2_train` has **zero test coverage** in test_stage2.py.

**4. `test_stage2.py` — oracle test uses `augment()` (single-env), not `augment_batch()`**
- `TestOracleObsAugmenter` only tests `.augment()` and `.obs_dim`. The batched `augment_batch()` (used in actual training) is **untested** in this file.
- `TestEndToEnd.test_oracle_env_loop` uses `oracle.augment(obs, step_idx)` — same single-env path. Never tests the batched path.

**5. `test_multi_env.py` — only tests `continuous=True` split mode, never `random_start=True`**
- All equivalence tests use `MultiTradingEnv(..., continuous=True)`. The actual training always uses `random_start=True`. The `random_start` mode (which is the production code path) has **zero test coverage**.

**6. `test_multi_env.py` — only tests `run_sequential()` and `step()`, never `step_arrays()`**
- The training hot path `step_arrays()` is completely untested. All multi-env tests go through `run_sequential()` (which uses the dict-based `env.step()` per-env loop, not the batched Numba path).

**7. `test_sim_vs_mt5.py` — only tests `TradingSim`, not the Numba sim that RL uses**
- The MT5 validation goes through `tools/strategies/base.py → run_strategy()` which uses `TradingSim.step()`. The Numba sim that TradingEnv actually uses is only validated transitively (numba↔python equivalence → python↔MT5 → numba↔MT5). If the equivalence tests miss an edge case, the Numba sim could diverge from MT5 silently.

**8. `test_rollout_eval.py` — no test for `evaluate_rollout_parallel`**
- Only `evaluate_rollout` (single-env) is tested. `evaluate_rollout_parallel` (the one used in `stage2_train` post-training eval) has **zero test coverage**.

**9. No integration test for the full `stage2_train` function**
- There's no test that runs `stage2_train()` end-to-end (even with 1 epoch on tiny data). The closest is `TestEndToEnd.test_env_step_with_mixed_policy` which manually steps 10 times — doesn't test the training loop, GAE, PPO update, or checkpoint saving.

### Summary: critical test gaps

| Gap | Severity | Related issues |
|-----|----------|---------------|
| `random_start=True` mode untested | **HIGH** — this is the production training path | #2, #16 |
| `step_arrays()` (batched Numba) untested | **HIGH** — this is the training hot path | #14 |
| `VectorizedRolloutBuffer` untested | **HIGH** — production buffer | #3 |
| `augment_batch()` untested | **MEDIUM** — tested transitively via training, but train/eval mismatch (#5) is untested | #5 |
| `evaluate_rollout_parallel` untested | **MEDIUM** — used in post-training eval | #10 |
| No `stage2_train` integration test | **MEDIUM** — would catch config/wiring bugs | all |
| `test_sim_freeze.py` swap tests may have wrong API signature | **MEDIUM** — false pass or broken | #1 |
| `test_sim_freeze.py` equity test relies on empty info dict | **LOW** — known bug, but test passes for wrong reasons | #9 |

---

## Bugs & Pitfalls Found (2026-03-17 debugging session)

### A. Post-swap stopout never fires — **FIXED**

**Bug**: `numba_step()` charges swap at step 4, but `_check_stopout()` only runs at steps 1+3 (before swap). Same in `core.py`. Result: swap drains balance to negative, position survives on unrealized PnL, episode ends via `done = balance <= 0` without recording a trade close. The position silently vanishes.

**Impact**: Agent could hold positions overnight, get massively penalized via balance drain, but the position was never properly liquidated. No trade close recorded → `n_trades=0` in eval even when massive PnL swings happened.

**Fix**: Added `_check_stopout()` immediately after swap deduction in both `numba_sim.py` and `core.py`.

**Tests needed**:
- [ ] **Swap triggers stopout**: Open max-leverage position, advance to hour 23 → swap should trigger stopout → `CI_HAS=1`, position closed, trade recorded
- [ ] **Swap + stopout in both sims**: Same scenario through `TradingSim` and `numba_step()` → identical balance, rewards, close_info
- [ ] **Balance-negative prevention**: After swap + stopout, balance may be negative but position must be closed (not silently abandoned)
- [ ] **Swap without stopout**: Small position (low margin) survives swap → balance drops but equity stays above 50% margin level → no stopout, position survives

### B. Swap was fixed $/lot — not asset-independent — **FIXED**

**Bug**: `SWAP_COST = 50.0` (later 500.0) per lot per night. For BTC at $100K, that's 0.05% of position value. For Gold (100oz contract at $2800), same $500 = 0.18% of position. Wildly different effective rates. At $500/lot for BTC with 20x leverage, a 3-lot position cost only $1,505/night vs $1,800+ daily PnL from 1% BTC move → swap didn't deter holding.

**Fix** (iterated 3 times):
1. First: % of position value → but with 20x leverage, even 5% of position value = 100% of balance → instant death
2. Final: **% of margin** (`swap = rate * margin = rate * pos_value / leverage`). `SWAP_RATE_DEFAULT = 0.50` (50% of margin per night). At max leverage: swap ≈ 48% of balance. Painful, survivable, leverage-independent, asset-independent.

**Tests needed**:
- [ ] **Swap scales with margin**: Same volume at different leverage → swap proportional to margin (inversely proportional to leverage)
- [ ] **Swap is asset-independent**: Gold and BTC at same margin level → same swap fraction of balance
- [ ] **Buy-and-hold is unprofitable**: Open position, hold overnight → net reward is negative even if price moves favorably (swap > unrealized gain from typical daily move)

### C. MT5 test doesn't exercise swap at all

**Root cause**: The SMA strategy closes positions at 23:50 (before swap hour 23:00). The MT5 comparison test never charges swap, so the post-swap stopout bug was invisible.

**Tests needed**:
- [ ] **MT5 swap comparison**: Strategy that holds overnight → compare swap charges between Python sim and MT5 (requires MT5 snapshot with overnight positions)
- [ ] Alternatively: **unit test that swap fires at correct hour** for all sim implementations (already exists but was using wrong API signature — now fixed)

### D. Parallel eval had `trades=[]` — metrics contradicted — **FIXED**

**Bug**: `evaluate_rollout_parallel()` set `trades=[]` because `step_arrays()` doesn't sync state to per-env `TradingEnv` objects. Result: `rollout_metrics()` reported `n_trades=0, total_pnl=0` alongside `pnl=32M` (from equity curve). Contradictory metrics in W&B.

**Fix**: Read `multi._close_infos[:, CI_HAS]` after each `step_arrays()` call, collect trades per-env, stitch in data order.

**Tests needed**:
- [ ] **Parallel eval trade count matches sequential**: Run same data through `evaluate_rollout()` (single env) and `evaluate_rollout_parallel()` (multi env) → trade count, total_pnl, and trade-level details should match (within stitching boundary effects)
- [ ] **Trade data correctness**: Each collected trade dict has valid `side`, `volume`, `entry_price`, `exit_price`, `pnl`, `commission`
- [ ] **No trades lost at chunk boundaries**: Position open at chunk end → should be in last chunk's trades (via stopout/episode-end close) or accounted for

### E. Parallel eval metrics all wrong — **FIXED**

**Bugs** (6 separate issues in `rollout_eval.py`, all related to parallel eval):
1. `n_episodes` defaulted to 1 instead of N → `pnl` was total across all envs, not mean per-env
2. `pnl` computed from `equity_curve[-1]` which was last env's final equity, not representative
3. Sharpe/Sortino computed from equity curve diffs that had huge spikes at env boundaries (equity resets to balance)
4. Max drawdown computed on concatenated curve — artificial drawdowns at boundaries
5. Buy-and-hold and perfect baselines computed on concatenated closes — inflated by price jumps at env boundaries
6. `swap_seen` not reset when env restarts → undercounted `overnight_holds`

**Fix**:
- PnL: `rewards.sum() / n_episodes` (mean per-env)
- Sharpe/Sortino: from `rewards / initial_balance` (no boundary artifacts)
- Max drawdown: per-env then averaged (using `env_lengths`)
- Baselines: per-env then averaged
- Added `env_lengths: list[int]` to `RolloutResult` for segmentation
- Reset `swap_seen[i]` on env restart

**Tests needed**:
- [ ] **Parallel metrics match sequential**: Same data → mean PnL from parallel ≈ PnL from single env (within floating-point)
- [ ] **Baselines per-env**: Verify buy-and-hold and perfect baselines are per-env averages, not cross-boundary
- [ ] **Max drawdown per-env**: Verify no artificial drawdowns at env boundaries

### F. Position survives indefinitely on unrealized PnL

**Not a bug** but a training pitfall: With 20x leverage on trending BTC, a long position generates massive unrealized PnL that keeps equity above stopout level even as balance drains from swap. The agent learns "go long and hold" because reward = equity delta, and equity increases from unrealized gains.

**Tests needed**:
- [ ] **Trending market: position eventually stops out**: Open long on synthetic uptrending data with swap → position must eventually stop out (swap drains balance faster than PnL accumulates, or market reverses)
- [ ] **Reward includes swap penalty**: Verify reward on swap bar is negative (equity drops from swap charge)
