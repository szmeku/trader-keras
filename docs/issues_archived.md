- [x] Problem with re-downloading specs on each validation, let's remove it. let's base solely on collect
- [x] **Naming**: Renamed `crypto_trader/sim/` → `crypto_trader/icmarkets_env/`
- [x] **MT5 auto mode**: `--auto` requires MT5 to be running (the `/config:` flag applies to the running instance). Added check + user notification in `mt5_tester.py`
- [x] **Execution delay**: Added `delay_bars` parameter to `TradingSim` (default 0). Actions are queued and executed N bars later. For MT5 validation, both sides use delay=0
- [x] **Env/strategy separation audit**: SMA strategy now uses `step()` exclusively. Netting handles close+reverse in one action. Rollover rejection handled by env, not strategy.
- [x] **Auto specs download**: Removed hardcoded fallback — strict error if specs JSON missing. Commission derived from `DEAL_COMMISSION` in EA deal history. `mt5_collect.py` fetches specs by default on every data export. `validate_sma.py` always fetches fresh specs.
- [x] **Validation uses step()**: SMA strategy refactored to go through `TradingSim.step()` — no direct `sim.position`/`sim.balance` manipulation. Netting added to `_check_pending_tick` (MT5 netting semantics). Tick processing reordered so limit orders at open price fill at tick[0]. Validated: 408/408 trades, 0 deviation.

## 2026-03-19 — Deep Code Audit

### 🐛 Bugs / Inconsistencies

- [x] **`volume_ratio` not shifted** (`crypto_trader/data/features.py:77`)
  - `df["volume_ma"]` is shifted by 1 (`np.roll(volume_ma, 1)`) but `df["volume_ratio"]` was NOT shifted.
  - **Fixed**: `volume_ratio` now computed from shifted `volume_ma` (`volumes[i] / mean(volumes[0..i-1])`). Test added: `test_volume_ratio_no_future_leakage`.

- [x] **`OracleObsAugmenter.augment()` (single-env) bypasses precomputed `_future_flat`** (`crypto_trader/rl/oracle_agent.py:72-80`)
  - **Fixed**: `augment()` now uses `self._future_flat[step_idx]` directly (O(1) lookup). `augment_batch()` already did this.
  - **Also fixed**: Added `test_augment_vs_augment_batch_consistency` to verify both paths produce identical output.

- [x] **Hardcoded obs vector indices in `evaluate_rollout`** (`crypto_trader/eval/rollout_eval.py:108-110`)
  - **Fixed**: Added `OBS_BAL=0, OBS_BID=4, OBS_ASK=5` named constants to `env.py`. `rollout_eval.py` now imports and uses them.

- [x] **`normalized_return_loss()` had a sign bug + TODO comments** (`crypto_trader/models/losses.py`)
  - **Bug fixed**: Was returning `capture - 1.0` (minimizing this = minimizing capture = wrong direction). Fixed to `1.0 - capture`.
  - **Clarified**: `normalized_return_loss()` = `apex_loss(lam=0)`. Documented in docstring.
  - **Tests added** in `test_apex_loss.py`: oracle→0, random→1, wrong→2, gradient direction, equals apex_lam0, scale invariant.

### 🧹 Dead Code

- [x] **`MultiOracleAugmenter` is dead code** (`crypto_trader/rl/oracle_agent.py:92-138`)
  - **Fixed**: Deleted (47 lines removed). Stage 2 uses `OracleObsAugmenter` only.

- [x] **Root `logger/` directory not deleted** (project root)
  - **Fixed**: `logger/__pycache__` deleted.

- [x] **Stray docstring in `StageConfig`** (`crypto_trader/config.py:115`)
  - **Fixed**: String expression removed.

- [x] **`from __future__ import annotations` remaining in files**
  - **Fixed**: Removed from `crypto_trader/trainer/stage2.py` and `crypto_trader/eval/rollout_eval.py`.

### ⚡ Performance

- [x] **`OracleObsAugmenter.augment()` performance**
  - **Fixed**: `augment()` now O(1) lookup via `_future_flat`.

### 🧪 Test Coverage

- [x] `normalized_return_loss()` — tests added, sign bug fixed (see bug section)

### 🏗️ Architecture / Design Debt

- [x] increased swap no profit
  - **Confirmed**: swap only fires at hour >= 23, once per day (flag prevents re-charge). No daytime charges.
  - **Root cause**: swap was % of **position value**, but with 20x leverage position value = 20x balance. Even 5% of position value = 100% of balance → instant death, no learning signal.
  - **Fixed**: swap is now % of **margin** (= position_value / leverage). `SWAP_RATE_DEFAULT = 0.50` (50% of margin per night). At max leverage: swap ≈ 48% of balance per night. Painful but survivable — agent can learn the time pattern. Also added `overnight_holds` eval metric.
  - **Also fixed**: post-swap stopout bug (swap could drain balance negative without closing position).

- [x] let's review our whole code if we don't have too many "sources of truth" and if we can reduce them somehow. Investigate deeply and report. With ideas how to reduce it.
  - **Done 2026-03-16**: Centralized PPO hyperparams in constants.py, unified policy param names (hidden_size/num_layers everywhere), removed config key remapping (_S1_KEY_MAP), renamed config fields to match YAML (loss_type→loss, scheduler_cfg→scheduler, feature_cols→features, data_pattern→pattern), removed deprecated max_rows alias, centralized N_ACTION_PARAMS in env.py, renamed r_squared→r2 to match returned key.

## 2026-03-20 — JAX Env Sim Discrepancies

- [x] **Commission = 0.0 in all trades** (trade register from overfitted policy)
  - **Not a bug**: BTCUSD has `commission_per_lot_side = 0.0` (ICMarkets Raw Spread: crypto is commission-free, cost is in spread). Pipeline correctly loads from `broker_commissions.json` → `instrument_specs.json` → `InstrumentSpec` → `JaxSpec`. Verified `compute_commission()` in `sim.py` works correctly — just `0.0 * volume = 0.0`.

## 2026-03-16 — Stage 2 Evaluation

- [x] **Stage 2 post-training eval**: `evaluate_rollout()` + `rollout_metrics()` — runs policy through env, computes PnL, Sharpe, trade stats, baselines, log ratios. Wired into `stage2_train()`.
- [x] **Shared trade metrics**: `eval/trade_metrics.py` reused by stage2 eval and `tools/validate.py`.
- [x] **Stage 2 eval-only mode**: `stage2.train: false` + `stage2.model: path.pt` evaluates without training (same pattern as stage1). Wired in `run.py`.
- [x] **Unified W&B eval for both stages**: `existing_wandb_run` works for stage2 (artifact download + run resume). `_resolve_model_path`/`_resolve_model_paths` are generic (`stage` param). `download_model_artifact` moved to `wandb_utils.py`. Stage2 now logs artifacts via `track_files("models/stage2", ...)`.
- [x] **Parallel eval metrics all wrong** (2026-03-17): 6 bugs in `evaluate_rollout_parallel` + `rollout_metrics` — n_episodes=1 (should be N), PnL from wrong equity endpoint, Sharpe/Sortino/drawdown/baselines had boundary artifacts from concatenated per-env data. All fixed: PnL from rewards, per-env metrics, `env_lengths` field in `RolloutResult`.

## 2026-03-12 — ENVSIMULATION (IC Markets Sim)

- [x] not enough vram
- [x] IN_PROGRESS wandb git add -A error
- [x] IN_PROGRESS dp exporting just state/"weights" not reimplementation in mq5
- [x] **Backtest Metrics Bug**: Moot — `backtest.py` gutted (2026-03-16), needs full rewrite for `icmarkets_env`
- [x] out of vram — see autotuner fix above
- [x] now small discrepencies python vs mt55
- [x] somethings wrong with autotuner
	- when we run a lot of data like in exp_long.yml, we're getting not enough vram error
	- **Root cause**: Two problems: (1) 95% safety margin was way too thin — CUDA caching allocator reserves 40-47% overhead beyond peak allocated, leaving <5% headroom; (2) probe ran each attempt from clean `empty_cache()` state, not matching real training's steady-state allocation pattern.
	- **Fixed** (3 files):
	  - `memory_tuner.py`: Lowered safety margin 95% → 75%; probe now runs 3 consecutive iterations per candidate (steady-state test, not clean-cache); added `reserve_bytes` for StridedLoader superbatch chunks
	  - `strided_loader.py`: Added `pending_vram_bytes()` — returns 0 for single mode (data already on GPU), max chunk bytes for multi mode
	  - `standard.py`: Passes `reserve_bytes` to probe; added OOM recovery (catch `OutOfMemoryError`, halve batch_size, retry epoch — up to 3 retries)
	- Verified locally: 4-stock training with auto batch_size runs with 25% headroom (was 5%)
- [x] no "commission_per_lot_side" for all ~/projects/data/instrument_specs.json specs
  - **Fixed**: Commission now comes from git-tracked `data/broker_commissions.json` (ICMarkets website). `fetch_specs()` was also commented out — uncommented. EA deal history commission code removed.
- [x] **Strategy-agnostic validation**: Refactor `validate_sma.py` → `validate.py` with pluggable Strategy protocol. Python strategy = abstraction that works for fixed scripts (SMA) or NN models. `validate.py sma --symbol BTCUSD --auto` dispatches to SMAStrategy class + SMAStrategy.mq5
  - **Done**: `tools/strategies/base.py` (Strategy protocol + `run_strategy()` runner), `tools/strategies/sma_crossover.py` (SMAStrategy class), `tools/validate.py` (generic CLI), `tools/mt5_tester.py` (generalized `write_ea_inputs`/`run_strategy_test`). Verified 1138/1138 trades identical output.
