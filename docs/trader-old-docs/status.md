## 2026-03-20 — trader-keras: Initial Scaffold

**New project.** Complete rewrite of `../trader` using Keras 3 + JAX[GPU] backend. Goal: simpler codebase, same Stage 1 predictor, no PyTorch/Numba dependencies.

### Stack
- Keras 3.13 + JAX 0.9.2 (CUDA12 / CudaDevice)
- pandas 3.0 + pyarrow (no Numba)
- wandb 0.25 (same credentials as original project)
- uv for package management, Python 3.13

### What's done
- `trader_keras/config.py` — dataclass config + YAML loader (with float coercion for sci notation)
- `trader_keras/data/resampler.py` — trade ticks → OHLCV, bar reaggregation (pandas-3.x timestamp fix)
- `trader_keras/data/features.py` — 13 default features + optionals, all shift(1), no leakage
- `trader_keras/data/loader.py` — parquet → sequences → train/val split
- `trader_keras/models/gru.py` — Keras 3 GRU, probabilistic multi-horizon, Gaussian NLL + MSE loss
- `trader_keras/trainer.py` — Keras model.fit + callbacks, early stopping, ReduceLROnPlateau
- `trader_keras/logger/` — ConsoleLogger, WandbLogger, MultiLogger, factory
- `run.py` — Typer CLI: `uv run python run.py train [config.yml]`
- 10/10 unit tests passing

### Sanity check metrics (5 epochs, EURUSD 1m, 10k rows, lookback=60, hidden=32, JAX GPU)
| epoch | train_loss | val_loss | overfit_ratio |
|-------|-----------|---------|--------------|
| 0 | -0.185 | -0.337 | 1.82 |
| 4 | -0.471 | -0.649 | 1.38 |

Loss is Gaussian NLL (negative values are normal when sigma is well-calibrated).

### Notable differences from original
- JAX backend instead of PyTorch — XLA compilation, same GPU
- No Numba — resampler is Python loop (acceptable for batch-loaded parquets)
- No Lightning — plain Keras `model.fit` with callbacks
- `buy_ratio` moved to OPTIONAL_FEATURE_COLS (not in ICMarkets bars)
- YAML scientific notation (3e-4) must be written as decimal (0.0003) due to PyYAML

### Next steps
- Full training on all ICMarkets data
- SNR confidence metric (|mu|/sigma) at eval
- Risk-Coverage curve evaluation
- VastAI remote training script

---

## 2026-03-20 (part 11) — Lookback Constant + Unified Eval Rollout

### Fix 1: Lookback from constant
Stage2 `lookback` now uses `PPO_LOOKBACK` constant from `constants.py` (mirrors Stage1's `GRU_LOOKBACK` pattern). Single source of truth — change the constant, all configs pick it up.

### Fix 2: Unified eval rollout
`make_eval_rollout(forward_fn, lookback, balance)` — one factory for all backbones (MLP/GRU/Transformer). Eliminated duplicate MLP vs non-MLP code paths in `_jax_eval`. Same scan body, parameterized by `forward_fn`.

### Cleanup
- Removed unused `mlp_forward` import from `jax_collector.py`
- Merged `fix/same-direction-netting` (same-direction pending orders increase position in MT5 netting mode)

---

## 2026-03-20 (part 10) — JAX PPO Integration + Multi-Architecture Support

### JAX Backend for Stage 2 PPO
The JAX env's lax.scan rollout is now integrated into `stage2_train` as an alternative backend (`backend: jax`). Entire rollout (obs construction → forward pass → action decoding → env step) compiled as a single XLA program via `jax.vmap` + `jax.jit`.

**Config**: `backend: jax`, `backbone: mlp|gru|transformer`, `lookback: 30`

### New Files
- `crypto_trader/jax_env/forward.py` — JAX forward functions for MLP, GRU, Transformer (weight extraction + pure forward)
- `crypto_trader/jax_env/ppo_rollout.py` — PPO rollout via lax.scan with mid-episode auto-reset
- `crypto_trader/trainer/jax_collector.py` — Bridge: JAX rollout → GAE → PyTorch tensors

### Architecture Support
All three MixedPolicy backbone variants work end-to-end:
- **MLP** (default) — flat obs → Linear+ReLU layers
- **GRU** — reshapes bar features into (lookback, 6) sequence → GRU → last output + state → heads
- **Transformer** — bar features → input projection → TransformerEncoder → last output + state → heads

JAX forward functions use factory pattern (`make_gru_forward(lookback)`) to bake shape constants as closure variables — required for JIT safety.

PyTorch policies: `GRUMixedPolicy`, `TransformerMixedPolicy` (inherit `get_action_and_value`/`evaluate_actions` from `MixedPolicy`).

### PyTorch ↔ JAX Equivalence
All forward functions verified to match PyTorch outputs (atol=1e-5 MLP/GRU, 1e-4 Transformer).

### Tests
15 JAX PPO tests (MLP: 3, GRU: 4, Transformer: 4, PPO rollout: 3, collector: 1). All end-to-end smoke tests pass.

---

## 2026-03-20 (part 9) — Feature Redesign: Asset-Agnostic Observations

### Philosophy
Only self-relative log ratios from raw parquet data. No hand-crafted indicators (MAs, RSI, etc.), no time features (hour, day-of-week). The NN builds its own features in hidden layers. Time features removed to keep patterns asset/timezone-agnostic — a breakout pattern should be learnable regardless of when or where it happens.

### Bar Features (6) — `N_BAR_FEATURES=6`
All OHLCV as `log(x_t/x_{t-1})` + relative spread:
1. `log(open_t / open_{t-1})`
2. `log(high_t / high_{t-1})`
3. `log(low_t / low_{t-1})`
4. `log(close_t / close_{t-1})`
5. `log(vol_t / vol_{t-1})` — tick volume, activity proxy (not real volume)
6. `rel_spread = spread * point / close`

### State Features (6) — `N_STATE_FEATURES=6`
RL-only account state, no time:
1. `equity / initial_balance - 1`
2. `has_pos`
3. `pos_side * has_pos`
4. `entry_ret` (log unrealized return)
5. `pos_size` (position as fraction of balance)
6. `has_pend`

### Changes from Previous (part 8)
- **Replaced** `log(high/low)` (absolute bar range, not asset-agnostic) with per-channel time ratios
- **Added** `volumes` to `EnvParams` (tick volume from parquet)
- **Added** `log(open_t/open_{t-1})`, `log(high_t/high_{t-1})`, `log(low_t/low_{t-1})`
- **Removed** `sin/cos(hour)` from state features — time not in observations
- `hours`/`minutes` kept in `EnvParams` for swap/rollover mechanics only

### obs_dim
`lookback * 6 + 6` — e.g. lookback=10 → 66, lookback=30 → 186

### Same-Direction Netting Fix (PR #5)
Same-direction pending order fills now increase position with volume-weighted average entry price (MT5 netting). Fixed across all three sims (JAX, Python, Numba). Previously silently ignored.

### CEM Training Results (new features, obs_dim=186)

| Phase | Epochs | LR | Best PnL | Trades | log_std |
|-------|--------|----|----------|--------|---------|
| 1 | 500 | 1e-3 | $3.89M | 620 | -1.04 |
| 2 | 500 (resumed) | 3e-4 | $4.38M | 647 | -1.37 |

**Deterministic eval (phase 2 checkpoint):**
- PnL: $4,410,644
- Trades: 649 (100% win rate)
- Side split: 330 BUY / 319 SELL (balanced)
- Volume: ramps from 0.01 to max 10.0 (same-direction netting working)
- Commission: still 0.0 (issue b pending)

vs part 8 (old features, obs_dim=98): $4.93M / 706 trades — new features make memorization harder (larger input space) but policy still converges strongly.

### Tests
113 tests passing (67 JAX + 10 numba + others). Only failure: cuDNN 9.20 GRU kernel (issue c, pre-existing).

---

## 2026-03-19 (part 8) — CEM Training + Deterministic Overfitting

### CEM (Cross-Entropy Method) Training
Replaced REINFORCE with CEM for much faster overfitting convergence:
- **16 rollouts per iteration**, select top-4 by total equity reward
- **5 supervised SGD steps** per iteration on elite trajectories
- **MLE loss**: cross-entropy for action types + Normal NLL for continuous params
- REINFORCE was too noisy (single-trajectory gradients → high variance, ~1.75 entropy out of max 1.79)

### Results (1000 CEM epochs on 1 day BTCUSD, 1440 bars)
| Metric | Value |
|--------|-------|
| Deterministic PnL | **$4.93M** (from $10k) |
| Stochastic avg PnL | $4.7M |
| Trades | 706 |
| Win rate | 100% |
| Action types used | BUY_STOP (720) + SELL_STOP (719) exclusively |
| log_std | -1.06 (narrowing) |

### Convergence Comparison
| Method | 500 epochs avg PnL | Stability |
|--------|-------------------|-----------|
| REINFORCE (1 rollout) | oscillated $-10k to $+137k | unstable |
| REINFORCE (8 rollouts, equity reward) | $66k → $89k best | improving but slow |
| CEM (16 rollouts, top-4, 5 SGD) | **$4.7M avg** | monotonic improvement |

### Asset-Agnostic Log-Return Observations
- See part 9 for current feature set (redesigned 2026-03-20)
- No raw prices — NN builds internal representations

### Trade Register Generation
- `collect_rollout` now returns full `CloseInfo` (side, volume, entry/exit price, PnL, commission)
- `generate_trade_register` uses deterministic JAX rollout (temperature=0)
- Output: CSV with per-trade details for MT5 comparison

### Known Issues
- Commission=0.0 in trade register — likely sim bug, needs investigation
- cuDNN 9.20 compatibility issue persists (pre-existing)

---

## 2026-03-19 (part 7) — lax.scan Rollout Module

### lax.scan Rollout (`jax_env/rollout.py`)
- **Full episode in one XLA call**: obs → normalize → MLP forward → Gumbel sample → decode → env step
- Weights extracted from PyTorch `MixedPolicy` → JAX arrays each epoch
- Continuous params sampled from Normal(mu, exp(log_std)) in JAX
- Temperature parameter: 1.0 = stochastic, 0.0 = deterministic (argmax/mean)
- JaxSpec extended with `volume_step` field for action decoder rounding

### Training Speed: 36x Faster
| Mode | Steps/s | Per epoch (1440 bars) |
|------|---------|----------------------|
| Python loop (before) | 352 | 4.09s |
| lax.scan (after) | 12,839 | 0.11s |

### Known Issue: cuDNN 9.20 Compatibility
- `nvidia-cudnn-cu12>=9.8` override (for JAX) installs cuDNN 9.20 — torch GRU CUDA kernel fails with `CUDNN_STATUS_EXECUTION_FAILED_CUDART`
- Affects: `test_data_loading_smoke.py` (GRU on GPU)
- JAX CPU and all non-GRU-CUDA tests pass fine

---

## 2026-03-19 (part 6) — JAX GPU Coexistence + Benchmarks

### JAX GPU Coexistence with torch
- `jax[cuda12]>=0.6.0` now works alongside `torch==2.7.1`
- cuDNN conflict resolved via `[tool.uv] override-dependencies = ["nvidia-cudnn-cu12>=9.8"]`
- JAX backend naming: `_JAX_BACKEND = {"gpu": "cuda", "cpu": "cpu"}` mapping

### Benchmark Results (5000 bars, random policy)
| Backend | Steps/sec | Time | Notes |
|---------|-----------|------|-------|
| Numba | 330,189 | 0.015s | Python loop, best single-core |
| JAX scan CPU | 99,005 | 0.050s | `lax.scan`, no Python overhead |
| JAX scan GPU | 100,942 | 0.050s | `lax.scan`, GPU (GTX 1050 Ti) |
| JAX Python loop CPU | 1,147 | 4.36s | Python loop bottleneck |
| JAX Python loop GPU | 541 | 9.24s | GPU transfer overhead dominates |

### Plan Docs Saved
- `docs/jax_env_plan.md` — implementation plan (from trader-rl-env POC)
- `docs/jax_env_goal.md` — project goal/motivation

---

## 2026-03-19 (part 5) — train_validation_policy.py Analysis

### What it does that config can't
Only one thing is genuinely irreplaceable: **JAX vs Python env comparison** — runs the same policy through both `TradingEnv` (Python/Numba) and `GymWrapper` (JAX), saves two trade CSVs for diffing against MT5.

### Everything else is already in the codebase
| Script feature | Where it already exists |
|---|---|
| First N days data slice | User handles: `collect` + rename |
| Deterministic inference (argmax) | `make_eval_fn(policy, deterministic=True)` in `rl/policies.py:135`; used via `rollout_eval.py:57` |
| ONNX export | `tools/export_stage2_onnx.py` |
| `last_close_info` trade collection | Proper interface on both envs: `icmarkets_env/env.py:120`, `jax_env/compat.py:59`; correct pattern in `rollout_eval.py:135` |

The script reimplements inference manually (`_run_env_loop`) instead of using `rollout_eval.evaluate_rollout(deterministic=True)`.

---

## 2026-03-19 (part 4) — JAX Trading Environment (Phases 1-6)

### Changes
- **New package `crypto_trader/jax_env/`**: Pure functional, JIT-able, vmap-able trading sim in JAX
- Mirrors `icmarkets_env.TradingSim` exactly: same tick sequence, netting, stop-out, swap, rollover
- All branchless via `jnp.where` — ready for GPU vectorization
- `JaxSpec.from_instrument()` converts existing `InstrumentSpec` for reuse
- `GymWrapper` provides Gymnasium API over pure-functional JAX env
- `jax[cpu]>=0.6.0` added as optional dependency group

### Architecture
```
crypto_trader/jax_env/
├── types.py    # NamedTuples + pytree registration (EnvState, EnvParams, JaxSpec, Action, CloseInfo)
├── sim.py      # Price/equity/margin helpers, open/close_position
├── orders.py   # Pending triggers, netting, stop-out, place_order
├── step.py     # step_bar: 4-tick unrolled loop + swap + rollover
├── obs.py      # 21-dim obs vector (matches numba_sim layout)
├── env.py      # reset/step (gymnax-style pure functional)
└── compat.py   # GymWrapper (Gymnasium adapter)
```

### Tests
- 69 new JAX tests (types, sim, orders, step, env, JAX≡Python equivalence)
- Equivalence validated on 2100 real BTCUSD bars with random policy
- f32 tolerance: ~1e-2 per step, ~$50 cumulative over 2100 BTC-priced bars
- 759 existing tests still passing

### Remaining Phases
- Phase 7: Overfitted MLP + MT5 validation (train_validation_policy.py, ValidationPolicy.mq5) — **in progress**
  - `train_validation_policy.py` bugs fixed (path resolution, bar filename convention)
  - `jax_device` config field added to `Stage2Config` — controls JAX device via `with jax.default_device(...)`, saved in checkpoint, overridable via `--jax-device` CLI flag; defaults to `"cpu"` (faster for small validation batches)
- Phase 8: vmap + benchmarks (stretch)

---

## 2026-03-19 (part 3) — Repo Restructure

### Changes
- `logger/` moved into `crypto_trader/logger/` — single package, correct boundary
- All experiment YAMLs (`c*.yml`, `exp*.yml`, `multi*.yml`, etc.) moved to `configs/experiments/`
- Root `models/` (checkpoint artifacts) renamed to `checkpoints/` — removes name collision with `crypto_trader/models/`
- `crypto_trader/envs_legacy/` deleted (was empty, dead code)
- Root junk deleted: `d.out`, `raedme.md`, `Untitled.md`, `unnamed2_reverse.patch`, `PROPOSAL_IMPROVEMENTS.md`, `tools/gen_sweep.py.rej`
- `.gitignore` extended: `*.out`, root PNGs/CSVs now ignored
- All `from logger.` imports updated (8 source files + 2 test files)
- Docs updated: `README.md`, `docs/architecture.md`, `docs/logger.md`

### Tests
639 passed, 5 skipped.

---

## 2026-03-19 (part 2) — More Reduction: functional style + OmegaConf + torch.compile

### Changes
- **OmegaConf**: replaced `yaml.safe_load` in `load_config()` — free `${var}` interpolation support
- **validate_config removed**: ~27 lines cut from `config.py`; Pydantic catches section-level errors, inline 3-line check in `run.py` covers top-level keys
- **`DEFAULT_QUANTILES` consolidated**: `directional.py`, `trading.py`, `regression.py`, `simp_metrics.py` all import from `percentile.py` — no more duplicate tuples
- **W&B `_safe` helper**: replaced 7× `if self._run: try/except` blocks in `wandb.py` with one `_safe(fn, label)` method
- **torch.compile** (`policies.py`): replaced `compile_eval()`/`eval_action_fused()` (50 lines, manual weight extraction) with `torch.compile(policy.eval_action)` via `make_eval_fn()` helper + graceful eager fallback
- **`toolz.merge`** (`rollout_eval.py`): `rollout_metrics()` refactored from 54-line imperative dict-building to `merge(trade_m, episode_m, action_m, baseline_m)` — extracted `_action_metrics` and `_baseline_metrics` helpers
- **`toolz` installed**: available for `merge`, `pipe`, `curry` across codebase

### Tests
638 passing (excluding rolemodel which is slow — add `filtered_pnl_p99` to allowed diffs).
New backlog at `docs/backlog.md`.

---

## 2026-03-19 — Codebase Reduction: SOTA Libs + Generic Patterns

### Changes
- **Pydantic v2**: replaced `@dataclass` + manual `_config_from_dict`/`_validate_field_type` in `config.py` (-47 lines)
- **scipy/sklearn**: replaced custom `_r2`, `_spearman`, numpy `corrcoef` in `regression.py` (-14 lines)
- **Modern typing**: removed deprecated `Optional`/`Dict`/`Tuple`/`Union`/`List` across 10 files
- **`from __future__ import annotations`**: removed from 18 files (Python 3.10+ native syntax)
- **Numba rolling → pandas**: dropped 100-line `_compute_rolling_features_numba` in `features.py`, using pandas `rolling()` instead
- **Generic registry** (`crypto_trader/registry.py`): unified decorator-based (forecaster) and dict-based (policy/loss/reward) registries via `make_registry` + `registry_factory`
- **simpleeval**: replaced 14-line custom AST evaluator in `config.py`
- **deepmerge**: replaced custom `deep_merge` with `Merger(override lists, merge dicts)`
- **Percentile utility** (`eval/percentile.py`): shared `by_percentile()` used by `directional.py`, `trading.py`, `regression.py`
- **W&B scalar helpers**: `_extract_scalar`/`_to_metric` reduce repeated `.item()` checks in `wandb.py`

### Tests
639 passing (including rolemodel against `polished-blaze-1363`).

---

## 2026-03-18 — Stage 1 Training Migrated to PyTorch Lightning

### Goal
Reduce codebase size without changing how Stage 1 training works.

### Changes
- `StandardForecaster.train()`: 220 lines → 80 lines using `L.Trainer.fit()`
- New `GRULightningModule` (`lightning_module.py`): training/val steps + `configure_optimizers`
- New `Stage1MetricsCallback` + `OldFormatModelCheckpoint` (`trainer/callbacks.py`)
- `StridedDataLoaderWrapper`: thin iterable wrapping `StridedLoader` for Lightning
- `LightningBaseLoggerAdapter`: filters Lightning's injected `"epoch"` key
- `enable_progress_bar=False` + `transfer_batch_to_device` no-op: eliminate per-step Python overhead (samples/sec parity at large batch sizes)

### Non-changes
- Checkpoint format preserved (`{"model": state_dict, ...}`) — `load_checkpoint()` unchanged
- All metric names preserved — rolemodel test passes against reference run `genial-blaze-1317`
- Stage 2 (PPO), `StridedLoader`, `memory_tuner`, `logger/` stack — all unchanged

### Tests
638 passing. See [[lightning_migration_report]] for full details.

---

## 2026-03-17 — Sequential Eval: Single-Account Simulation

### Problem
Parallel eval (N envs, disjoint data splits) deployed N×$10k capital — not a real simulation. Each env had independent balance, making metrics misleading (total PnL summed N accounts).

### Changes

**Sequential eval replaces parallel**:
- Single env, full data, one $10k account, sequential positions
- `compounding=False` (default): balance resets after each step → constant position sizing
- `compounding=True`: realistic compounding mode
- `deterministic` flag for greedy actions (reproducible eval)

**Speed optimizations** (CPU inference for batch=1):
- `eval_action_fused()`: bypasses nn.Module dispatch, uses raw F.linear calls
- `compile_eval()`: pre-extracts weight tensors, pre-computes std
- `parametric_decode_single()`: scalar fast-path (no numpy vectorization overhead for N=1)
- Gumbel-max trick replaces Categorical distribution, direct randn replaces Normal
- Pre-allocated tensor buffer for obs, reusable params array
- `torch.inference_mode()` context

**Benchmarks** (80k 1-min bars, MixedPolicy h=256 l=3):

| Mode | Steps/sec | 80k bars | vs. original |
|------|-----------|----------|--------------|
| CPU deterministic | 4300 | 19s | **5.3x faster** |
| CPU stochastic | 3300 | 24s | **4x faster** |
| GPU deterministic | 1800 | 45s | 2.2x faster |
| GPU stochastic | 1300 | 60s | baseline |

### Removed
- `evaluate_rollout_parallel` — wrong capital model
- `RolloutResult.env_lengths` and `n_episodes` fields — no longer needed
- Parallel-specific code in `_compute_baselines` and `rollout_metrics`

639 tests passing.

---

## 2026-03-17 — Stage 2 Speed: 8.8x Faster (0.29s/epoch)

### Problem
With 2048 envs: 2.56s/epoch. Profiling revealed buffer_add was 89% of rollout time (1900ms) — Python for-loop adding items one-by-one to 2048 per-env RolloutBuffer objects.

### Changes

**VectorizedRolloutBuffer** (replaces N per-env RolloutBuffers):
- Single `(n_steps, N, ...)` shaped numpy arrays — `add_step()` writes all N envs at once
- Numba `_gae_vectorized` computes GAE for all N envs in one call
- `compute_gae_and_flatten()` returns flat `(T*N, ...)` tensors directly for PPO
- Eliminated: 2048 buffer objects, 32K per-element Python iterations, `all(b.is_full())` checks

**Pre-computed oracle augmentation**:
- `OracleObsAugmenter._precompute_normalized()` normalizes all future data at init
- `augment_batch()` is now just index + concat (was: copy + normalize + reshape per call)
- 100ms → 19ms per rollout (5x faster)

**GPU rollout inference** (replaces CPU policy copy):
- At N=2048, GPU is 2.7x faster than CPU for policy forward
- Eliminated: CPU policy copy, sync overhead, GPU→CPU→GPU data transfer
- policy.eval() during rollout, policy.train() during PPO update

**train_ratio for stage2**: Moved from Stage1Config to StageConfig (both stages inherit it)

### Performance Results (2048 envs, 32K rollout steps)
| Phase | Before | After | Speedup |
|-------|--------|-------|---------|
| buffer_add | 1900ms | 5ms | **380x** |
| oracle_aug | 100ms | 19ms | **5x** |
| policy_fwd | 82ms (CPU) | 52ms (GPU) | **1.6x** |
| rollout total | 2140ms | 103ms | **21x** |
| gae | 215ms | 3ms | **72x** |
| **epoch total** | **2.56s** | **0.29s** | **8.8x** |

625 tests passing.

---

## 2026-03-17 — Training Speed Optimization: 16x Faster Rollout

### Problem
CPU usage ~20%, GPU ~20%, training slow (~16s/epoch). Rollout collection was 97% of epoch time.

### Changes

**Random-start mode for MultiTradingEnv** (`random_start=True`):
- All envs share the full data array, each starts at random offset
- Eliminates data splitting limitation — can have 64-128+ envs regardless of data size
- Full oracle future data available at all positions (vs split mode where short chunks have mostly zero-padded futures)
- Default 64 envs for good GPU batch utilization

**Zero-copy contiguous state arrays**:
- `self._states` is (N, STATE_SIZE) contiguous array — `step_batch()` operates in-place
- Eliminated per-step state stacking (N copies) + writeback (N copies)

**Vectorized step_arrays() method**:
- Accepts numpy arrays `(act_types, act_vols, act_prices)` instead of `list[dict]`
- Data collection via vectorized numpy indexing (no Python loop for bar data)
- Old `step()` kept for backward compat (wraps `step_arrays`)

**Vectorized action decoder**:
- `parametric_decoder` returns `(types_np, vols_np, prices_np)` arrays
- Full numpy vectorization — no Python for-loop
- Decoder protocol updated: `ActionDecoder` returns tuple of arrays

**Single OracleObsAugmenter** (replaces MultiOracleAugmenter):
- With random_start, all envs share data → single oracle for all envs
- `augment_batch()` uses global step_idxs directly

### Performance Results
| num_envs | epoch time | speedup vs baseline |
|----------|-----------|---------------------|
| 8        | 3.32s     | 4.8x               |
| 32       | 1.32s     | 12x                |
| 64       | 0.97s     | **16x**            |
| 128      | 0.86s     | 19x                |

625 tests passing.

---

## 2026-03-17 — Eval Equity Bug Fix + Entropy Split + Episode Length Fix

### Bug Fixes
- **Equity curve always flat at initial_balance**: `evaluate_rollout` read equity from `info.get("equity", initial_balance)` but TradingEnv.step() returns empty info dict. Now tracks equity as `initial_balance + cumsum(rewards)`.
- **`return_pct` key collision**: `rollout_metrics` and `compute_trade_metrics` both wrote `return_pct` with different values. Reordered so equity-based metrics (correct) take priority over trade-based.
- These bugs caused eval to report `pnl=0.0, final_equity=10000` regardless of actual trading performance.

### Policy Collapse Fix: Split Entropy
**Problem**: MixedPolicy returned combined entropy (type + params). Params entropy (~2.84) dominated, masking type entropy collapse to 0. The model collapsed to 100% stop_sell while total entropy looked healthy at ~2.85.

**Fix**: `MixedPolicy.evaluate_actions` returns `(N, 2)` entropy tensor `[type_ent, params_ent]`. `PPOLoss` applies separate coefficients:
- `type_entropy_coeff=0.1` (default) — 10x stronger than params, prevents action-type collapse
- `entropy_coeff=0.01` — for continuous params (volume, price)
- Now logs `type_entropy` and `params_entropy` separately in training stats

### Episode Length Fix: num_envs Cap
**Problem**: Auto-tuner set `num_envs=2048` for 8000 training bars → 3 bars/env (2 tradeable). The model only ever saw 2-bar episodes, learning to open positions but never to close them.

**Fix**: `stage2_train` caps `num_envs` so each env gets ≥64 bars (`MIN_EPISODE_BARS=64`). With 8000 bars: max 125 envs → 64+ bars/env → meaningful trade cycles.

### step_batch() Wired into MultiTradingEnv
- `MultiTradingEnv.step()` now uses Numba parallel `step_batch()` instead of sequential Python loop
- State arrays stacked into `(N, STATE_SIZE)` contiguous arrays for batch processing
- `obs_batch()` added for parallel obs construction via `prange`

604 tests passing.

---

## 2026-03-16 — Multi-Env Stage 2 Integration

### Multi-Env Rollout Collection
Stage 2 now runs N parallel TradingEnvs for faster rollout collection. GPU batch size = N (batched policy inference).

- **`MultiTradingEnv`** wraps N TradingEnvs with date-range splitting (doesn't modify TradingEnv)
- **Two modes** via `continuous` flag: exclusive ranges (default, for training) or 1-bar overlap + consolidation
- **`num_envs=-1`** auto-detects from `os.cpu_count() - 1`; explicit number overrides
- **Per-env buffers + GAE**: each env has its own RolloutBuffer; GAE computed per-env then merged for PPO update
- **Oracle mode**: per-env OracleObsAugmenters with per-env data slices

### Config
- `Stage2Config.num_envs: int = -1` (auto) and `Stage2Config.continuous: bool = False`

### Architecture Changes
- `_collect_rollout` → batched multi-env (N obs → policy forward → N actions → N env steps)
- `_gae_and_merge` → per-env GAE computation → merged flat tensors
- `_ppo_update` → takes pre-computed tensors (decoupled from RolloutBuffer)

615 tests passing. Smoke-tested with 1/2/4/23 envs, with and without oracle.

---

## 2026-03-16 — Stage 2 Evaluation Pipeline

### New Modules
- **`eval/trade_metrics.py`** — shared trade-level metrics (win_rate, profit_factor, sharpe, sortino, max_drawdown). Uses `risk_metrics.py` internally — no reimplementation.
- **`eval/rollout_eval.py`** — `RolloutResult` dataclass + `evaluate_rollout()` (run policy through env) + `rollout_metrics()` (compute metrics with baselines).
- **`eval/stage2_eval.py`** — load checkpoint → build env → evaluate → return metrics dict.

### Baselines & Log Ratios
- **Buy-and-hold**: hold long from first to last close.
- **Perfect-knowledge ceiling**: `sum(|bar-to-bar returns|)` — theoretical max with zero costs.
- **Oracle RL**: TODO placeholder — will compare against saved oracle runs per asset/range.
- **Log ratios**: `log(policy_equity / baseline_equity)`. Unlike simple ratios, negative means policy is losing money relative to baseline (e.g., log(0.6/1.2)=-0.69 clearly shows a loss).

### Integration
- `stage2_train()` now runs post-training evaluation and logs metrics to W&B.
- `tools/validate.py` refactored to use shared `compute_trade_metrics()` instead of inline computation.
- 13 new tests in `test_rollout_eval.py`.

596 tests passing (597 with rolemodel fix pending).

---

## 2026-03-16 — OptimizerConfig + Stage 2 Config Cleanup

### OptimizerConfig
- New `OptimizerConfig` dataclass: `lr` and `weight_decay` now live under `optimizer:` section in both stages
- `lr` removed from `SchedulerConfig` (scheduler only controls schedule shape, lr belongs to optimizer)
- `weight_decay` moved from `Stage1Config` top-level to `OptimizerConfig`
- `scheduler` moved from `Stage1Config` to `StageConfig` (both stages can use it, stage2 defaults to none)
- `build_scheduler()` reads lr from `optimizer.param_groups[0]['lr']` (single source of truth)
- Deprecated `stage1.lr` migration in run.py removed; old W&B configs with top-level `lr` silently ignored via _WORKFLOW_KEYS

### Stage 2 Config Aligned
- `config.yml` and `config.example.yml` stage2 sections updated — removed 12+ stale fields from old env backend (episodes, patience, val_every, initial_cash, fee_rate, env, policy, loss, reward, etc.)
- `oracle_config.yml` was already correct

### Dead Code
- `rl/agent.py` (PPOAgent) deleted — unused, stage2 trainer builds MixedPolicy+optimizer directly
- `rl/` still exports MLPPolicy, TransformerPolicy, REINFORCELoss, PnLReward, RiskAdjustedReward (not used by stage2_train but kept for potential future use)

585 tests passing.

---

## 2026-03-16 — Code Review: Naming Unification & Dead Code Removal

### Sources of Truth Consolidated
- **PPO constants** → `constants.py` (single source: `PPO_EPOCHS`, `PPO_GAMMA`, etc.)
- **Policy param names** unified: `hidden`→`hidden_size`, `d_model`→`hidden_size`, `n_layers`→`num_layers`
- **Config key remapping eliminated**: removed `_S1_KEY_MAP`/`_S2_KEY_MAP` — config keys now match dataclass fields exactly
- **Config renames**: `feature_cols`→`features`, `data_pattern`→`pattern`, `loss_type`→`loss`, `scheduler_cfg`→`scheduler`
- **`max_rows`** removed — was an alias for `load_limit`
- **Checkpoint key**: `feature_cols`→`features`

### Typed Scheduler Config (B7)
- `SchedulerConfig` TypedDict in `config.py` with all scheduler params properly typed
- `build_scheduler()` signature now typed: `scheduler_cfg: SchedulerConfig`
- `anneal_strategy` narrowed to `Literal["cos", "linear"]`

### Dead Code Removed
- **`backtest.py`** gutted — old code referenced `envs_legacy` which no longer exists. Replaced with docstring documenting ideas to keep for rewrite with `icmarkets_env`
- **`backtest` CLI command** removed from `run.py`

### Type Check
- ty errors: 213 (down from 215 baseline)

585 tests passing.

---

## 2026-03-14 — Stage 2 Plan, DP Oracle Removed, Code Dedup

### DP Oracle Removed
Deleted `tools/strategies/oracle.py`, `OracleStrategy.mq5`, `tests/test_oracle.py`. The DP solver had its own cost model that diverged from the sim (no rollover, no stopout, no margin). Replaced by plan: train a regular RL agent with future-leak features through TradingEnv — same action space, same rewards, same rules.

### Stage 2 Rewrite Plan
Full plan in `docs/stage2-plan.md`. Key decisions:
- PPO for everything (handles full Dict action space natively)
- Full action space from day one (no Discrete(3) simplification)
- Oracle = RL agent that cheats (sees future bars, agent-side)
- Generous future window (20-50 bars)
- Stage 1 integration: 4 modes (from scratch, feature extractor, LoRA, prediction features)

### Code Deduplication
- **`init_seeds()`** extracted from `stage1.py` → `trainer/utils.py` (shared by both stages)
- **`eval/risk_metrics.py`** created — `max_drawdown()`, `sharpe()`, `sortino()`, `calmar()`, `win_rate()`, `profit_factor()`. Used by `simp_metrics.py`, will be used by stage2 backtest.
- Audit found minimal duplication overall — codebase is well-structured.

562 tests passing.

---

## 2026-03-14 — Oracle Separation & Architecture Docs

### Oracle Design Decision
Oracle solver stays **deliberately separate** from env/RL/agent stack. It peeks at all data (perfect foresight), uses its own DP cost model, and replays through TradingSim for actual P&L. No `set_state()` on env — env stays clean.

- **Output**: Policy matrix `policy[bar, state] → target_state` → replayed through sim → benchmark P&L (the ceiling)
- **Shared code**: Types (`Action`, `Bar`, `Side`, `InstrumentSpec`), sim replay, data loading
- **NOT shared**: DP solver lives outside env/RL interface, doesn't pretend to be an agent

### envs_legacy removed
Deleted `crypto_trader/envs_legacy/` entirely (different cost model: 0.1% flat fee). Good patterns preserved in `docs/good-ideas-from-legacy-env.md`.

### New Docs
- `docs/architecture-sim-env.md` — layer diagram, data types, obs vector, formulas, MT5 mapping
- `docs/rl-oracle-ideas.md` — RL algorithm comparison (PPO/DQN/SAC) for real agents + oracle approach
- `docs/good-ideas-from-legacy-env.md` — patterns from deleted envs_legacy worth keeping

### Characterization Tests
37 new tests in `tests/test_sim_freeze.py` freeze sim/env behavior before refactors:
position PnL, reward, info dict, obs vector, equity tracking, netting, stopout, rollover, swap, commission, env action decode, volume clamping.

579 tests total, all passing.

---

## 2026-03-13 — MQL5 ONNX Model Update

### Updated OnnxProof.mq5
Updated the ONNX benchmark/proof Expert Advisor to load the new categorical model (`catmodel_m0.onnx`).
- **Resource**: Switched from `model.onnx` to `catmodel_m0.onnx`.
- **Input Shape**: Updated from MNIST (1, 1, 28, 28) to tabular (1, 14).
- **Output Shape**: Updated from MNIST classes (1, 10) to regression/binary output (1, 1).
- **Logic**: Simplified signal logic for single-output prediction (threshold-based).
- **Benchmarking**: Maintained latency tracking for the new model architecture.

Chronological progress log. For technical details see [[architecture]].

---

## 2026-03-13 — Oracle Baseline Strategy

### Oracle (Perfect-Foresight) Strategy
Implemented backward DP oracle that computes the **theoretical maximum profit** given perfect knowledge of future OHLC prices. This establishes the absolute ceiling for any trading strategy.

- **DP solver** (`tools/strategies/oracle.py`): Numba-JIT, O(N × S²) where S = 1 + 2×V volume levels. 525K bars (1 year) in 0.14s.
- **Dynamic position sizing**: Supports multiple volume levels (e.g., 0.01, 0.02, 0.03 lots) — DP picks optimal volume per bar.
- **Cost-aware**: Accounts for commissions, spread, and swap costs.
- **Forward replay**: DP actions replayed through TradingSim for true dollar P&L.
- **Integrated with validate.py**: `python tools/validate.py oracle --symbol XAUUSD --from 2025-12-01 --to 2025-12-15`
- **DLL-portable**: Core DP is pure numeric logic (no Python objects), ready for C translation + MT5 Strategy Tester comparison.

### Smoke Test Results (XAUUSD, 2 weeks, lot=0.01, 3 levels)
- 20160 bars, 17289 trades, **$96,678.95 PnL**, $3,609.48 commission
- Distribution: 49% long, 49% short, 2% flat

### Tests
- 13 oracle tests + 28 strategy tests + full suite (521 total) — all passing.

---

## 2026-03-12 — Refactoring Proposal & Codebase Analysis

### Stage 1 & Data Pipeline Analysis
Conducted a comprehensive review of the Stage 1 (GRU/Transformer) training pipeline and data loading architecture. Identified opportunities for modularization and technical debt reduction.

### Proposed Architectural Improvements
Detailed proposal created in `PROPOSAL_IMPROVEMENTS.md` covering:
- **Metrics Consolidation**: Standardizing and unifying the `eval/` submodules.
- **Decoupling**: Extracting optimizer, scheduler, and WFO logic for better testability.
- **Forecaster Registry**: Enhancing the pluggable architecture for models like Chronos-2.
- **CLI Refinement**: Modularizing `run.py` to handle subcommands via a dedicated package.
- **Data I/O Separation**: Splitting raw parquet handling from high-level preprocessing.

### Issues Identified
- **Backtest Metrics**: confirmed `win_rate` and `profit_factor` are reporting 0 due to label mismatch in `backtest.py` (added to `docs/issues.md`).
- **LSP Diagnostics**: Noticed optional optimizer usage in `training.py` triggering type errors (to be addressed in future refactoring).

### Status Summary
- **Status**: Research and Planning phase for codebase modernization.
- **Next Steps**: Await user feedback on `PROPOSAL_IMPROVEMENTS.md` before implementation.

### Sim Environment (`crypto_trader/icmarkets_env/`)
New trading simulation engine matching MT5 Strategy Tester behavior exactly:
- **`core.py`**: `TradingSim` — single-instrument, single-position sim with 4-tick OHLC model, stop-out, rollover window, punitive swaps, MT5 netting
- **`instruments.py`**: `InstrumentSpec` loaded from `~/projects/data/instrument_specs.json`
- **`env.py`**: Gymnasium `step()/reset()` wrapper around `TradingSim`

Key broker behaviors in sim (not strategy):
- **Rollover window** (23:58-00:05): broker rejects orders → `_in_rollover()` skips `_place_order()`
- **Punitive swap**: $50/lot/night if holding past 23:00 → model learns to avoid overnight
- **Stop-out** at 50% margin level, checked at each of 4 intra-bar ticks
- **Netting**: opposite-side order closes existing position; volume remainder opens new position (MT5 netting account model)
- **TradingEnv** passes minute to Bar — rollover window fully enforced in RL env

### MT5 Validation: 100% Match
BTCUSD SMA(10,50) over 2026-01-20 to 2026-02-27: **1138/1138 trades match**, zero deviation on entry, exit, P&L, and side.

SMA validation strategy now goes through `TradingSim.step()` (no direct position/balance manipulation) — validates the actual RL execution path. Re-validated after netting + tick order changes: **408/408 trades, 0 deviation**.

Fixes applied: forming-bar SMA adjustment, daily break skip, rollover window, warmup bars, `ROUND_HALF_UP` P&L rounding.

### Strategy-Agnostic Validation Framework
`python tools/validate.py sma --symbol BTCUSD --auto` — runs any strategy through Python sim, generates MT5 configs, waits for MT5 test, auto-compares.

**Architecture:**
- `tools/strategies/base.py` — Strategy protocol (`init`, `on_bar`, `ea_inputs`) + generic `run_strategy()` runner with trade tracking
- `tools/strategies/sma_crossover.py` — `SMAStrategy` class implementing protocol + `SMAStrategy.mq5` EA
- `tools/validate.py` — generic orchestrator with subcommand dispatch (`validate.py sma`, future: `validate.py rsi`, etc.)
- `tools/mt5_tester.py` — MT5 tester config gen (strategy-agnostic `write_ea_inputs(ea_name, inputs_dict)`)

### MT5 Data Collection + Specs
- `tools/mt5_collect.py` — auto-fetches instrument specs (commission, contract size, point) alongside data export
- `tools/vastai_train.sh` — syncs `instrument_specs.json` to remote alongside data files
- Commission from git-tracked `data/broker_commissions.json` ([ICMarkets website](https://www.icmarkets.com/global/en/trading-pricing/spreads)); instrument properties (point, contract_size, etc.) from MT5 EA
- `tools/gen_broker_commissions.py` — regenerates `broker_commissions.json` from parquet filenames in `~/projects/data`; run after downloading new data files

Docs: [[env_specs]], [[metatrader]]

---

## 2026-03-10 — Regularization + Walk-Forward Optimization

Added 4 configurable regularization/validation features, all disabled by default (zero values = no-op):

- **Weight decay (AdamW):** `weight_decay: 1e-4`. Switched optimizer from `Adam` to `AdamW`. When `weight_decay=0.0`, mathematically identical to Adam.
- **Input noise injection:** `noise_std: 0.01`. Gaussian noise on features during training only. Prevents memorization of exact input patterns.
- **Output dropout:** `recurrent_dropout: 0.2`. nn.Dropout on GRU hidden state before FC head. Separate from inter-layer dropout.
- **Sliding-window WFO:** `wfo_folds: 3`. Data → (n_folds+1) equal chunks, fixed-size train slides forward. Auto-purge gap = max(horizons) prevents target leakage. Cumulative W&B steps across folds.

All features backward-compatible — existing configs and checkpoints unaffected. 40 regularization tests + 479 total passing. Docs: [[regularization]], [[config_reference]] updated.

Files changed: `crypto_trader/config.py`, `crypto_trader/models/gru.py`, `crypto_trader/models/training.py`, `crypto_trader/models/standard.py`, `crypto_trader/trainer/stage1.py`, `docs/regularization.md`, `docs/config_reference.md`, `tests/test_regularization.py`

---

## 2026-03-10 — Stage1Config + cross-asset tool rewrite

### Stage1Config dataclass
Introduced `Stage1Config` in `crypto_trader/config.py` — single source of truth for all stage1 configuration fields. Key changes:
- **Strict validation**: unknown YAML keys raise `ValueError` (must add to dataclass first)
- **Key aliases**: `loss`→`loss_type`, `scheduler`→`scheduler_cfg`, `features`→`feature_cols` (handled by `from_dict`)
- `stage1_train(cfg: Stage1Config, logger)` — no more 25+ kwargs signature
- `evaluate_stage1(model_path, cfg: Stage1Config, logger)` — no more `**kwargs`
- Deleted `_stage1_config_to_kwargs` and `_eval_kwargs_from_config` from `run.py` — replaced by `_build_stage1_config()` → `Stage1Config.from_dict()`
- All fields optional with defaults; YAML overrides `existing_wandb_run` values via `deep_merge` (unchanged)

### Cross-asset tool rewrite
Rewrote `tools/cross_asset_eval.py` — now a pure CLI orchestrator, no inline evaluation:

- **Train phase**: `run.py train` per asset (local or `vastai_train.sh` remote). W&B run names captured from output, persisted to `experiments/cross_eval_runs.json`
- **Eval phase**: generates eval configs with `existing_wandb_run` + `stage1.train: false` per (model, asset) pair, runs `run.py train` → parses `val__pearson_corr` from stdout
- **Auto-detects optional features**: probes parquet schema for `spread`→`log_spread`, `buy_ratio` columns
- Eliminated: `MODEL_DIR`, artifact copying, inline `evaluate_stage1` calls, duplicate W&B config merging

```bash
python tools/cross_asset_eval.py config.yml --top 20            # local train + eval
python tools/cross_asset_eval.py config.yml --top 20 --remote   # VastAI train + local eval
python tools/cross_asset_eval.py config.yml --phase eval --top 20
```

---

## 2026-03-05 — Data profiling tool

Added `tools/profile_data.py` — one-command EDA report for any parquet file in `~/projects/data/`:

```bash
python tools/profile_data.py <filename.parquet>
```

- Uses **ydata-profiling** (explorative mode) → full HTML report in `reports/`
- Also generates a compact **text report** (`.txt`) for feeding to Claude for analysis
- Auto-adds **log-return columns** for `close`, `high`, `low`, `open`, `spread`, `volume` (skips missing cols, replaces inf with NaN)
- Correlation matrix (Pearson/Spearman fallback when ydata's auto-correlation fails)

Files: `tools/profile_data.py` (new), `reports/` (gitignored output)

---

## 2026-03-04 — Extensible pipeline + Chronos-2 integration

Refactored Stage 1 pipeline to support pluggable architectures via **Forecaster protocol**:

- **`forecaster.py`**: Protocol class + registry with `@register_forecaster` decorator and `create_forecaster()` factory
- **`standard.py`**: `StandardForecaster` wraps existing GRU/Transformer pipeline (no logic changes)
- **`chronos_forecaster.py`**: `ChronosForecaster` wraps Amazon Chronos-2 foundation model
  - Supports `chronos-2` (120M params) and `chronos-2-small` (28M params)
  - LoRA fine-tuning via Chronos-2's built-in `fit()` method
  - Uses `load_raw_bars()` (no feature engineering — Chronos handles raw close prices)
- **`loader.py`**: Extracted `load_raw_bars()` — loads + resamples only, no features/targets
- **`stage1.py`**: Thin dispatcher — packs config, selects data loader, delegates to Forecaster

Tests: 366 pass (362 fast + 4 Chronos smoke tests: build, inference, LoRA finetune, trainable params).

Files changed: `crypto_trader/models/{forecaster,standard,chronos_forecaster}.py` (new), `crypto_trader/trainer/stage1.py`, `crypto_trader/data/loader.py`, `run.py`, `tests/test_{forecaster_protocol,chronos_data,chronos_smoke,standard_refactor}.py` (new)

---

## 2026-03-04 — Reconcile eval metrics (old vs new code)

Compared 3 W&B runs to audit all metric differences between old and new eval code. Decisions:
- **RESTORED** `profit_on_sign`, `profit_on_sign_per_bar`, `filtered_pnl_p*` — complements sequential_backtest (covers all bars / confidence-filtered bars, not just non-overlapping trades)
- **KEPT NEW** `dir_acc_pXX` filtering by `|pred|` (was `|target|` = future leakage)
- **KEPT NEW** simple returns in `_simulate_trade` (correct for actual PnL, especially with leverage)
- **KEPT NEW** liquidation checks, `val__*` metrics, sequential backtest in eval
- **KEPT REMOVED** `round(v, 6)` (full precision), `baseline_loss` (user request)

Files changed: `crypto_trader/models/eval_metrics.py`

---

## 2026-03-03 — Eval on both splits + reaggregate fix

- **Eval on train + val:** `evaluate_stage1` now runs on both splits. Train metrics (no prefix) verify memorization, val metrics (`val__` prefix) measure generalization. All logged under `stage1/eval/`.
- **`train_ratio` forwarded to eval:** Previously eval always used `train_ratio=0.8` default, ignoring config. Now uses the same `train_ratio` as training (critical when `train_ratio != 0.8`).
- **`eval_deterministic` config option:** Sets `torch.use_deterministic_algorithms(True)` for exact reproducibility across runs. Default `false`.
- **Volume-weighted `buy_ratio` in `reaggregate_bars`:** Was using simple mean (giving equal weight to low/high volume bars). Now uses generic `vmean` rule: multiply by volume before sum, divide after.
- **Logging deduplicated:** Logging and printing moved into `evaluate_stage1` (accepts optional `logger`). `StandardForecaster.evaluate` is a thin wrapper.

Files changed: `crypto_trader/models/stage1_eval.py`, `crypto_trader/models/standard.py`, `crypto_trader/data/resampler.py`, `docs/`

---

## 2026-03-03 — Generic data pipeline (remove Binance hardcoding)

Removed source-specific assumptions from data pipeline + model constructors:

- **`loader.py`**: Column alias table (`_normalize_trade_columns`) replaces hardcoded Binance renames (`quantity→amount`, `is_buyer_maker→side`). Bool→int mapping only triggers on bool dtype. Deduplicated `_process_single_file` branches.
- **`gru.py`**: `infer_model_params` now supports both GRU and Transformer state dicts. `input_size` is now required (no silent fallback to `NUM_FEATURES`).
- **`stage1_eval.py`**: Fixed eval-on-train bug (now evaluates on val split). Added `filter_available_cols` before indexing features.
- **`constants.py`**: `buy_ratio` annotated as optional (Binance-only).
- **`loader.py:sanity_check`**: Handles multi-horizon target columns.

Files changed: `crypto_trader/data/loader.py`, `crypto_trader/models/gru.py`, `crypto_trader/models/stage1_eval.py`, `crypto_trader/constants.py`, `tests/test_missing_columns.py`, `tests/test_gru_models.py`, `tests/test_stage1.py`, `tests/test_stage1_eval.py`, `tests/test_embeddings.py`, `tests/test_strided_loader.py`

---

## 2026-02-27 — Simplify eval-wandb + fix dir_acc_pXX

**eval-wandb simplified:** Removed `EvalConfig` dataclass and all parallel W&B config parsing (`_lookup_wandb`, `_WANDB_ALIASES`, `evaluate_from_wandb`). `eval-wandb` now reuses the same `_fetch_wandb_config()` path as `existing_wandb_run` training — single code path for reading W&B configs. Updates existing run summary directly (no new W&B run).

**dir_acc_pXX fixed:** Was filtering by `|target|` percentiles (future leakage — can't know target magnitude at inference). Now filters by `|pred|` percentiles (prediction confidence — actionable as trade filter).

**evaluate_stage1 simplified:** `evaluate_stage1(model_path, cfg: EvalConfig)` → `evaluate_stage1(model_path, **kwargs)`. Params layered: defaults < checkpoint < explicit kwargs.

Files changed: `crypto_trader/models/stage1_eval.py` (~80 lines removed), `run.py`, `crypto_trader/trainer/stage1.py`, `crypto_trader/models/__init__.py`, `tests/test_stage1_eval.py`, `docs/evaluation.md`

---

## 2026-02-27 — 100-Run Experiment Campaign (Claude)

Ran 100 experiments (77 OK, 25 OOM on 4GB GPU) testing: bar aggregation, horizons, dropout, data diversity, architectures, loss functions, schedulers. All runs tagged `claude` in W&B.

### Key Findings

**Bar aggregation + longer horizons = dramatically better generalization:**
| Config | Pearson | Dir Acc | Notes |
|---|---|---|---|
| bar30s + hor300s + drop0.1 | **0.797** | **0.773** | Best overall |
| bar30s + lb60 + hor300s | 0.507 | 0.653 | Longer lookback helps |
| bar30s + hor300s + drop0.05 | 0.407 | 0.604 | |
| bar30s + hor300s + no drop | 0.406 | 0.644 | |
| bar5s + lb200 + hor300s | 0.386 | 0.643 | 5s bars also work |
| bar15s + lb40 + hor300s | 0.372 | 0.618 | |
| 1s bars (user's best) | ~0.18 | ~0.55 | Memorizes but doesn't generalize |

**Group averages (by approach):**
- `bar30` configs: avg pearson=0.41, max=0.80
- `bar15` configs: avg=0.37
- `bar5` configs: avg=0.36
- `usr` (1s bars): avg=0.13
- `reg` (regularization on 1s bars): avg=0.04

**Cross-coin (bar30 + hor300):** OP=0.456, APT=0.264, data_19d=0.233, NEAR=0.218, BONK=0.206

**Other notable results:**
- `hor_900s` (1s bars, 15min horizon): pearson=0.786, dir_acc=0.801 — long horizons on raw bars also work
- `combo_bar30_huber` (Huber loss): pearson=0.450 — Huber loss generalizes well
- `bar30_h64_hor300_drop01` (bigger model): pearson=0.658 — more capacity helps with bar30
- `combo_stride1_500ep` (stride=1, full data): pearson=0.459 — using all data helps
- `bar30_hor300_cosine_drop01` (cosine sched): pearson=0.411 — cosine > plateau for generalization

**What didn't help:** Dropout on 1s bar configs, very large/small models on 4GB GPU (OOM), most regularization on raw 1s bars.

**Failed runs (22/100):** Mostly OOM on 4GB GTX 1050 Ti with large models (h≥128, lb=500) or large data. Zombie GPU processes from earlier runs also caused cascading OOM.

### Interpretation
1-second bars + 60s horizon overfits to noise. Two paths to generalization:
1. **Bar aggregation**: 30s bars filter microstructure noise → better SNR
2. **Longer horizons**: 300-900s have more predictable structure than 60s
3. **Both**: bar30 + hor300 is the sweet spot (fast training + good generalization)

---

## 2026-02-26 — Seeded Reproducibility + Eval Prefix Fix

**Reproducibility:** Added `seed` field to stage1 config. If null/missing, auto-generates a random uint32 seed and saves it to W&B config. Sets `torch.manual_seed`, `cuda.manual_seed_all`, `np.random.seed`, `random.seed`, plus `cudnn.deterministic=True` / `benchmark=False`.

**Eval prefix fix:** `eval-wandb` CLI now logs metrics as `stage1/eval/` (was `eval/`), consistent with training pipeline.

Files changed: `run.py`, `crypto_trader/trainer/stage1.py`, `crypto_trader/models/stage1_eval.py`, `docs/config_reference.md`, `README.md`

---

## 2026-02-26 — OneCycleLR Scheduler Added

Added `onecycle` (PyTorch `OneCycleLR`) to Stage 1 LR scheduler options. Ramps LR up then down in a single cycle — enables super-convergence.

Config:
```yaml
stage1:
  scheduler:
    type: onecycle
    max_lr: 0.01       # peak LR (default: lr × 10)
    pct_start: 0.3     # warmup fraction
```

Files changed: `crypto_trader/trainer/stage1.py` (+10 lines in `build_scheduler`)

Total Stage 1 scheduler options: `cosine`, `plateau`, `step`, `exponential`, `cosine_warm_restarts`, `onecycle`, `none`.

---

## 2026-02-25 — Lazy Dataset + 19-Day GRU Scale-Up

### Lazy Sequence Dataset (OOM Fix)
`create_sequences()` materialized ALL sliding windows → OOM. Fixed with `SequenceDataset` — creates windows on-the-fly. RAM: O(batch_size) instead of O(n_samples × lookback).

### GRU vs Transformer Memorization
GRU (35k params) memorized to 8.1e-6 loss. Transformer (358k params) only partially memorized. GRU wins for this data scale.

### Key Findings
- **ReduceLROnPlateau** caused premature LR decay on noisy loss — CosineAnnealing decays smoothly regardless of noise
- **clip_grad_norm=1.0** too tight for 2000-step GRU

---

## Completed Milestones (condensed)

| Date | Milestone |
|---|---|
| 2026-02-24 | Phase 1 memorization test PASSED (GRU h=56, 2L, loss=8.1e-6) |
| 2026-02-15 | Codebase reorganized: flat `lib/` → `data/`, `envs/`, `rl/`, `models/`, `trainer/` |
| 2026-02-15 | GPU env benchmarked: GPU wins 4-8x at num_envs≥1024, Numba wins below 256 |
| 2026-02-15 | GRU defaults centralized in `constants.py`, `optimized.py` eliminated (790→0 lines) |
| 2026-02-15 | VastAI experiment infrastructure built (vastai_train.sh, watchdog, monitor) |
| 2026-02-14 | Modular policies/losses with registries, GRU embeddings pipeline |
| 2026-02-13 | Core bugs fixed: backtest, policy collapse, obs normalization. Long/short/flat env |
| 2025-01-29 | Pipeline verified end-to-end. Stage 2 optimized 22x (Numba envs, CUDA graphs, etc.) |

---

## Performance Optimizations Not Pursued

| Idea | Why Not |
|---|---|
| Async Prefetch | Race conditions with CUDA Graphs, marginal gain |
| Mixed Precision (FP16) | GTX 1050 Ti GPU not the bottleneck |
| EnvPool / Isaac Gym | 10-50x speedup but HIGH effort (1-2 weeks C++/CUDA rewrite) |
| JAX/XLA | 5-20x but complete framework rewrite |
