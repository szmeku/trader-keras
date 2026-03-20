- moze olac speed
	- tylko do wyniku najprostsza droga
	- otestowane porzadnie

# Note
- if you're not sure about the issue, wanna propose some solution before implementing ask under the issue and go to next  one.
- don't fix rolemodel test i'll fix it after
- tackle issues one by one

=wait for overfit we don't need future bars right? so let's avoid it. we can overfit so much that result will be the same as having future bars (at least should be) and will be much
  simpler code right?


# Issues

## 2026-03-19 — Deep Code Audit

### 📏 File Size Violations (CLAUDE.md: 150–200 lines max)

These files exceed the limit and are candidates for splitting:

| File | Lines | Suggestion |
|---|---|---|
| `crypto_trader/trainer/stage2.py` | 411 | Extract `_collect_rollout`, `_gae_and_merge`, `_ppo_update` to `trainer/ppo_loop.py` |
| `crypto_trader/models/chronos_forecaster.py` | 388 | Extract data prep + LoRA config to `models/chronos_data.py` |
| `crypto_trader/icmarkets_env/core.py` | 374 | Consider splitting state machine logic |
| `crypto_trader/icmarkets_env/numba_sim.py` | 336 | Could split step vs obs vs spec sections |
| `crypto_trader/models/standard.py` | 335 | Extract loss building to `models/loss_builder.py` |
| `crypto_trader/icmarkets_env/multi_env.py` | 315 | Could split random_start vs split mode |

### 🧪 Test Coverage Gaps

- [ ] `OracleObsAugmenter.augment()` vs `augment_batch()` consistency — no test verifying they produce the same output at the same step
- [ ] `MultiOracleAugmenter` — no tests (can be deleted)
- [ ] `load_offset` parameter path in `load_single_file` — verify tested in `test_data_loader.py`
- [ ] `eval/evaluation.py` (`coverage_accuracy`, `risk_coverage_curve`) — exists but not wired into training and unclear if tested

### 🏗️ Architecture / Design Debt

- [ ] **`DataConfig` duplication**: `crypto_trader/config.py` has `DataConfig(pattern, load_limit, load_offset)` AND `StageConfig` has the same three fields. `run.py` bridges them via `_inject_data_defaults()`. Consider: either use `DataConfig` directly and remove from `StageConfig`, or eliminate `DataConfig` and use `StageConfig.pattern` as the sole source.

- [ ] **Stage1Config vs StageConfig `leverage` field**: `StageConfig` has `leverage: float = 1.0` inherited by Stage1. Stage 1 never uses leverage. Move `leverage` to `Stage2Config` only.

- [ ] **`REINFORCELoss.compute()` signature mismatch**: `BaseLoss.compute()` declares `actions: torch.Tensor` but `PPOLoss` (and Stage 2 training) passes `actions: dict`. The abstract type is wrong. Minor since `REINFORCELoss` is unused.

- envs tests vs mt5 expand
  - introduce more sma tests for more assets
  - more "strategies"
- [ ] below could be a draft of CLAUDE.md changes or some maintanenace task?
        let's review our code really deeply searching below problems with report and ideas how too solve we'll discuss later.5
        - too many "sources of truth" and if we can reduce them somehow
        - too many names for the same things but in differnt places, causing us to do unecessary mappings

## 2026-03-20 — Reward & Training Design

- [ ] **Reward should be realized PnL, not equity change**
  - Currently CEM ranks rollouts by `eq_sum` (sum of per-bar `equity - prev_equity`). PPO also uses equity-change reward.
  - Equity reward includes unrealized floating P&L — policy gets credit for riding a trend even without closing profitably.
  - For oracle/CEM overfitting, equity and PnL converge to same final value, but the **convergence path** matters for real PPO training — PnL teaches the policy to actually close trades.
  - **Fix**: Rank CEM elites by `close_pnl_sum`. For PPO, switch step reward to realized close PnL (sparse but honest signal).

- [ ] **HOLD bias hack should be removed**
  - `type_head.bias = [-2, 1, 1, 1, 1, -2]` artificially discourages HOLD/CLOSE. With PnL reward the policy should learn when to hold naturally.
  - Remove the bias override, let uniform init + proper reward drive behavior.

- [ ] **`load_limit` takes from start of file, not end**
  - `load_raw_bars('icmarkets_btcusd_*.parquet', load_limit=1440)` loads 2011 data ($0.68 BTC) instead of recent data ($70K). CEM script uses `df.iloc[-N:]` correctly.
  - Either add `load_offset` to config or support negative `load_limit` meaning "last N rows".

## 2026-03-20 — JAX Env Sim Discrepancies

- [x] **Same-direction pending order silently ignored** — FIXED 2026-03-20
  - Fixed across all 3 sims (JAX `orders.py`, Python `core.py`, Numba `numba_sim.py`)
  - Same-direction fills now increase position with volume-weighted average entry price (MT5 netting)
  - CEM training re-run confirms: 649 trades, $4.41M PnL, 100% win rate, balanced BUY/SELL

- [ ] **cuDNN 9.20 breaks torch GRU CUDA kernel** (pre-existing)
  - `nvidia-cudnn-cu12>=9.8` override (for JAX) installs cuDNN 9.20 — torch GRU fails with `CUDNN_STATUS_EXECUTION_FAILED_CUDART`
  - Affects: `test_data_loading_smoke.py`, `test_rolemodel.py`, `test_strided_loader.py` (all use GRU on GPU)
  - **Workaround**: Run JAX tests with `JAX_PLATFORMS=cpu`. Non-GRU tests unaffected.

## 2026-03-16 — Stage 2 Evaluation

- [ ] **Oracle RL baseline**: Store oracle training results keyed by `(symbol, bar_seconds, date_range)`. Load in `rollout_metrics()` for comparison. Currently placeholder (`baseline_oracle_pnl=0`).

## 2026-03-12 — ENVSIMULATION (IC Markets Sim)

- [ ] different mt5 accounts give different data
	- to avoid in the future we should save metadata of collected data from which account downloaded (demo/not and server)
	- also metatrader test strategy should save info about it and then when validate validates first should check accounts infos if match (dome/not and server) if doesn't match should throw errors
	- we shouuld also copy config, results, trades, logs and everything connected to our run sma strategy so we can create test of our environment based on it so we can refactor env/sim/validator etc later without loosing trust in it
- [ ] env documentation should be fully updated
  - [ ] we can change env requirement to env spec (+ some not implemented ideas or bugs if there are some, or quirks)
- [ ] check if our code regarding sim, strategy, validation, using strategy interface etc is documented so we can implement new one
- [ ] on my account i dont see btcusd etc, solve with Kristof
- [ ] more validate options
  - modelling (real ticks), delays, leverage,
- [ ] validator for other assets shows discrepancies between our env/sim or strategy vs mt5 ones
- [ ] check if in our env/sim or around it we for sure don't have future leakage
- [ ] **Sim regression tests from MT5 snapshots**: Use `--save-snapshot` to capture MT5 tester output (trades, config, logs, settings) into `tests/fixtures/mt5_snapshots/`. Write pytest tests that run the Python sim and compare against the snapshot (same as `--compare-snapshot` but automated). This lets us refactor `icmarkets_env/` and `TradingSim` freely — any breakage shows as a diff against the MT5 ground truth. Snapshots needed: SMA on BTCUSD, SMA on XAUUSD, Oracle on XAUUSD (different strategies + symbols to cover edge cases).
- [ ] **Multi-asset/range validation**: Test sim vs MT5 on other symbols (XAUUSD, EURUSD) and date ranges to confirm the match isn't BTCUSD-specific
- [ ] **Strategy-agnostic validation**: Create 2 more test strategies with matching Python + MQL5 implementations to prove env is generic. Ideas:
  - RSI mean-reversion: buy RSI(14) < 30, sell RSI(14) > 70 — tests different indicator math
  - Bollinger breakout: buy close > upper band, sell close < lower band — tests stddev calculation
- [ ] **Real execution engine — rollover safety**: For a future real execution engine (not sim), consider adding a hard 00:00-00:04 order rejection as a safety net. In theory, the model should learn to avoid this window from the punitive swap cost + order rejection in the sim, but a real-money engine may want defense-in-depth
- [ ] **RL agent / NN validation in MT5 via ONNX**: MT5 has **built-in ONNX runtime** (since build 3800+). `OnnxCreate()`, `OnnxRun()` run actual model inference inside Strategy Tester. Pipeline: PyTorch → `torch.onnx.export()` → `.onnx` file → MQL5 EA loads and runs. This validates the full pipeline (model → action) on MT5's own data, not replayed trades. **This is the path for NN strategies.**
- [ ] **Speed for RL**: Current `TradingSim` is pure Python per-tick loop. For RL training with thousands of envs, need Numba/CUDA vectorized version. Existing `crypto_trader/envs/vectorized.py` (Numba) and `gpu.py` exist but use the old env model. Port after validation is solid
