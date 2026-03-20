# Architecture

Full system reference for the crypto RL trading system.

**See also:** [[README]] (install & CLI), [[logger]] (logging API), [[status]] (progress log)

---

## Data Flow

```
Parquet files (tick trades or pre-aggregated bars)
        │
        ├──── load_data() ──── resample + date filter + features + targets (GRU/Transformer)
        │                                     │
        ├──── load_raw_bars() ── resample + date filter, no features (Chronos-2 / Stage 2)
        │                                     │
        ├──────────────────────────────┐      │
        ▼                              ▼      │
  STAGE 1: GRU/Transformer/Chronos  STAGE 2: RL
  ┌──────────────────────┐       ┌──────────────────────────────┐
  │ Input: (L, N_feat)   │       │ icmarkets_env.TradingEnv     │
  │ GRU or Transformer   │       │ obs: 21-dim (TradingSim)     │
  │ → Linear(N_horizons) │       │ or 21 + n_future*5 (oracle)  │
  │ MSE or Gaussian NLL  │       │                              │
  │ → gru_predictor.pt   │       │ MixedPolicy (shared backbone)│
  └──────────────────────┘       │ → type head: Categorical(6)  │
                                 │ → params head: Normal(2)     │
                                 │ → value head: scalar         │
                                 │ PPO only                     │
                                 │ → stage2_YYYYMMDD.pt         │
                                 └──────────────────────────────┘
                                          │
                                          ▼
                                    BACKTEST
                                 ┌──────────────────────┐
                                 │ Needs rewrite for     │
                                 │ icmarkets_env         │
                                 └──────────────────────┘
```

---

## Stage 1: Supervised Predictor

**Files:** `crypto_trader/trainer/stage1.py` (dispatcher), `crypto_trader/models/forecaster.py` (protocol + registry), `crypto_trader/models/standard.py` (GRU/Transformer), `crypto_trader/models/chronos_forecaster.py` (Chronos-2)

Predicts future returns/prices. Uses a **Forecaster protocol** with pluggable implementations:

```python
class Forecaster(Protocol):
    def prepare_data(self, raw_bars, config) -> data: ...
    def build_model(self, config, n_features) -> model: ...
    def train(self, model, data, config, logger) -> path: ...
    def evaluate(self, model_path, config, logger) -> metrics: ...
```

Two implementations:
- **`StandardForecaster`** — wraps GRU/Transformer; training uses PyTorch Lightning (`L.Trainer` + `GRULightningModule`). See [[lightning_migration_report]].
- **`ChronosForecaster`** — wraps Chronos-2 (raw OHLCV → HF Trainer + LoRA)

**Stage 1 training flow (Lightning):**

```
StandardForecaster.train()
  → GRULightningModule(model, loss_fn, config)   # models/lightning_module.py
  → Stage1MetricsCallback                         # trainer/callbacks.py
  → OldFormatModelCheckpoint                      # trainer/callbacks.py
  → EarlyStopping
  → L.Trainer(
        logger=LightningBaseLoggerAdapter(base_logger),  # crypto_trader/logger/lightning_adapter.py
        enable_progress_bar=False,
        gradient_clip_val=clip_grad_norm,
    ).fit(module, StridedDataLoaderWrapper(train_loader))
```

`StridedDataLoaderWrapper` is a thin iterable wrapping `StridedLoader` — batches are already on GPU, so `GRULightningModule.transfer_batch_to_device` is a no-op.

`stage1_train()` dispatches via `create_forecaster(arch)` — CLI unchanged.

### Architecture Options (`stage1.arch`)

**GRU** (default):
```
Input: (batch, lookback, N_features)
  → nn.GRU(N_features, hidden_size, num_layers, dropout)
  → final hidden state → Linear(hidden_size, n_horizons)
Output: predicted returns (or mu+sigma if probabilistic)
```

**Transformer**:
```
Input: (batch, lookback, N_features)
  → nn.Linear(N_features, hidden_size) + positional encoding
  → nn.TransformerEncoder(num_layers, nhead=n_heads, d_ff=hidden_size*4)
  → mean pool over time → Linear(hidden_size, n_horizons)
Output: predicted returns (or mu+sigma if probabilistic)
```
`n_heads` configurable (default: auto = `hidden_size // 16`, min 1). Must divide `hidden_size` evenly.
Memory scales O(lookback²).

**Chronos-2** (`arch: "chronos-2"` or `"chronos-2-small"`):
```
Pretrained 120M-param encoder-only time series foundation model (amazon/chronos-2)
  → Raw close prices as target (Chronos handles scaling internally)
  → Optional covariates (volume, spread, etc.)
  → Fine-tune via LoRA (default) or full fine-tuning
  → HuggingFace Trainer with built-in loss
```
No feature engineering needed — uses `load_raw_bars()` instead of `load_data()`.
Supports: `chronos-2` (120M, amazon/chronos-2), `chronos-2-small` (28M, autogluon/chronos-2-small).

> **Transformer variants — which is good for what:**
> - **Encoder-only** (BERT-style): encodes a fixed-length sequence → prediction. Best for classification/regression tasks like ours.
> - **Decoder-only** (GPT-style): autoregressive generation — each token attends only to past tokens. Best for sequential generation.
> - **Encoder-decoder** (original Transformer, T5): sequence-to-sequence mapping. Best for translation, summarization.
>
> **We use encoder-only everywhere** — `TransformerPredictor` (Stage 1) uses `nn.TransformerEncoder` with no decoder, which is correct for predicting a fixed output from a sequence/feature vector. `TransformerPolicy` in `rl/policies.py` follows the same pattern but is not currently used by Stage 2 training.

### LR Schedulers (`stage1.scheduler.type`)

7 options: `cosine` (default), `plateau`, `step`, `exponential`, `cosine_warm_restarts`, `onecycle`, `none`. See [[config_reference#`scheduler` options]] for parameters.

`onecycle` (OneCycleLR) ramps LR up then down in a single cycle — enables super-convergence with faster training and better generalization.

The learning rate comes from `optimizer.lr` (not `scheduler.lr`). The `scheduler` block only controls the schedule shape (type, patience, step size, etc.). Example:

```yaml
stage1:
  optimizer:
    lr: 0.001
    weight_decay: 1e-4
  scheduler:
    type: cosine
```

### Loss Modes

- **Standard** (`probabilistic: false`): MSE loss
- **Probabilistic** (`probabilistic: true`): Gaussian NLL — model outputs (mu, sigma), trained with negative log-likelihood. Supports `magnitude_alpha` for weighting large moves.

### Target

$$y_t = \frac{p_{t+h}}{p_t} - 1$$

where $h$ is the horizon in bars. Multi-horizon: separate head per horizon.

### Input Features (`FEATURE_COLS`)

Default active features (14 columns, from `constants.py`):
```
high, low, volume, buy_ratio,
log_returns, volatility, volume_ma, volume_ratio,
log_volume, log_volatility, log_volume_ma,
vol_regime, price_accel, time_in_day_counter
```
Plus optional: `log_spread` (present in ICMarkets data).

Many more features are computed by `create_features()` but commented out in `FEATURE_COLS`. Override with `stage1.features` in config to use a custom subset. The full set of computable features includes: `open, close, returns, close_pos, mom_5/10/20, mom_5/10/20_sign, ret_lag1-5, log_ret_lag1-5, dist_ma_5/10/20/50, log_tr`.

All features use `.shift(1)` — no future leakage.

### Stage 1 Metrics

| Metric | Interpretation |
|---|---|
| `stage1/train_loss` | Epoch training loss (MSE or NLL) |
| `stage1/val_loss` | Validation loss — primary quality signal, used for early stopping |
| `stage1/overfit_ratio` | val_loss / train_loss (1.0 = generalizing, >3 = overfitting) |
| ~~`stage1/coverage_acc`~~ | *Not currently computed* — functions exist in `evaluation.py` but are not called |
| ~~`stage1/rc_auc`~~ | *Not currently computed* — functions exist in `evaluation.py` but are not called |
| `stage1/samples_per_sec` | Training throughput |
| `stage1/step_loss` | Per-batch training loss (logged every `log_every` steps) |

---

## Stage 2: RL Policy

**Files:** `crypto_trader/trainer/stage2.py`, `crypto_trader/rl/policies.py` (MixedPolicy), `crypto_trader/rl/losses.py` (PPOLoss), `crypto_trader/rl/oracle_agent.py` (OracleObsAugmenter), `crypto_trader/icmarkets_env/env.py` (TradingEnv), `crypto_trader/icmarkets_env/core.py` (TradingSim)

### Environment

Stage 2 uses `icmarkets_env.TradingEnv` — a single-instrument Gymnasium env wrapping `TradingSim`. It is a single (non-vectorized) env replicating MT5 IC Markets trading rules (pending orders, stop-out, commission, spread, swap via `InstrumentSpec`).

`TradingEnv.from_dataframe(df, symbol, balance, leverage)` builds the env from a parquet-loaded DataFrame.

### Action Space

Dict action space:

| Key | Space | Meaning |
|---|---|---|
| `type` | `Discrete(6)` | 0=HOLD, 1=LIMIT_BUY, 2=LIMIT_SELL, 3=STOP_BUY, 4=STOP_SELL, 5=CANCEL |
| `params` | `Box(2,)` | `[volume_frac (0–1), price_offset_pct (−1 to 1)]` |

`volume_frac`: fraction of max affordable volume. `price_offset_pct`: price offset from current bid/ask as % of price.

`N_ACTION_PARAMS = 2` is defined in `crypto_trader/icmarkets_env/env.py` (single source of truth).

### State Space

`OBS_DIM = 21` — the observation vector from `TradingSim.obs()`. Defined in `crypto_trader/icmarkets_env/env.py`.

In oracle mode: `obs_dim = 21 + n_future * 5` (21 base + n\_future bars of OHLC+spread appended by `OracleObsAugmenter`).

### MixedPolicy

`MixedPolicy(input_dim, n_types=6, n_params=2, hidden_size, num_layers)` — shared MLP backbone with three heads:

```
Input (obs_dim) → [Linear(hidden_size) → ReLU] × num_layers
                        ├─→ type_head:   Linear(hidden_size, n_types)   [Categorical logits]
                        ├─→ params_mu:   Linear(hidden_size, n_params)  [Normal mean]
                        └─→ value_head:  Linear(hidden_size, 1)         [scalar]
```

Combined log_prob = log_prob_type + log_prob_params (compatible with PPOLoss).

`rl/agent.py` (PPOAgent) was deleted — `stage2_train()` builds policy + optimizer directly.

### Oracle Mode

`OracleObsAugmenter` augments the env observation with future OHLC+spread bars (agent-side; the env stays clean). Future OHLC is normalized relative to current close price. Spread is kept raw. Bars beyond the dataset end are zero-padded.

The oracle's converged P&L is the theoretical ceiling for any real agent.

### Loss

PPO only (no REINFORCE option in `stage2_train`):

$$\mathcal{L} = -\min(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\varepsilon)\hat{A}_t) + c_v \mathcal{L}_v - c_e H$$

### Observation Normalization (Welford's Online Algorithm)

$$\hat{s} = \frac{s - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + 10^{-8}}}$$

Updated online from rollout buffer. Saved in checkpoint.

### PPO Hyperparameters

All PPO hyperparameters are centralized in `constants.py`:

| Constant | Value |
|---|---|
| `PPO_EPOCHS` | 4 |
| `PPO_GAMMA` | 0.99 |
| `PPO_GAE_LAMBDA` | 0.95 |
| `PPO_CLIP_EPS` | 0.2 |
| `PPO_ENTROPY_COEFF` | 0.05 |
| `PPO_VALUE_COEFF` | 0.5 |
| `PPO_GRAD_CLIP` | 0.5 |

### Stage 2 Config

```yaml
stage2:
  optimizer:
    lr: 3.0e-4
  # scheduler: null  (no scheduler for PPO by default)
```

### Unused RL Components (kept for future use)

`rl/policies.py` still exports `MLPPolicy`, `TransformerPolicy`, `create_policy`, `POLICY_REGISTRY`. `rl/losses.py` still exports `REINFORCELoss`, `create_loss`, `LOSS_REGISTRY`. `rl/rewards.py` still exports `PnLReward`, `RiskAdjustedReward`, `create_reward`, `REWARD_REGISTRY`. These are not used by `stage2_train()` but are retained for potential future use.

### Stage 2 Metrics

**Training signals:**

| Metric | Good | Bad |
|---|---|---|
| `stage2/train_reward` | increasing | flat or declining |
| `stage2/train_entropy` | 0.3–0.8 | < 0.1 (collapsed) or ~1.0 (random) |

**Not quality signals:** `pg_loss`, `v_loss` (training loss is noisy).

**Post-training evaluation** (`stage2/eval/*`, via `eval/rollout_eval.py`):

| Metric | Description |
|---|---|
| `pnl`, `return_pct` | Total P&L and return percentage |
| `sharpe`, `sortino`, `max_drawdown` | Risk-adjusted returns from equity curve |
| `n_trades`, `win_rate`, `profit_factor` | Trade-level stats |
| `baseline_buyhold_pnl` | Buy-and-hold baseline |
| `baseline_perfect_pnl` | Theoretical ceiling: `sum(\|bar returns\|)` |
| `log_ratio_vs_buyhold` | `log(policy_equity / bh_equity)` — negative = losing to B&H |
| `log_ratio_vs_perfect` | `log(policy_equity / perfect_equity)` — always negative |
| `action_*_pct` | Distribution of action types (hold, limit_buy, etc.) |

---

## Environment Backend

Stage 2 uses a single, non-vectorized `icmarkets_env.TradingEnv`. The old vectorized backends (`numba`/`gpu`/`rust_cpu`) have been removed from the main branch.

The old `VectorizedTradingEnv` (Numba), `TorchTradingEnv` (GPU), and `RustCpuEnv` (Rust/PyO3) have been deleted. Historical benchmark reports: [[bench_env_report]], [[bench_full_report]].

---

## JAX Trading Environment

**Files:** `crypto_trader/jax_env/` — pure functional, JIT-able, vmap-able trading sim in JAX.

Mirrors `icmarkets_env.TradingSim` exactly (same tick sequence, netting, stop-out, swap, rollover), but in float32 JAX arrays for GPU acceleration. Validated against Python TradingSim via `test_jax_vs_python.py` (2100 real BTCUSD bars).

```
crypto_trader/jax_env/
├── types.py      # NamedTuples (EnvState, EnvParams, JaxSpec, Action, CloseInfo) + pytree registration
├── sim.py        # Pure price/equity/margin/commission helpers, open/close_position
├── orders.py     # Pending order triggers, netting, stop-out, place_order — all branchless (jnp.where)
├── step.py       # step_bar: 4-tick unrolled loop, swap, rollover — main sim entry point
├── obs.py        # make_obs: 21-dim observation vector (matches numba_sim.numba_obs)
├── env.py        # reset/step: gymnax-style pure functional interface
└── compat.py     # GymWrapper: Gymnasium adapter (holds JAX state internally)
```

**Design:**
- All state explicit via `EnvState` NamedTuple (no hidden mutation)
- Branchless via `jnp.where` — both branches always execute, condition selects result
- `JaxSpec.from_instrument(spec)` converts existing `InstrumentSpec` for reuse
- f32 throughout — ~1e-2 tolerance vs f64 Python sim, ~$50 over 2100 BTCUSD bars

---

## Fee Structure

Fees are handled by `TradingSim` via `InstrumentSpec` (not a flat `fee_rate` config). `InstrumentSpec` encodes commission (per lot), spread (bid/ask gap in points), and swap (overnight financing). These vary per instrument and are loaded from `crypto_trader/icmarkets_env/instruments.py`.

---

## Backtest

**Status:** Removed (2026-03-16). Old code used `envs_legacy` which no longer exists.

Ideas to keep for rewrite (using `icmarkets_env`):
- Equity curve metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- GRU embedding extraction for feature_mode='gru_embedding'
- Equity vs buy-and-hold dual-panel plot
- Use `MixedPolicy` + `TradingEnv.from_dataframe()`

---

## Checkpoints

**Stage 1** (saved to `artifacts/gru_predictor_YYYYMMDD_HHMMSS.pt`):
```python
{model, n_horizons, probabilistic, horizons, lookback, bar_seconds, y_std, x_mean, x_std, arch, n_heads, features}
```

**Stage 2** (saved to `artifacts/stage2_YYYYMMDD_HHMMSS.pt` or `artifacts/stage2_oracle_YYYYMMDD_HHMMSS.pt`):
```python
{policy, obs_mean, obs_var, obs_count, config}
```

---

## Key Files

| File | Purpose |
|---|---|
| `run.py` | Entry point: `python run.py {train\|squeeze}` |
| `config.yml` | All configuration |
| `crypto_trader/config.py` | Config loading, `Stage1Config` dataclass (single source of truth for stage1 fields) |
| `crypto_trader/constants.py` | Feature cols, GRU defaults |
| `crypto_trader/data/loader.py` | Load + cache parquet data |
| `crypto_trader/data/resampler.py` | Trade → OHLCV bar resampling |
| `crypto_trader/data/features.py` | Feature engineering (Numba-accelerated) |
| `crypto_trader/models/forecaster.py` | Forecaster protocol + registry |
| `crypto_trader/models/standard.py` | StandardForecaster (GRU/Transformer) |
| `crypto_trader/models/chronos_forecaster.py` | ChronosForecaster (Chronos-2 + LoRA) |
| `crypto_trader/models/gru.py` | GRU + Transformer predictors |
| `crypto_trader/models/sequences.py` | SequenceDataset for GRU/Transformer training |
| `crypto_trader/models/training.py` | StandardForecaster training loop |
| `crypto_trader/models/losses.py` | Loss functions (MSE, MAE, Huber, Gaussian NLL, ApexLoss, NormalizedReturnLoss) |
| `crypto_trader/models/embeddings.py` | Pre-compute GRU embeddings |
| `crypto_trader/eval/evaluation.py` | Probabilistic metrics: coverage accuracy, risk-coverage AUC |
| `crypto_trader/icmarkets_env/env.py` | TradingEnv (Gymnasium wrapper, N_ACTION_PARAMS, OBS_DIM) |
| `crypto_trader/icmarkets_env/core.py` | TradingSim — MT5 IC Markets simulation engine |
| `crypto_trader/icmarkets_env/instruments.py` | InstrumentSpec, load_instrument |
| `crypto_trader/jax_env/` | JAX trading env — pure functional, JIT-able, vmap-able (mirrors TradingSim) |
| `crypto_trader/eval/rollout_eval.py` | `evaluate_rollout` + `rollout_metrics` — policy evaluation through env |
| `crypto_trader/eval/stage2_eval.py` | Load stage2 checkpoint → evaluate → metrics |
| `crypto_trader/eval/trade_metrics.py` | Shared trade-level metrics (uses risk_metrics.py) |
| `crypto_trader/rl/policies.py` | MixedPolicy (active), MLPPolicy, TransformerPolicy (unused) |
| `crypto_trader/rl/losses.py` | PPOLoss (active), REINFORCELoss (unused) |
| `crypto_trader/rl/rewards.py` | PnLReward, RiskAdjustedReward (unused by stage2_train) |
| `crypto_trader/rl/rollout_buffer.py` | GAE rollout buffer |
| `crypto_trader/rl/obs_normalizer.py` | Welford online normalizer |
| `crypto_trader/rl/oracle_agent.py` | OracleObsAugmenter — future-bar obs augmentation |
| `crypto_trader/trainer/stage1.py` | Stage 1 training loop |
| `crypto_trader/trainer/stage2.py` | Stage 2 RL training loop (PPO + MixedPolicy) |
| `crypto_trader/trainer/memory_tuner.py` | Auto-tune batch/env params for GPU/RAM |
| `crypto_trader/logger/base.py` | BaseLogger + ConsoleLogger |
| `crypto_trader/logger/wandb.py` | W&B logger |
| `tools/vastai_train.sh` | Remote training on Vast.ai |
| `tools/diff_runs.py` | Compare two W&B runs (diff only) |
| `tools/show_run.py` | Show config + summary for one W&B run |
| `crypto_trader/wandb_utils.py` | Shared W&B utilities (resolve_run, download_model_artifact, etc.) |

## Codebase Size (2026-03-19 — updated)

**Python lines by package:**

| Package | Lines |
|---|---|
| `crypto_trader/models/` | 1,588 |
| `crypto_trader/icmarkets_env/` | 1,553 |
| `crypto_trader/eval/` | 1,174 |
| `crypto_trader/trainer/` | 997 |
| `crypto_trader/rl/` | 731 |
| `crypto_trader/data/` | 601 |
| `crypto_trader/logger/` | 404 |
| `tools/` | ~2,600 (27 files) |
| `tests/` | ~8,500 (34 files) |
| **Total** | **~19,650** |

**Largest files (CLAUDE.md limit: 150–200 lines):**

| File | Lines | Over limit? |
|---|---|---|
| `crypto_trader/trainer/stage2.py` | 411 | ⚠️ yes |
| `crypto_trader/models/chronos_forecaster.py` | 388 | ⚠️ yes |
| `crypto_trader/icmarkets_env/core.py` | 374 | ⚠️ yes |
| `crypto_trader/icmarkets_env/numba_sim.py` | 336 | ⚠️ yes |
| `crypto_trader/models/standard.py` | 335 | ⚠️ yes |
| `crypto_trader/icmarkets_env/multi_env.py` | 315 | ⚠️ yes |
| `run.py` | 447 | ⚠️ yes |
