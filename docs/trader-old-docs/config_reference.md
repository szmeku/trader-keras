# Config Reference

Complete reference for all `config.yml` options.

**See also:** [[architecture]] (system design), [[README]] (install & CLI)

---

## CURRENT PHASE: STAGE 1 ONLY

> **Stage 2 (RL) and validation are DISABLED for now.**
> All experiments use `stage2.skip: true` and `train_ratio: 1.0` (memorization-first).
> We will address Stage 2 and val/generalization AFTER finding promising Stage 1 configs.

---

## Config Sections

```yaml
existing_wandb_run: "run-name"  # Optional: reproduce a W&B run (other keys = overrides)
stage1:    # Stage 1 supervised predictor
stage2:    # Stage 2 RL policy
data:      # Data loading
autotuner: # GPU memory auto-tuning
logging:   # Experiment tracking
```

### Reproducing a W&B Run

Set `existing_wandb_run` to a W&B run name/ID. The full config is fetched from W&B, then any other keys in the file are deep-merged as overrides. The probed batch size from the original run is used for exact reproduction (override with `batch_size: -1` to re-probe).

```yaml
existing_wandb_run: lilac-pine-547

stage1:
  lookback: 800
  stride: 800
```

---

## `stage1` — Supervised Predictor

> **Type:** All `stage1` fields are defined in `Stage1Config` (`crypto_trader/config.py`). Adding a new YAML field requires adding it to the dataclass first — unknown keys raise `ValueError`. Config keys match dataclass fields 1:1.

> **Units:** `lookback`, `stride`, `horizons` are all **in bars**. Only `bar_seconds` is in seconds. YAML values support arithmetic expressions (e.g. `horizons: 60*5` → 300).

| Field | Type | Default | Valid Values | Description |
|---|---|---|---|---|
| `arch` | str | `"gru"` | `"gru"`, `"transformer"`, `"chronos-2"`, `"chronos-2-small"` | Model architecture |
| `train` | bool | `true` | | Set `false` for **eval-only mode** — skips training, resolves model, runs `evaluate_stage1` only |
| `model` | str/null | `null` | path to .pt | Pre-trained model path. Required when `train: false`. If set with `train: true`, skips Stage 1 training and uses this model for Stage 2 |
| `epochs` | int | 50 | >0 | Max training epochs |
| `patience` | int | 5 | >0 | Early stopping patience. Set = `epochs` to disable |
| `hidden_size` | int | 64 | >0 | GRU hidden / Transformer d_model |
| `num_layers` | int | 2 | >0 | Stacked GRU layers or Transformer encoder layers |
| `dropout` | float | 0.2 | 0.0–1.0 | Inter-layer dropout (standard PyTorch GRU/Transformer) |
| `noise_std` | float | 0.0 | ≥0 | Gaussian noise std injected into input features during training. 0 = disabled. See [[regularization]] |
| `recurrent_dropout` | float | 0.0 | 0.0–1.0 | Output dropout on GRU hidden state before FC head. 0 = disabled. See [[regularization]] |
| `wfo_folds` | int | 1 | ≥1 | Walk-Forward Optimization folds. 1 = standard single split. ≥2 = sliding-window WFO with auto-purge. See [[regularization]] |
| `n_heads` | int/null | `null` | >0, must divide `hidden_size` | Transformer attention heads. `null` → auto (`hidden_size // 16`, min 1). Ignored for GRU |
| `optimizer` | dict | `{lr: 3e-4, weight_decay: 0.0}` | see below | Optimizer params (lr, weight_decay) |
| `lookback` | int | 86400 | ≥10 | Sequence length in bars. Min 10 (rolling window constraint) |
| `clip_grad_norm` | float | 1.0 | >0 | Gradient clipping norm |
| `bar_seconds` | int | 1 | 1, 5, 60… | OHLCV bar duration in seconds |
| `stride` | int | 1 | >0 | Step between sequence starts. `stride=60` → 60× fewer samples |
| `horizons` | int/list | 60 | ≥1 | Prediction horizon(s) **in bars** (not seconds). List → multi-head output |
| `probabilistic` | bool | false | | If true: output (mu, sigma), Gaussian NLL loss. If false: MSE |
| `magnitude_alpha` | float | 0.0 | 0.0–1.0 | Weight exponent for magnitude-weighted NLL (only with `probabilistic: true`) |
| `min_coverage` | float | 0.01 | 0.0–1.0 | *Currently unused* — coverage_acc metric not computed in training loop |
| `train_ratio` | float | 0.8 | 0.0–1.0 | Fraction of data for training. 1.0 = no validation (memorization tests) |
| `log_every` | int | 10 | >0 | Log step_loss to W&B every N optimizer steps |
| `batch_size` | int | 1024 | >0 or -1 | Batch size. -1 = auto-probe via OOM binary search |
| `loss` | str | `"mse"` | `"mse"`, `"mae"`, `"huber"`, `"apex"` | Loss function (non-probabilistic mode only). See [[metrics_report]] for Apex Loss details |
| `apex_lam` | float | `1.0` | ≥0 | Downside penalty weight for Apex Loss. `0` = pure capture ratio |
| `dynamic_allocation` | bool | `false` | | Use `tanh(pred)` as fractional position size instead of `sign(pred)` (binary). Only meaningful with `loss: apex` |
| `seed` | int/null | `null` | 0–2³²-1 | Random seed. null → auto-generated. Saved to W&B config for reproducibility |
| `scheduler` | dict/null | `null` | see below | LR scheduler config. null → cosine default. Defined in `StageConfig` (shared with stage2) |
| `train_ratio` | float | 0.8 | 0.0–1.0 | Fraction of data for training (rest = validation) |
| `start_date` | str/null | `null` | ISO date `"YYYY-MM-DD"` | Keep only bars with `timestamp >= start_date`. Applied after resampling, before train/val split. null = no lower bound |
| `end_date` | str/null | `null` | ISO date `"YYYY-MM-DD"` | Keep only bars with `timestamp < end_date` (**exclusive**). Applied after resampling, before train/val split. null = no upper bound |
| `features` | list/null | `null` | list of column names | Custom feature columns. null → `FEATURE_COLS` from constants.py |
| `eval_deterministic` | bool | false | | Force deterministic CUDA ops during eval (exact reproducibility, may be slower) |
| `run_val` | bool | true | | If false, skip validation split during training (useful with `train_ratio: 1.0`) |
| `run_eval` | bool | true | | If false, skip post-training `evaluate_stage1()` call |
| `use_attention` | bool | false | | GRU only: pool over all timestep outputs via learned attention instead of last hidden state |
| `load_offset` | int | 0 | ≥0 | Skip first N raw rows from parquet before loading |

### Chronos-2 specific options

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | str/null | `null` | HuggingFace model ID or local path. `null` → resolved from `arch` |
| `fine_tune_mode` | str | `"lora"` | `"lora"` (LoRA adapters), `"full"` (all params), `"freeze"` (inference only) |
| `lora_rank` | int | 8 | LoRA rank (lower = fewer trainable params) |
| `lora_alpha` | int | 16 | LoRA scaling factor |
| `covariates` | list | `[]` | Additional input series beyond close price, e.g. `["volume"]` |
| `context_length` | int | 512 | How many past bars the model sees |
| `prediction_length` | int | 60 | Bars to forecast. Falls back to `horizons` if not set |

Note: For Chronos-2, `epochs` = number of fine-tuning steps (not epochs). `batch_size` = number of time series per step.

### Eval-only mode

Set `stage1.train: false` to skip training and just evaluate an existing model. The model is resolved from `stage1.model` (local path) or `existing_wandb_run` (downloads W&B artifact).

```yaml
# Eval a local model on different data (console-only, no W&B update)
stage1:
  train: false
  model: "artifacts/gru_predictor_20260310.pt"
data:
  pattern: "icmarkets_btcusd_*_1m.parquet"
logging:
  provider: console
```

```bash
python run.py train eval_config.yml
```

Eval runs `evaluate_stage1()` on both train/val splits, prints all metrics, and logs to the configured logger. With `provider: console`, results print to stdout without updating any W&B run.

### Architecture details

- **GRU**: `nn.GRU(input, hidden_size, num_layers)` → final hidden → linear head. Fast, good for long sequences.
- **Transformer**: `nn.Linear → pos_encoding → TransformerEncoder(num_layers, nhead=n_heads)` → mean pool → linear head. Memory O(lookback²). Encoder-only (no decoder needed for regression tasks).
- **Chronos-2**: Pretrained 120M-param encoder-only foundation model. Fine-tuned with LoRA by default (~1-3% trainable params). Uses raw close prices — no feature engineering.

### Loss modes

| `probabilistic` | `loss` | Loss | Output |
|---|---|---|---|
| `false` | `mse` | MSE | Single scalar per horizon |
| `false` | `mae` | L1 | Single scalar per horizon |
| `false` | `huber` | Huber | Single scalar per horizon |
| `false` | `apex` | Apex Loss | Single scalar per horizon |
| `true` | — | Gaussian NLL | (mu, sigma) per horizon |

With `probabilistic: true` and `magnitude_alpha > 0`, large moves are weighted more:
`weight = |target|^magnitude_alpha`

**Apex Loss** (`loss: apex`) — oracle-normalised directional efficiency with downside penalty. Scale-invariant (dimensionless). `apex_lam` controls the downside term weight. See [[metrics_report]] for full formula and motivation.

### `optimizer` options

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | float | 3e-4 | Initial learning rate for AdamW (stage1) / Adam (stage2) |
| `weight_decay` | float | 0.0 | AdamW weight decay. 0 = identical to Adam. Stage 2 ignores this. |

### `scheduler` options

Controls learning rate scheduling. Omit or set `null` for default cosine annealing. Defined in `StageConfig` (shared between stage1 and stage2).

| `type` | Class | Extra fields |
|---|---|---|
| `cosine` (default) | `CosineAnnealingLR` | `eta_min_ratio` (0.01), `T_max` (=epochs) |
| `plateau` | `ReduceLROnPlateau` | `factor` (0.5), `patience` (10) |
| `step` | `StepLR` | `step_size` (30), `gamma` (0.1) |
| `exponential` | `ExponentialLR` | `gamma` (0.95) |
| `cosine_warm_restarts` | `CosineAnnealingWarmRestarts` | `T_0` (10), `T_mult` (1), `eta_min_ratio` (0.01) |
| `onecycle` | `OneCycleLR` | `max_lr` (optimizer.lr×10), `pct_start` (0.3), `anneal_strategy` ("cos"), `div_factor` (25), `final_div_factor` (1e4), `steps_per_epoch` (1) |
| `none` | constant LR | — |

Examples:
```yaml
# Cosine warm restarts
stage1:
  optimizer:
    lr: 0.001
  scheduler:
    type: cosine_warm_restarts
    T_0: 20
    T_mult: 2
    eta_min_ratio: 0.01

# OneCycleLR (super-convergence)
stage1:
  optimizer:
    lr: 0.001
  scheduler:
    type: onecycle
    max_lr: 0.01       # peak LR (default: optimizer.lr × 10)
    pct_start: 0.3     # fraction of training spent warming up
```

---

## `stage2` — RL Policy

> Stage 2 uses `MixedPolicy` + `PPOLoss` + `icmarkets_env.TradingEnv`. Policy, loss, and env backends are not configurable via YAML — they are fixed in the current implementation.

| Field | Type | Default | Description |
|---|---|---|---|
| `skip` | bool | false | Skip Stage 2 entirely (train only GRU) |
| `epochs` | int | 100 | PPO update rounds |
| `batch_size` | int | 512 | PPO mini-batch size |
| `rollout_steps` | int | 2048 | Transitions per PPO update |
| `clip_grad_norm` | float | 0.5 | Gradient clipping |
| `entropy_coeff` | float | 0.01 | PPO entropy coefficient (continuous params) |
| `type_entropy_coeff` | float | 0.1 | PPO entropy coefficient for action-type head — 10x default keeps action-type from collapsing |
| `patience` | int | 0 | Early-stopping patience on reward. 0 = disabled |
| `num_envs` | int | -1 | Parallel training envs. -1 = auto-detect (defaults to 64). All envs share the full data (random_start mode) |
| `continuous` | bool | false | Env boundary mode. false = disjoint data splits (legacy), true = 1-bar overlap at boundaries |
| `action_encoding` | str | `"parametric"` | Action decoder type. Only `"parametric"` supported |
| `train_ratio` | float | 0.8 | Fraction of data for training (rest = validation) |
| `hidden_size` | int | 256 | MixedPolicy hidden dim |
| `num_layers` | int | 3 | MixedPolicy layers |
| `dropout` | float | 0.0 | Policy dropout |
| `bar_seconds` | int | 60 | OHLCV bar duration |
| `balance` | float | 10000 | Starting balance |
| `leverage` | float | 20.0 | Trading leverage |
| `oracle` | bool | true | Oracle mode (peek at future bars) |
| `n_future` | int | 50 | Future bars in oracle mode. Real-world horizon = `n_future × bar_seconds` (e.g. 10 bars × 60s = 600s) |
| `optimizer` | dict | `{lr: 3e-4}` | Optimizer config (see `optimizer` options above) |
| `scheduler` | dict/null | null | LR scheduler config. null = fixed lr (recommended for PPO) |
| `device` | str | `""` | PyTorch device override. Empty = auto-detect (`cuda` if available) |
| `jax_device` | str | `"cpu"` | JAX device for `jax_env` (`"cpu"` or `"gpu"`). CPU is faster for small batches (no JIT warmup / transfer overhead). Saved in checkpoint; overridable via `--jax-device` CLI flag in `train_validation_policy.py` |
| `seed` | int/null | null | Random seed. null → auto-generated |

---

## `data` — Data Loading

| Field | Type | Default | Valid Values | Description |
|---|---|---|---|---|
| `pattern` | str | `"*.parquet"` | glob pattern | Matched against `~/projects/data/` |
| `load_limit` | int/null | null | >0 or null | Max raw rows from parquet (before processing). null = all |
| `load_offset` | int | 0 | ≥0 | Skip first N raw rows from parquet |

Data pipeline: parquet → resample to `bar_seconds` OHLCV → **date filter** (`start_date`/`end_date` from `stage1`, if set) → compute features → compute targets (`horizons` bars ahead) → 80/20 time-series split (configurable via `train_ratio`).

> **Note:** `start_date`/`end_date` are `stage1` fields, not `data` fields. They are only applied during Stage 1 data loading — Stage 2 does not pass them through to its own `load_data()` call.

---

## `autotuner` — GPU Memory Auto-Tuning

| Field | Type | Default | Valid Values | Description |
|---|---|---|---|---|
| `ram_pct` | int | 80 | 1–100 | % of free RAM to budget |
| `s1_vram_pct` | int | 97 | 1–100 | % of GPU VRAM for Stage 1 batch probe |

Auto-tuner behavior:
- **Stage 1**: When `batch_size = -1`, runs OOM binary search to find max batch size
- **Stage 2**: Uses 70% of VRAM to compute `rollout_steps`, `batch_size`, `num_envs`
- `num_envs` capped at 8192

---

## `logging` — Experiment Tracking

| Field | Type | Default | Valid Values | Description |
|---|---|---|---|---|
| `provider` | str/list | `["wandb", "console"]` | `"wandb"`, `"console"`, or list | Logger backend(s) |
| `tags` | list | `[]` | list of strings | Extra tags added to the W&B run |

| Provider | Required Env Vars |
|---|---|
| `wandb` | `WANDB_API_KEY` (opt: `WANDB_PROJECT`, `WANDB_ENTITY`) |
| `console` | None |

List format (e.g. `[wandb, console]`) → MultiLogger that fans out to all.

---

## Recommended Configurations

### Quick test (seconds)
```yaml
stage1:
  epochs: 5
  patience: 3
  arch: gru
  hidden_size: 64
  num_layers: 3
  lookback: 60
  bar_seconds: 1
stage2:
  skip: true
data:
  pattern: "binance_aaveusdt_20250105_0000_to_20250105_0100.parquet"
  load_limit: 5000
logging:
  provider: console
```

### Memorization test (verify model capacity)
```yaml
stage1:
  epochs: 500
  patience: 500
  arch: transformer
  dropout: 0.0
  train_ratio: 1.0
  probabilistic: true
stage2:
  skip: true
data:
  pattern: "binance_aaveusdt_20250105_0000_to_20250105_0100.parquet"
logging:
  provider: [wandb, console]
```

### Full training (RTX 4090, VastAI)
```yaml
stage1:
  epochs: 200
  patience: 20
  arch: gru
  hidden_size: 64
  num_layers: 3
  lookback: 86400
  horizons: [60, 300, 900, 3600, 7200]  # in bars (e.g. at bar_seconds=1 → 1min to 2hr)
  probabilistic: true
  stride: 60
stage2:
  skip: true
data:
  pattern: "binance_aaveusdt_*.parquet"
logging:
  provider: wandb
```

### Chronos-2 fine-tuning (LoRA)
```yaml
stage1:
  arch: chronos-2
  fine_tune_mode: lora
  lora_rank: 8
  lora_alpha: 16
  context_length: 512
  prediction_length: 60
  covariates: [volume]
  epochs: 1000        # = fine-tuning steps
  optimizer:
    lr: 1e-5
  batch_size: 32
stage2:
  skip: true
data:
  pattern: "*.parquet"
logging:
  provider: [wandb, console]
```

### Full training (GTX 1050 Ti, local)
```yaml
stage1:
  epochs: 200
  patience: 20
  arch: gru
  hidden_size: 64
  num_layers: 2
  lookback: 86400
  stride: 60
stage2:
  skip: true
data:
  pattern: "binance_aaveusdt_*.parquet"
logging:
  provider: wandb
```

---

## `cross_eval` — Cross-Asset Transfer Matrix

Config for `tools/cross_asset_eval.py`. All fields are optional — CLI flags override config values.

| Field | Type | Default | CLI Flag | Description |
|---|---|---|---|---|
| `top` | int/null | `null` | `--top N` | Limit to the **N largest assets** by total parquet file size. `null` = use all discovered ICMarkets assets |

Assets are discovered from `~/projects/data/icmarkets_*_1m.parquet`, ranked by total file size (largest first). `--top 20` keeps only the 20 most data-rich assets — useful to avoid running a full N×N matrix on dozens of small assets.

```yaml
cross_eval:
  top: 20
```

### CLI usage

```bash
python tools/cross_asset_eval.py config.yml --top 20              # train + eval locally
python tools/cross_asset_eval.py config.yml --top 20 --remote     # train on VastAI, eval locally
python tools/cross_asset_eval.py config.yml --phase eval --top 20  # eval only (needs runs.json)
python tools/cross_asset_eval.py config.yml --phase train --top 5  # train only
```

| Flag | Description |
|---|---|
| `--top N` | Top N assets by data size (overrides `cross_eval.top` in config) |
| `--phase {train,eval,all}` | Run train only, eval only, or both (default: `all`) |
| `--remote` | Train on VastAI via `vastai_train.sh` instead of locally |

### Output

- `experiments/cross_eval_runs.json` — persisted `{asset: wandb_run_name}` map (survives restarts)
- `experiments/cross_eval_matrix.csv` — N×N pearson correlation matrix
- Summary: top transfer pairs, best generalizers, Ward clustering (3 and 5 clusters)
- `configs/cross_eval/cluster_k{k}_c{id}.yml` — per-cluster training configs (k=3 and k=5)

### Cluster configs

After eval, the tool generates one training config per cluster. Each config uses `data.pattern` as a list of all member asset globs, so a single model trains on the whole cluster:

```yaml
# Example: configs/cross_eval/cluster_k3_c1.yml
data:
  pattern:
    - "icmarkets_eurusd_*_1m.parquet"
    - "icmarkets_gbpusd_*_1m.parquet"
logging:
  tags: ["cluster-k3-c1"]
```

Train a cluster model directly:
```bash
python run.py train configs/cross_eval/cluster_k3_c1.yml
```

---

## Hyperparameter Sweeps

Use `tools/gen_sweep.py` to generate experiment configs, then run sequentially on VastAI:

```bash
.venv/bin/python tools/gen_sweep.py        # → experiments/sweep_configs/*.yml
bash tools/vastai_train.sh a.yml,b.yml,c.yml             # comma-separated, sequential
```

All configs share one VastAI instance. Each run logs to W&B independently. Artifacts saved per-config in `experiments/`.
