# trader-keras

Keras 3 + JAX-GPU rewrite of the crypto trading predictor. Simplified, no PyTorch, no Numba.

**See also:** [[docs/architecture.md]] (system design), [[docs/status.md]] (progress)

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Framework | Keras 3.x | Backend-agnostic, clean API |
| Backend | JAX + CUDA | Max GPU throughput, JIT/XLA |
| Experiment tracking | W&B | Same as original project |
| Data | pandas + pyarrow | Simple, no Numba dependency |
| Package manager | uv | Fast, consistent with original |

---

## Install

```bash
# From project root
uv pip install -e .

# Verify GPU
KERAS_BACKEND=jax uv run python3 -c "import jax; print(jax.devices())"
```

---

## CLI

```bash
# Train with default config.yml
KERAS_BACKEND=jax uv run python3 run.py train

# Train with custom config
KERAS_BACKEND=jax uv run python3 run.py train my_config.yml
```

---

## Configuration

Edit `config.yml`:

```yaml
stage1:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  lookback: 60          # bars of context
  horizons: [1, 5, 10, 30, 60]   # bars ahead to predict
  bar_seconds: 60       # 1m bars
  stride: 1             # sequence stride (increase to reduce dataset size)
  epochs: 100
  patience: 10
  batch_size: 1024
  lr: 0.0003            # Use decimal, NOT scientific notation (3e-4) — YAML quirk
  train_ratio: 0.8
  probabilistic: true   # Gaussian NLL, outputs (mu, sigma) per horizon

data:
  pattern: "icmarkets_eurusd_*_1m.parquet"
  data_dir: "~/projects/data"
  load_limit: null      # set e.g. 10000 for fast iteration

logging:
  provider: console     # "wandb", "console", or [wandb, console]
  project: "trader-keras"
```

**YAML gotcha:** `3e-4` is parsed as a string by PyYAML. Use `0.0003` instead.

---

## Architecture

```
Parquet files (OHLCV bars)
    │
    ▼
create_features()    # 14 features, all shift(1) — no leakage
create_targets()     # log(close[t+h] / close[t]) for each horizon
    │
    ▼
GRU stack (Keras 3)
  Input: (batch, lookback, n_features)
  → GRU × num_layers
  → final hidden state (batch, hidden_size)
  → per-horizon Dense heads: (mu, log_sigma) each
  Output: (batch, n_horizons, 2)
    │
    ▼
Gaussian NLL loss    # proper scoring rule; penalizes both wrong mu and wrong sigma
```

---

## Project Structure

```
trader_keras/
├── __init__.py
├── config.py           # Dataclass config + YAML loader
├── constants.py        # FEATURE_COLS, OPTIONAL_FEATURE_COLS
├── trainer.py          # Stage 1 training loop (Keras model.fit + callbacks)
├── data/
│   ├── loader.py       # parquet → sequences (x_train, y_train, x_val, y_val)
│   ├── features.py     # Technical indicators (no leakage)
│   └── resampler.py    # Trade ticks → OHLCV bars, bar reaggregation
├── models/
│   └── gru.py          # GRU model, gaussian_nll_loss, mse_loss
└── logger/
    ├── base.py         # BaseLogger, ConsoleLogger, MultiLogger
    ├── wandb_logger.py # W&B adapter (reads WANDB_API_TOKEN from .env)
    └── factory.py      # create_logger(cfg)

tests/
├── test_data_pipeline.py
└── test_model.py

run.py                  # CLI entry point
config.yml              # Default config
```

---

## Tests

```bash
KERAS_BACKEND=jax uv run pytest tests/ -v
```

---

## Data

Parquet files expected in `~/projects/data/`. Supports:
- Pre-aggregated OHLCV bars (`icmarkets_*_1m.parquet`)
- Raw tick trades (`timestamp, price, amount, side`) — auto-resampled

Features (13 default + optionals when present):
- OHLCV: `high, low, volume`
- Returns: `log_returns, volatility, volume_ma, volume_ratio`
- Log-scale: `log_volume, log_volatility, log_volume_ma`
- Derived: `vol_regime, price_accel, time_in_day_counter`
- Optional: `buy_ratio` (Binance data), `log_spread` (ICMarkets data)

All features use `.shift(1)` — no future leakage.
