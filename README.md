# trader-keras

Keras 3 + JAX rewrite of the trading predictor.

## Quick Start

```bash
uv pip install -e .
python run.py                                    # train with defaults
python run.py stage1.lr=0.001 data.load_limit=50000  # override params
python run.py --config-name=bench                # use bench config
WANDB_MODE=disabled python run.py                # no W&B
```

## Hydra Config

Config is managed by [Hydra/OmegaConf](https://hydra.cc/). No custom config loading — Hydra handles YAML parsing, merging, CLI overrides, and config groups.

**Override any param from CLI:**
```bash
python run.py stage1.lr=0.001 stage1.epochs=50 data.load_limit=10000
```

**Config groups** — when we need shared fields across stages (e.g. predict vs RL), Hydra handles it via YAML defaults and merging. No Python base classes needed — just config files.

See [Hydra config groups docs](https://hydra.cc/docs/tutorials/structured_config/config_groups/).

## Features

6 log-ratio features (matching trader-t5x):

| Feature | Formula |
|---|---|
| `log_open` | `log(open / prev_open)` |
| `log_high` | `log(high / prev_high)` |
| `log_low` | `log(low / prev_low)` |
| `log_close` | `log(close / prev_close)` |
| `log_volume` | `log(volume / prev_volume)` clipped [-10, 10] |
| `norm_spread` | `spread / close` clipped [0, 0.1] |

## Tests

```bash
WANDB_MODE=disabled uv run pytest tests/ -v
```
