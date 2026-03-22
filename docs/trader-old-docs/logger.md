# Logger

Generic ML/trading logger module with multiple backends. See [[README]] for experiment tracking setup.

---

## Quick Start

```python
from crypto_trader.logger import create_logger

# Create logger by provider name
logger = create_logger(provider="wandb", tags=["production"])

# Log static parameters
logger.log_param("model/type", "MLP")
logger.log_params({"lr": 1e-4, "batch_size": 64})

# Log time-series metrics
logger.log("train/loss", 0.5, step=0)
logger.log_metrics({"train/loss": 0.5, "val/loss": 0.6}, step=100)

# Track files
logger.track_files("models/best", "models/best_model.pt")

# Cleanup
logger.wait()
logger.stop()
```

## Environment Setup

### Weights & Biases
```bash
# .env
WANDB_API_KEY=your-api-key
WANDB_PROJECT=crypto-trader          # optional, defaults to "crypto-trader"
WANDB_ENTITY=your-entity             # optional
```

---

## Providers

| Provider | Config value | Env vars needed |
|---|---|---|
| [W&B](https://wandb.ai) | `wandb` | `WANDB_API_KEY` |
| Console | `console` | None |

Switch providers by changing `logging.provider` in `config.yml`. All providers implement the same `BaseLogger` interface.

---

## API Reference

### BaseLogger (interface)

| Method | Description |
|---|---|
| `log(key, value, step=None)` | Log time-series metric |
| `log_param(key, value)` | Log static parameter |
| `log_metrics(dict, step=None)` | Log batch of metrics |
| `log_params(dict)` | Log batch of parameters |
| `log_summary(dict)` | Log final summary metrics (W&B summary section, not config) |
| `wait()` | Flush async data |
| `stop()` | End run |

### WandbLogger

Extends `BaseLogger` with:

| Method | Description |
|---|---|
| `log_image(key, path)` | Upload image as `wandb.Image` |
| `log_artifact(path)` | Upload as W&B artifact |
| `track_files(key, path)` | Upload as W&B artifact |

### ConsoleLogger

Prints all logs to stdout. Useful for testing and debugging.

### MultiLogger

Fans out to multiple backends. Created automatically when `logging.provider` is a list (e.g. `[wandb, console]`). Delegates all `BaseLogger` methods to each child logger.

---

## Factory Function

```python
from crypto_trader.logger import create_logger

logger = create_logger(
    provider=["wandb", "console"],  # "wandb", "console", or list
    tags=["experiment-1"],
)
```

`WandbLogger` accepts `fallback_to_console=True` to fall back gracefully when credentials are missing.
