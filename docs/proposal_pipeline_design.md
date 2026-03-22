# Pipeline Architecture

## Status

All core steps implemented and tested (70 tests passing).

| Step | File | Status |
|---|---|---|
| `load` | `steps/data.py` | Done |
| `featurize` | `steps/data.py` | Done |
| `window` | `steps/data.py` | Done |
| `env` | `steps/env.py` | Done |
| `model` | `steps/model.py` | Done |
| `checkpoint` | `steps/model.py` | Done |
| `fit_supervised` | `steps/train.py` | Done |
| `fit_rl` | `steps/rl.py` | Done |
| `save` | `steps/train.py` | Done |

| Pipeline | Steps | Status |
|---|---|---|
| `predict` | load, featurize, window, model, checkpoint, fit_supervised, save | Done |
| `rl` | load, featurize, env, model, checkpoint, fit_rl, save | Done |
| `predict_then_rl` | predict pipeline → freeze → RL heads → fit_rl | Not started |

## Architecture

Each step is `ctx → ctx`. A pipeline is a list of steps. `pipe` runs them.

```python
PIPELINES = {
    "predict": [load, featurize, window, model, checkpoint, fit_supervised, save],
    "rl":      [load, featurize, env,   model, checkpoint, fit_rl,         save],
}
```

Config selects pipeline and optionally skips steps:

```bash
python run.py                                    # default: predict
python run.py pipeline=rl                        # switch to RL
python run.py pipeline.skip=[checkpoint]         # skip checkpoint loading
python run.py train.lr=0.001                     # override params
```

## Hydra config groups

```
conf/
  config.yaml              # shared: backbone, data, wandb
  pipeline/
    predict.yaml           # @package _global_ — train params
    rl.yaml                # @package _global_ — data overrides + env + rl params
    bench.yaml             # @package _global_ — benchmark config
```

Pipeline yamls use `@package _global_` so `train`, `env`, `rl` merge at root level.
`cfg.train.lr` works everywhere — no `cfg.pipeline.train` nesting.

## Remaining work

- `predict_then_rl` composed pipeline (use Keras model surgery to slice backbone)
- Backbone swapping via registry (GRU, LSTM, Transformer) — see `plan_registry.md`
- Streaming GRU (batch-as-isolation, lookback-as-reset) — see `open_lookback_windowing.md`
