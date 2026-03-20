# Proposal: Composable Pipeline Architecture

## Core Idea

Pipeline = sequence of step functions, composed from shared building blocks.

Each step is `ctx → ctx`. A pipeline is a list of steps. `pipe` runs them.

```python
def pipe(ctx, *steps):
    for step in steps:
        ctx = step(ctx)
    return ctx
```

## Structure

```
trader_keras/
  config.py        # Hydra dataclasses
  backbone.py      # BACKBONES registry (GRU, LSTM, Transformer, ...)
  steps.py         # atomic step functions: ctx → ctx
  pipelines.py     # compositions of steps
  run.py           # entry point, PIPELINES registry
```

## Steps — shared building blocks

Every function is `ctx → ctx`:

```python
def load(ctx):        # parquet → DataFrame
def featurize(ctx):   # log-ratio features
def window(ctx):      # sliding window → (X, Y) splits (supervised)
def env(ctx):         # wrap same bars as stepping environment (RL)
def model(ctx):       # backbone + heads from config (registries)
def checkpoint(ctx):  # load weights + optional freeze
def fit_supervised(ctx):  # keras model.fit()
def fit_rl(ctx):          # rollout → PPO update loop
def save(ctx):            # save model + wandb.finish()
```

## Pipelines — composed from steps

```python
predict = [load, featurize, window, model, checkpoint, fit_supervised, save]
rl      = [load, featurize, env,   model, checkpoint, fit_rl,         save]
```

New pipelines = list operations:

```python
predict_then_rl = predict[:-1] + [freeze, swap_heads, fit_rl, save]
```

## Dispatch — registry, no ifs

```python
PIPELINES = {"predict": predict, "rl": rl}

@hydra.main(...)
def main(cfg):
    pipe({"cfg": cfg}, *PIPELINES[cfg.pipeline])
```

```bash
python run.py pipeline=predict
python run.py pipeline=rl backbone.checkpoint=artifacts/xxx.keras backbone.freeze=true
```

## Shared data flow

The environment IS historical data. Both supervised and RL consume the same bars DataFrame — `window` and `env` are two views of the same data:

```
parquet → features → bars DataFrame
                          │
              ┌───────────┴───────────┐
              │                       │
         window (supervised)     env (RL)
         pre-slice windows       step through bars
         → (X, Y) batches       sequentially
```

5 of 7 steps are fully shared. Only the data iteration interface (`window` vs `env`) and the training loop (`fit_supervised` vs `fit_rl`) differ.

## Pipe library

`toolz.pipe` is the Python Ramda equivalent, but our `pipe` is 3 lines — skip the dependency unless we need `curry`/`compose` later.
