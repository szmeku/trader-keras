# Proposal: Registry Pattern for Swappable Components

## Core Idea

Three registries — `BACKBONES`, `HEADS`, `LOSSES` — same pattern. Swap anything via config, zero ifs. Adding a new component is one decorated function.

## Pattern

```python
REGISTRY = {}

def _register(fn):
    REGISTRY[fn.__name__] = fn
    return fn
```

## BACKBONES registry

Contract: any backbone must be `(batch, seq, features) → (batch, hidden)`.

```python
BACKBONES = {}

@_register
def gru(cfg, n_features):
    inputs = keras.Input(shape=(cfg.lookback, n_features))
    x = inputs
    for i in range(cfg.num_layers):
        x = layers.GRU(cfg.hidden_size, return_sequences=(i < cfg.num_layers - 1),
                       dropout=cfg.dropout, name=f"gru_{i}")(x)
    return keras.Model(inputs, x, name="gru_backbone")

@_register
def lstm(cfg, n_features):
    ...

@_register
def transformer(cfg, n_features):
    ...
```

Config:

```yaml
backbone:
  type: gru            # registry lookup
  hidden_size: 64
  num_layers: 2
  checkpoint: null
  freeze: false
```

## HEADS registry

Contract: takes `hidden_size` and config, returns head layers.

```python
HEADS = {}

@_register
def predict(hidden_size, cfg):
    """Per-horizon Dense heads → (batch, n_horizons, n_out)"""
    ...

@_register
def policy(hidden_size, cfg):
    """Action distribution head for RL"""
    ...

@_register
def value(hidden_size, cfg):
    """Value function head for RL"""
    ...
```

## LOSSES registry (already implemented)

```python
LOSSES = {}

@_register
def gaussian_nll(y_true, y_pred): ...
gaussian_nll.n_out = 2

@_register
def mse(y_true, y_pred): ...
mse.n_out = 1
```

## Model assembly — no ifs

```python
def build_model(cfg, n_features):
    backbone = BACKBONES[cfg.backbone.type](cfg.backbone, n_features)
    head = HEADS[cfg.heads.type](backbone.output_shape[-1], cfg)
    return assemble(backbone, head)
```

## Benefits

- Adding a new backbone/head/loss = one function + decorator
- Config drives all choices — no code changes to try new architectures
- Same pattern everywhere — learn once, apply to all three registries
- Checkpoint/freeze works on any backbone — the contract guarantees compatibility
