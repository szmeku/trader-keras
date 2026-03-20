# Proposal: Registry Pattern for Swappable Components

## Core Idea

Three registries — `BACKBONES`, `HEADS`, `LOSSES` — same pattern. Swap anything via config, zero ifs. Adding a new component is one decorated function.

## Generic registry factory (from trader)

trader has a battle-tested `make_registry()` that creates register/create/registry for any component type:

```python
# registry.py
def make_registry(description: str = "item"):
    reg: dict[str, type] = {}

    def register(key: str):
        def decorator(cls):
            reg[key] = cls
            return cls
        return decorator

    def create(key: str, **kwargs):
        if key not in reg:
            raise ValueError(f"Unknown {description} {key!r}. Available: {sorted(reg)}")
        return reg[key](**kwargs)

    return register, create, reg
```

One factory, all registries:

```python
register_backbone, create_backbone, BACKBONES = make_registry("backbone")
register_loss, create_loss, LOSSES = make_registry("loss")
```

## BACKBONES registry

Contract: any backbone must expose `get_embedding()` → `(batch, hidden)`.

Both prediction heads and policy heads consume the same embedding. Freeze/unfreeze applies at this boundary.

```python
@register_backbone("gru")
class GRUBackbone:
    def get_embedding(self, x):
        """(batch, seq, features) → (batch, hidden)"""
        out = self.gru(x)
        return out[:, -1, :]
```

Config: `backbone.type: gru` → registry lookup.

## LOSSES registry (already implemented)

```python
@_register_loss
def gaussian_nll(y_true, y_pred): ...
gaussian_nll.n_out = 2

@_register_loss
def mse(y_true, y_pred): ...
mse.n_out = 1
```

## Shared config base (from trader)

trader uses `StageConfig` base class with shared fields inherited by stage-specific configs:

```python
@dataclass
class StageConfig:
    hidden_size: int = 64
    num_layers: int = 2
    lookback: int = 60
    batch_size: int = 1024
    lr: float = 3e-4
    seed: int = 42

@dataclass
class PredictConfig(StageConfig):
    horizons: list[int] = field(default_factory=lambda: [10])
    loss: str = "gaussian_nll"

@dataclass
class RLConfig(StageConfig):
    algorithm: str = "ppo"
    rollout_steps: int = 2048
```

## Lessons from trader

**Take:** `make_registry()`, `get_embedding()` backbone contract, `StageConfig` base class

**Avoid:** forecaster protocol bundles prepare_data + build_model + train + evaluate into one class — can't mix steps across architectures
