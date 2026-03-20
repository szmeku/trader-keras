# Plan: Backbone with get_embedding() contract

## What

Backbone exposes `get_embedding(x) → (batch, hidden)` — everything before heads is shared, everything after is task-specific.

```
Input (batch, seq, features)
        │
    BACKBONE (shared, freezable)
        │
   get_embedding() → (batch, hidden)
        │
   ┌────┴────┐
predict    RL
heads      heads
```

## Reference

trader's GRUPredictor already does this: `../trader/crypto_trader/models/gru.py` lines 120-126.

```python
def get_embedding(self, x):
    out, _ = self.gru(x)
    h = self._pool(out)
    return h
```

## How it enables workflows

- **Predict**: backbone + predict heads → supervised fit
- **RL**: backbone + policy/value heads → RL fit
- **Pretrain → RL**: train backbone+predict heads → freeze backbone → attach RL heads → RL fit
- **Fine-tune**: load checkpoint → unfreeze backbone → continue training

Freeze/unfreeze applies at the `get_embedding()` boundary: `backbone.trainable = False`.

## Status

Not started — implement when building backbone swapping (together with registry).
