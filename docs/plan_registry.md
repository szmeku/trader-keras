# Plan: Extract generic registry from trader

## What

Copy `make_registry()` from `../trader/crypto_trader/registry.py` (lines 1-29).

One factory function that creates `(register, create, registry_dict)` for any component type.

## Where to use

1. **LOSSES** — replace current hand-rolled `LOSSES` dict + `_register_loss` in `models/gru.py`
2. **BACKBONES** — when we add backbone swapping (GRU, LSTM, Transformer)
3. **PIPELINES** — for pipeline dispatch in `run.py`

## Source

```
../trader/crypto_trader/registry.py  — full file, 39 lines
```

## Destination

```
trader_keras/registry.py
```

## Status

Not started — extract when we begin implementing the pipeline architecture.
