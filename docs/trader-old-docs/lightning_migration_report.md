# PyTorch Lightning Migration Report

> Planned 2026-03-17 · **Completed 2026-03-18** — Stage 1 migration done, Stage 2 unchanged.

---

## Status: DONE (Stage 1)

### What was done

Stage 1 (`StandardForecaster.train()`) migrated from a manual loop to Lightning. Stage 2 (PPO) stays custom — Lightning doesn't fit RL.

**New files:**

| File | Purpose |
|---|---|
| `crypto_trader/models/lightning_module.py` | `GRULightningModule` — training/val steps, `configure_optimizers` |
| `crypto_trader/trainer/callbacks.py` | `Stage1MetricsCallback`, `OldFormatModelCheckpoint`, `stopped_epoch()` |
| `logger/lightning_adapter.py` | `LightningBaseLoggerAdapter` — bridges Lightning → BaseLogger |

**Modified files:**

| File | Change |
|---|---|
| `crypto_trader/models/standard.py` | `train()`: 220 lines → 80 lines (uses `L.Trainer`) |
| `crypto_trader/models/strided_loader.py` | Added `StridedDataLoaderWrapper` (~20 lines) |
| `crypto_trader/models/training.py` | Kept `train_epoch` for tests; added docstring note |

**Deleted:** Nothing — `train_epoch` kept as re-export in `gru.py` (tests depend on it).

### Architecture after migration

```
run.py → stage1_train()
  → StandardForecaster.prepare_data()   # unchanged
  → StandardForecaster.build_model()    # unchanged
  → StandardForecaster.train()
      → GRULightningModule(model, loss_fn, config)
      → Stage1MetricsCallback           # epoch timing, LR, overfit_ratio, step_loss
      → OldFormatModelCheckpoint        # legacy {"model": state_dict, ...} format
      → EarlyStopping
      → L.Trainer(
            logger=LightningBaseLoggerAdapter(base_logger),
            enable_progress_bar=False,  # avoid per-step tqdm overhead
            ...
        ).fit(module, StridedDataLoaderWrapper(train_loader), val_dl)
  → StandardForecaster.evaluate()       # unchanged
```

### What Lightning replaced

| Old code | Lightning equivalent |
|---|---|
| Manual `for epoch` loop (~220 lines) | `Trainer.fit()` |
| Manual early stopping patience counter | `EarlyStopping` callback |
| `torch.save()` best-model logic (20 lines) | `OldFormatModelCheckpoint` callback |
| Manual `clip_grad_norm_` call | `Trainer(gradient_clip_val=..., gradient_clip_algorithm="norm")` |
| Manual epoch/step counter | `trainer.current_epoch`, `trainer.global_step` |
| Manual LR scheduler step | Lightning calls scheduler per epoch automatically |

### What was NOT migrated (as planned)

- Stage 2 PPO loop — RL doesn't fit supervised paradigm
- `StridedLoader` — kept as-is, wrapped via `StridedDataLoaderWrapper`
- `memory_tuner.py` batch size probe — kept, `Tuner.scale_batch_size()` can't use StridedLoader
- `logger/base.py + wandb.py` — kept, bridged via `LightningBaseLoggerAdapter`
- Checkpoint format — kept legacy `{"model": state_dict, ...}` via `OldFormatModelCheckpoint`
- `seed_everything()` — kept `init_seeds()` (Lightning's version consumes random state)

### Key implementation notes

**`"epoch"` key filtering**: Lightning injects `"epoch"` into every metric dict before calling the adapter. `LightningBaseLoggerAdapter.log_metrics()` filters it to prevent a spurious `summary.epoch` key in W&B.

**`transfer_batch_to_device` no-op**: `GRULightningModule` overrides this to return the batch unchanged — `StridedLoader` already places tensors on the training device, Lightning's default `.to(device)` call was redundant.

**`enable_progress_bar=False`**: Tqdm updates every step including string formatting. For small batch sizes on a fast GPU, this Python overhead was the main source of samples/sec regression vs. the old manual loop.

**LR timing**: Lightning logs LR via callback before the scheduler steps (old code logged after). `summary.stage1/lr` differs by 1 epoch — accepted, reference run updated to use Lightning code.

**WFO**: Lightning resets `global_step` per fold. `config["_global_step_offset"]` is set after each fold so the next fold's logging continues from the right step.

### Test coverage

- `tests/test_rolemodel.py` — regression test against W&B reference run `genial-blaze-1317`
- `tests/test_lightning_adapter.py` — 15 tests for the adapter
- All 638 tests pass

---

## Original Plan (2026-03-17, preserved for reference)

See git history for the original planning content.
