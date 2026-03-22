# Regularization & Generalization

Techniques to prevent overfitting in Stage 1 (GRU/Transformer predictor). All disabled by default (zero values) — no impact on existing configs.

Related: [[config_reference]]

---

## Weight Decay (AdamW)

```yaml
stage1:
  optimizer:
    weight_decay: 1e-4  # 0.0 = disabled (default)
```

**What it does:** Decouples weight decay from the gradient update. Every step, all weights are multiplied by `(1 - lr * weight_decay)`, pulling them toward zero independently of the loss gradient.

**Why it prevents memorization:** Large weights encode specific training patterns. Weight decay penalizes weight magnitude, forcing the model to find solutions with smaller weights — which are more general. Unlike L2 regularization (which adds `lambda * ||w||^2` to the loss), AdamW applies decay *after* the adaptive learning rate, so it works correctly with Adam's per-parameter scaling.

**How to disable:** Set `weight_decay: 0.0`. AdamW with `weight_decay=0` is mathematically identical to Adam — verified in tests.

**Typical values:** `1e-5` to `1e-3`. Start with `1e-4`.

---

## Input Noise Injection

```yaml
stage1:
  noise_std: 0.01  # 0.0 = disabled (default)
```

**What it does:** During training only, adds Gaussian noise `N(0, noise_std^2)` to input features at each batch. At validation/inference time, no noise is added.

**Why it helps:** Acts as data augmentation — the model sees slightly different versions of each training example, preventing it from memorizing exact input patterns.

**How to disable:** Set `noise_std: 0.0`. When zero, the noise generation is skipped entirely (no overhead).

**Typical values:** `0.001` to `0.05`. Features are z-scored (std ~1), so `0.01` means ~1% noise. Start with `0.01`.

---

## Recurrent Dropout

```yaml
stage1:
  dropout: 0.1            # standard inter-layer dropout (PyTorch GRU built-in)
  recurrent_dropout: 0.2   # 0.0 = disabled (default)
```

**What it does:** Applies `nn.Dropout` on the GRU's final hidden state before the FC head.

**Why two dropout knobs?**
- `dropout`: PyTorch GRU's built-in — applied between stacked GRU layers (only active when `num_layers > 1`).
- `recurrent_dropout`: applied to GRU output before FC head. Works regardless of `num_layers`.

**How to disable:** Set `recurrent_dropout: 0.0`. No extra module is created — zero overhead.

**Typical values:** `0.1` to `0.3`. Don't go above `0.5`.

---

## Walk-Forward Optimization (WFO)

```yaml
stage1:
  wfo_folds: 3  # 1 = standard single split (default), ≥2 = sliding-window WFO
```

**What it does:** Sliding-window cross-validation for time series. Tests generalization across multiple future periods instead of relying on a single val split.

**How it works:** Data is divided into `wfo_folds + 1` equal chunks. Each fold trains on one chunk and validates on the next:

```
wfo_folds=3 → 4 chunks: [A][B][C][D]

Fold 0: train=[A]   val=[B]    (train on oldest, validate on next)
Fold 1: train=[B]   val=[C]    (window slides forward)
Fold 2: train=[C]   val=[D]    (most recent data)
```

Each fold trains from scratch. The last fold's model is returned.

**Why sliding window (not expanding)?** Crypto markets have heavy concept drift — old data can hurt more than help. Fixed-size train window discards stale data, keeping the model focused on recent patterns.

**Auto-purge:** A gap of `max(horizons)` rows is automatically inserted between train and val to prevent target leakage (the last training sample's target uses future data up to `horizon` bars ahead).

**How to disable:** Set `wfo_folds: 1`. Uses standard `train_ratio` split. No WFO overhead.

**Logged metrics:**
- `wfo/fold_N/best_val_loss` — per-fold best validation loss
- `wfo/mean_val_loss` — mean across folds
- `wfo/std_val_loss` — std across folds (lower = more stable generalization)
- `wfo/purge` — auto-computed purge gap (rows)

**Practical notes:**
- Each fold trains from scratch → `wfo_folds=3` takes ~3x training time
- Chunk size must be > `2 * lookback` rows
- `train_ratio` is ignored when `wfo_folds ≥ 2` (WFO manages its own splits)

---

## Recommended Combinations

### Conservative (first try)
```yaml
stage1:
  optimizer:
    weight_decay: 1e-4
  noise_std: 0.01
```

### Moderate
```yaml
stage1:
  optimizer:
    weight_decay: 1e-4
  noise_std: 0.01
  recurrent_dropout: 0.15
```

### Full regularization + validation
```yaml
stage1:
  optimizer:
    weight_decay: 1e-4
  noise_std: 0.01
  recurrent_dropout: 0.2
  wfo_folds: 3
```
