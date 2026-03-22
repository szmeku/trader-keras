# Stage 1: Probabilistic Multi-Horizon GRU Proposal

**See also:** [[architecture#Stage 1: GRU Predictor (Supervised)]], [[ideas]], [[status]]

---

## Summary

Replace the current single-horizon MSE regression with a **probabilistic multi-horizon** architecture.

**Symbol legend:**
- `μ` (mu) — the model's **predicted return**: "I expect price to move +0.05% over the next 30s." Think of it as the center of a bell curve of possible outcomes — not an average of past prices, but the single best guess of the future return. Also called "expected return."
- `σ` (sigma) — **standard deviation of the prediction** (σ IS the standard deviation — not a coincidence). It describes the spread of the model's uncertainty. μ ± 2σ gives the 95% confidence interval: "I expect +0.05% return and I'm 95% sure it lands between +0.01% and +0.09%."
- `SNR` (Signal-to-Noise Ratio) — `|μ| / σ`: how large the predicted move is relative to the predicted uncertainty. SNR=3 means the predicted move is 3× larger than the noise floor — a high-conviction signal. SNR=0.1 means the model is barely guessing.

**What we change:**

1. Predict `(μ, σ)` per horizon instead of just a point estimate.
2. Use **Gaussian NLL loss** — a proper scoring rule that penalizes both wrong predictions AND wrong uncertainty. Unlike MSE, it cannot be minimized by predicting zero everywhere. Full explanation below.
3. Derive **SNR confidence** = `|μ| / σ` per bar — a natural measure of how strongly the model believes in a direction.
4. **Selectively predict** — only generate a signal on the top `coverage` fraction of highest-SNR bars. This is configurable (default 1%, see Config section).

This makes Stage 1 self-aware: it knows when it knows something and when it doesn't.


## Proposed Architecture

One shared GRU backbone, multiple output heads — all trained jointly.

```
Input: (batch, lookback, 36 features)
    │
    ▼
GRU(36 → hidden=64, layers=2, dropout=0.2)   ← shared backbone
    │
    ▼
hidden state h_t: (batch, 64)
    │
    ├──► Head 1s:   Linear(64, 2) → (μ₁,  σ₁)
    ├──► Head 5s:   Linear(64, 2) → (μ₅,  σ₅)
    ├──► Head 10s:  Linear(64, 2) → (μ₁₀, σ₁₀)
    ├──► Head 30s:  Linear(64, 2) → (μ₃₀, σ₃₀)
    ├──► Head 60s:  Linear(64, 2) → (μ₆₀, σ₆₀)
    └──► Aux head:  Linear(64, 2) → (future_max, future_min)  [optional]
```

Each head outputs `(μ, σ)` for one horizon. `σ` is passed through `softplus` to keep it positive.

**Why μ?** μ is the model's best point estimate of the log return at that horizon — the "where am I predicting price goes?" part. We need it to generate a trade direction: `sign(μ)` = long or short.

**Why σ?** σ is the model's estimated uncertainty about that prediction. It lets us distinguish "I predict +0.05% return ± 0.01%" (high conviction) from "I predict +0.05% return ± 5%" (low conviction, noise). Without σ we can't know whether to trust μ.

**Why log return as target?**
Yes — log returns are better than simple returns and we should use them:
- **Additive across horizons:** `log_ret_60s = log_ret_1s + log_ret_2s + ...` — natural multi-horizon decomposition
- **More Gaussian in distribution** — Gaussian NLL assumption is more valid
- **Scale-invariant** — works identically for $100 and $50,000 assets without normalization hacks
- **Already in our features** (`log_returns` column exists) — consistent throughout the pipeline

Target formula per horizon `h` (in bars):

$r^{\log}_h = \log\left(\frac{\text{close}[t+h]}{\text{close}[t]}\right)$

---

## Loss Function

### Gaussian NLL (per horizon)

**Symbol legend:**
- `r_h` — actual log return at horizon h (the ground truth)
- `μ_h` — model's predicted mean log return at horizon h
- `σ_h` — model's predicted standard deviation (spread of uncertainty) at horizon h

**Is Gaussian σ the right uncertainty measure?**

Gaussian assumes returns follow a bell curve — which is approximately true for short horizons (1–60s) but under-estimates rare large moves (fat tails). Alternatives:

| Approach | Predicts | Pros | Cons |
|---|---|---|---|
| **Gaussian (μ, σ)** ← proposed | mean + std | Simple, clean NLL loss, good starting point | Under-estimates tails, assumes symmetry |
| **Quantile regression (q05, q50, q95)** | 3 percentiles | No distributional assumption, handles fat tails naturally | No single confidence number, harder to combine across horizons |
| **Laplace (μ, b)** | mean + scale | Fatter tails than Gaussian, still a proper scoring rule | Less common, slightly more complex |

**Recommendation:** Start with Gaussian. For 1–60s returns the Gaussian assumption is not badly wrong, and the NLL loss is well-understood. If post-training calibration shows systematic tail mis-coverage, switch to quantile regression — the architecture is identical, only the loss and output heads change.

$$\mathcal{L}_h = \frac{(r_h - \mu_h)^2}{\sigma_h^2} + \log \sigma_h^2$$

**Why this works (and MSE doesn't):**
The first term `(r-μ)²/σ²` pushes μ to be accurate AND σ to be large enough to cover errors.
The second term `log σ²` penalizes making σ too large — it grows without bound as σ → ∞.

These two terms are in tension: the model cannot minimize loss by inflating σ (second term increases), and cannot minimize loss by predicting zero (first term blows up when μ is wrong). The only optimum is a calibrated `(μ, σ)` pair. This is called a **proper scoring rule**.

**Total loss:**
$$\mathcal{L} = \sum_{h \in \{1,5,10,30,60\}} \mathcal{L}_h + \lambda \cdot \mathcal{L}_{\text{aux}}$$

where `λ = 0.1` and `L_aux = MSE(predicted max/min, actual max/min)` over the horizon window.

---

## Confidence Scoring: SNR

After training, compute per-bar confidence at each horizon:

$$\text{SNR}_h = \frac{|\mu_h|}{\sigma_h}$$

High SNR = strong directional belief relative to uncertainty.

**Usage at inference:**
```python
# Compute SNR for each horizon
snr = abs(mu) / sigma  # shape: (T, 5) for 5 horizons

# Combine: use max SNR across horizons
confidence = snr.max(axis=1)

# Select top coverage% of bars (configurable)
threshold = np.quantile(confidence, 1 - coverage)
signal_mask = confidence > threshold
```

The threshold `τ` is computed post-training from the held-out validation distribution — no retraining needed to change coverage.

---

## Selective Prediction

**Don't trade on every bar — only trade when confident.**

```
coverage = fraction of bars where a signal is generated
           (e.g., 0.01 → signal on 1% of bars)

Goal: maximize accuracy/return on signal bars,
      subject to coverage ≥ min_coverage
```

**Why `min_coverage` matters:** Without a minimum, the model could cherry-pick one trivially easy bar and claim perfect accuracy. `min_coverage = 0.01` ensures the strategy triggers on at least 1% of bars — enough to be useful, configurable to taste.

**Primary evaluation metric: Risk-Coverage curve**

Plot directional accuracy vs coverage α:

```
accuracy
  |  ↑ ideal: steep drop-off
  |    (top 1% is very accurate)
  |
  |          flat = model has no selectivity
  |_________________________ coverage (0% → 100%)
```

A good model has a steep curve: top 1% of confident predictions are highly accurate; accuracy degrades gracefully as coverage increases toward 100%.

---

## Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| NLL per horizon | `(r-μ)²/σ² + log σ²` | Lower than constant-σ baseline |
| Directional accuracy at top `min_coverage` | `sign(μ) == sign(r)` on top-SNR bars | > 55% |
| Calibration | % correct when model says P=90% | ~90% (reliability diagram) |
| Risk-Coverage AUC | Area under acc vs coverage curve | > flat (random confidence) |
| IC at top coverage | `Pearson(μ, r)` on signal bars | > 0.05 |

**Baseline to beat:**
- Zero predictor: `NLL = log(Var(r_h))` — optimal constant, no model needed
- Persistence: predict same direction as last bar
- Random confidence: no selectivity → flat risk-coverage curve

---

## Development Data: Smallest File

```
binance_aaveusdt_20250105_0000_to_20250105_0100.parquet
117KB | 3855 raw tick trades | 1 hour of AAVE/USDT
→ ~3600 1-second bars after resampling
→ ~3430 GRU sequences (lookback=60)
→ Train: ~2744  |  Val: ~686
```

One training epoch takes milliseconds. This is the correct file for fast iteration.

**Caveat:** 1 hour = one market regime. Validate final model on a second small file
(`binance_axsusdt_20251011_2359_to_20251015_2359.parquet`, 4 days) before drawing conclusions.

---

## Implementation Plan

### Step 1 — Overfit Sanity Check (prerequisite)
Train current GRU on tiny file with no early stopping (100 epochs).
Confirm `train_loss → 0` (GRU can memorize data). If it can't, fix pipeline first.

### Step 2 — Switch to Log Return Targets + σ Head + NLL Loss
- Change `create_target()` to use log return: `log(close[t+h] / close[t])`
- Change `GRUPredictor` to output `(μ, σ)` for the existing 60s target
- Replace `nn.MSELoss` with Gaussian NLL
- Confirm train_loss decreases and σ is calibrated (not all near 0 or all near ∞)

### Step 3 — Multi-Horizon Heads
Add 5 `(μ, σ)` heads for horizons `[1, 5, 10, 30, 60]`.
Update `create_target()` to create 5 target columns.
Sum NLL losses across horizons.

### Step 4 — SNR + Risk-Coverage Evaluation
Post-training, compute SNR distribution on val set.
Plot risk-coverage curve. Compare against random baseline.
Check accuracy at top `min_coverage` fraction.

### Step 5 — Auxiliary Max/Min Head (optional)
Add head predicting `(future_max_log_return, future_min_log_return)` over the horizon window.
Gives model information about range — useful for vol regime detection and stop-loss logic.

---

## Config Changes Needed

```yaml
stage1:
  data_pattern: "binance_aaveusdt_20250105_0000_to_20250105_0100.parquet"
  max_rows: null
  bar_seconds: 1
  horizons: [1, 5, 10, 30, 60]   # multi-horizon output heads
  lookback: 60
  epochs: 100
  patience: 15
  batch_size: 64
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  lr: 0.001
  loss: gaussian_nll              # proper scoring rule, replaces mse
  target: log_return              # log(close[t+h]/close[t]), replaces simple return
  use_aux_head: false             # optional future max/min head
  min_coverage: 0.01              # configurable: fraction of bars to signal on (default 1%)
```

---

## What This Unlocks for Stage 2

Stage 1 now produces per-bar:
- `(μ_h, σ_h)` per horizon → full distribution over future returns
- `SNR` confidence → gate entries: only trade when SNR > τ
- `signal_direction` = sign(μ) at highest-SNR horizon → direct long/short suggestion

Stage 2 becomes simpler: a policy that converts a probabilistic signal into position sizing.

If `SNR < τ` → stay flat (free). If `SNR ≥ τ` → trade in direction of `sign(μ)` with size proportional to `SNR`.

As discussed: **"stage 2 (policy layer) is trivial math"** — once Stage 1 correctly models the distribution.
