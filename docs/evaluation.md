# Evaluation

Stage 1 prediction quality metrics and CLI usage.

**See also:** [[config_reference]] (training config), [[architecture]] (system design), [[README]] (install & CLI)

---

## Re-evaluating a Model

Set `stage1.train: false` in a config with `existing_wandb_run` to re-evaluate without training:

```yaml
existing_wandb_run: radiant-sky-305
stage1:
  train: false
logging:
  provider: [wandb, console]  # wandb → updates old run; remove wandb for local-only eval
```

```bash
python run.py train eval_config.yml
```

**Behavior depends on loggers:**
- **wandb in `logging.provider`** + `existing_wandb_run` → **resumes the original W&B run** and updates its summary with `stage1/eval/*` metrics. No new run created.
- **wandb NOT in `logging.provider`** → evaluates locally (console output only), no W&B interaction.

**Flow:**
1. Fetches full config from the W&B run
2. Deep-merges local YAML overrides on top (e.g. `stage1.batch_size`, `data.load_limit`)
3. Downloads model artifact from the run
4. Runs inference on **both train and val splits**
5. If wandb logger active: updates the existing run's summary
6. Prints results

Run can be specified as: display name (`radiant-sky-305`), run ID (`q0dnhvew`), or full path (`entity/project/id`).

### Evaluate a local model

```yaml
stage1:
  train: false
  model: path/to/gru_predictor.pt
logging:
  provider: console
```

Loads checkpoint, reads config from it (lookback, horizons, stride, bar_seconds), runs inference on **both train and val splits**, prints all metrics.

### How it works

Eval reuses the **exact same data pipeline** as training:
- `load_data()` → time-series split (using `train_ratio` from config) → evaluates on **both** splits
- Same normalization (x_mean, x_std from checkpoint)
- Same sequence construction (SequenceDataset with lookback + stride)
- Predictions denormalized by y_std before computing metrics

**Train metrics** (no prefix) verify the model learned its training data. **Val metrics** (`val__` prefix) measure generalization to unseen data.

```
stage1/eval/direction_accuracy      ← train split
stage1/eval/val__direction_accuracy  ← val split
```

### Deterministic eval

By default, cuDNN GRU forward has tiny non-determinism across runs. To get exact reproducibility:

```yaml
stage1:
  eval_deterministic: true
```

This sets `torch.use_deterministic_algorithms(True)` during eval. May be slower.

---

## Metrics Legend

### Direction Metrics

| Metric | Key | Description |
|---|---|---|
| Direction accuracy | `direction_accuracy` | Fraction where `sign(pred) == sign(target)`. 0.5 = random. Includes near-zero targets where sign is noise |
| Dir acc by percentile | `dir_acc_p{50,75,90,95,99}` | Direction accuracy on samples where `\|pred\| >= percentile`. Filters by prediction confidence — usable as trade filter at inference time |
| Weighted dir accuracy | `weighted_dir_acc` | Direction accuracy weighted by `\|target\|`. Getting big moves right matters more than small ones |

> **Note:** `direction_accuracy` can be <100% even for a perfect model because near-zero targets have arbitrary sign. The percentile variants (`dir_acc_p50+`) filter these out.

### Regression Metrics

| Metric | Key | Description |
|---|---|---|
| R² | `r2` | `1 - SS_res/SS_tot`. 1.0 = perfect, 0 = predicts mean, negative = worse than mean |
| Pearson correlation | `pearson_corr` | Linear correlation between pred and target. 1.0 = perfect linear relationship |
| Spearman correlation | `spearman_corr` | Rank correlation. Measures monotonic relationship (robust to outliers) |
| MAE by percentile | `mae_p{50,75,90,95,99}` | Mean absolute error on samples above each `\|target\|` percentile |
| Calibration ratio | `calibration_ratio` | `mean(\|pred\|) / mean(\|target\|)`. 1.0 = predicted magnitude matches actual |
| Pred magnitude | `pred_magnitude` | Average `\|pred\|` |
| Target magnitude | `target_magnitude` | Average `\|target\|` |

### Profit Metrics

| Metric | Key | Description |
|---|---|---|
| Profit on sign | `profit_on_sign` | `sum(sign(pred) * target)` — ideal PnL if trading every bar on predicted direction (no fees) |
| Profit per bar | `profit_on_sign_per_bar` | Same, averaged per bar. Positive = directional edge exists |
| Filtered PnL | `filtered_pnl_p{50,75,90,95}` | `mean(sign(pred) * target)` only when `\|pred\| >= percentile`. Bets fixed $1 long/short per trade (direction only, not magnitude-weighted). Tests: "does trading only on high-magnitude predictions improve PnL?" |

### Simplified Trading Metrics

Fast, fee-free directional edge measurement. Compares model-guided trading against a perfect oracle baseline using log returns. All keys prefixed `simp_`.

**Source:** `crypto_trader/eval/simp_metrics.py`

#### Baselines (oracle — knows the future)

| Metric | Key | Formula | Description |
|---|---|---|---|
| Both | `simp_baseline_both` | $\sum_{i} \lvert r_i \rvert$ | Perfect oracle that captures every move (long when $r>0$, short when $r<0$) |
| Long only | `simp_baseline_long` | $\sum_{i} \max(0,\; r_i)$ | Oracle that only goes long |
| Short only | `simp_baseline_short` | $\sum_{i} \max(0,\; -r_i)$ | Oracle that only goes short |

where $r_i$ is the true log return at bar $i$. Note: $\text{baseline\_long} + \text{baseline\_short} = \text{baseline\_both}$.

#### Pred-based (uses prediction sign, counts actual return)

| Metric | Key | Formula | Description |
|---|---|---|---|
| Both | `simp_pred_both` | $\sum_{i} \operatorname{sign}(\hat{r}_i) \cdot r_i$ | Go long when $\hat{r}>0$, short when $\hat{r}<0$, earn actual $r$ |
| Long only | `simp_pred_long` | $\sum_{i:\,\hat{r}_i > 0} r_i$ | Only enter long positions |
| Short only | `simp_pred_short` | $\sum_{i:\,\hat{r}_i < 0} (-r_i)$ | Only enter short positions |
| Per-bar variants | `simp_pred_{both,long,short}_per_bar` | $\frac{1}{N}\sum\ldots$ | Normalized by total number of bars |

where $\hat{r}_i$ is the predicted log return.

#### Diffs & Ratios (comparison to oracle)

Both use log returns, so the math is clean either way:
- **Diffs** (`pred − baseline`): absolute shortfall in log-return units. 0 = oracle.
- **Ratios** (`pred / baseline`): fraction of available return captured. 1.0 = oracle, 0 = random, negative = anti-correlated. Scale-free — comparable across bar sizes, dataset lengths, and volatility regimes.

| Metric | Key | Formula | Interpretation |
|---|---|---|---|
| Diff both | `simp_diff_both` | `simp_pred_both − simp_baseline_both` | 0 = oracle, negative = return left on the table |
| Diff long | `simp_diff_long` | `simp_pred_long − simp_baseline_long` | Same, long side only |
| Diff short | `simp_diff_short` | `simp_pred_short − simp_baseline_short` | Same, short side only |
| Ratio both | `simp_ratio_both` | `simp_pred_both / simp_baseline_both` | 1.0 = oracle, 0 = random |
| Ratio long | `simp_ratio_long` | `simp_pred_long / simp_baseline_long` | Same, long side only |
| Ratio short | `simp_ratio_short` | `simp_pred_short / simp_baseline_short` | Same, short side only |

#### Percentile sub-metrics (by $\lvert\hat{r}\rvert$ — no future leakage)

Filters to top $k\%$ predictions by $\lvert\hat{r}\rvert$ magnitude, then computes the same pred metrics on that subset. Tests: *"does the model perform better when it's more confident?"*

For each percentile threshold $q \in \{50, 75, 90, 95, 99\}$ (i.e. top 50%, top 25%, top 10%, top 5%, top 1%):

| Metric | Key | Formula |
|---|---|---|
| Both | `simp_p{q}_both` | $\sum_{i:\,\lvert\hat{r}_i\rvert \geq Q_q} \operatorname{sign}(\hat{r}_i) \cdot r_i$ |
| Long / Short | `simp_p{q}_long`, `simp_p{q}_short` | Same split as above, on the filtered subset |
| Per-bar | `simp_p{q}_{both,long,short}_per_bar` | Normalized by number of bars in the subset |
| Baseline | `simp_p{q}_baseline_both` | $\sum_{i:\,\lvert\hat{r}_i\rvert \geq Q_q} \lvert r_i \rvert$ — oracle return on same subset |
| Diff | `simp_p{q}_diff_{both,long,short}` | `pred − baseline` on the percentile-matched subset |
| Ratio | `simp_p{q}_ratio_{both,long,short}` | `pred / baseline` on the percentile-matched subset |

where $Q_q$ is the $q$-th percentile of $\lvert\hat{r}\rvert$. Both diffs and ratios compare against the oracle **on the same bars** the model chose to trade (not the full dataset).

> **Reading the results:** If `simp_p95_both_per_bar` >> `simp_pred_both_per_bar`, the model's confidence is well-calibrated — higher conviction predictions produce better trades. `simp_ratio_both` close to 1.0 (or `simp_diff_both` close to 0) means strong directional edge.

#### Multi-asset behavior

When multiple assets are loaded (e.g. `data: "*.parquet"`), `load_data()` assigns each file a `segment_id` and concatenates them. The simplified metrics operate on **flattened 1D arrays** — all assets are mixed together with no per-asset grouping.

**What this means:**

- **Oracle is aggregated** — `simp_baseline_both = sum(|targets|)` sums absolute returns across all assets. High-volatility coins dominate the baseline.
- **No per-asset breakdown** — `simplified_trading_metrics()` doesn't receive `segment_ids`. You can't tell from these metrics which coin the model performs well on.
- **Percentile thresholds mix assets** — the top-k% confidence filters treat all bars equally regardless of which coin they're from.
- **Ratios can be misleading** — a model strong on BTC but weak on BONK will show a blended ratio that hides the per-asset picture. To get per-coin ratios, filter by `segment_id` post-hoc.

> **Note:** The sequential backtest (`backtest_metrics.py`) *does* respect segment boundaries — it won't open a trade in BTC and close it in ETH. The simp metrics don't need this since they're per-bar (no multi-bar positions).

### Sequential Backtest Metrics

Non-overlapping trade simulation (`sequential_backtest`). Walks predictions in chronological order; when `|pred| >= threshold` and flat, opens a position with a **limit TP** at the model's predicted price level. Scans bars causally (longs check high ≥ TP, shorts check low ≤ TP). If TP isn't hit, exits at horizon-end close. Skips `horizon_bars` after each entry (no overlapping positions). Never trades across segment boundaries.

Each metric has a `_pXX` suffix = prediction confidence threshold (only top `100-XX%` strongest predictions trade):

| Metric | Key | Description |
|---|---|---|
| Trades | `bt_trades_p{50,75,90,95}` | Number of trades taken at this threshold. Fewer = stricter filter |
| Total PnL | `bt_pnl_p{50,75,90,95}` | Sum of per-trade log-return PnL (entry→TP or entry→horizon close) |
| Avg PnL | `bt_avg_pnl_p{50,75,90,95}` | `bt_pnl / bt_trades` — average PnL per trade |
| Win rate | `bt_winrate_p{50,75,90,95}` | Fraction of trades with positive PnL |
| TP hit rate | `bt_tp_rate_p{50,75,90,95}` | Fraction of trades where the limit TP was reached before horizon end |

> **Reading the results:** Ideally `bt_avg_pnl` increases and `bt_winrate` stays high as the threshold rises (p50→p95). This means the model's confidence is well-calibrated — stronger predictions lead to better trades.

---

## Training Metrics (logged to W&B)

### Per-epoch (time series)

| Key | Description |
|---|---|
| `stage1/train_loss` | Training loss (MSE, MAE, Huber, or Gaussian NLL depending on config) |
| `stage1/val_loss` | Validation loss (only when `train_ratio < 1.0`) |
| `stage1/overfit_ratio` | `val_loss / train_loss` — how much the model overfits (only when `train_ratio < 1.0`) |
| `stage1/step_loss` | Per-batch training loss (logged every `log_every` optimizer steps) |
| `stage1/baseline_loss` | Naive baseline loss (persistence model) for comparison |
| `stage1/train_loss_vs_baseline` | `train_loss / baseline_loss` — <1.0 means beating the baseline |
| `stage1/lr` | Current learning rate |
| `stage1/samples_per_sec` | Training throughput |
| `stage1/epoch_time_sec` | Wall time per epoch |
| `global_step` | Monotonic optimizer step counter (= x-axis for W&B charts) |

### End-of-training (single values)

| Key | Description |
|---|---|
| `stage1/best_val_loss` | Best validation (or train) loss achieved |
| `stage1/total_time_sec` | Total training wall time |
| `stage1/total_epochs` | Epochs completed |
| `stage1/total_steps` | Total optimizer steps |
| `stage1/early_stopping_epoch` | Epoch where early stopping triggered (if applicable) |

### W&B internal keys

| Key | Description |
|---|---|
| `_runtime` | Seconds since W&B run started (float) |
| `_step` | W&B internal step counter (mirrors `global_step`) |
| `_timestamp` | Unix epoch timestamp of the log call |
| `_wandb.runtime` | Same as `_runtime` (integer) |

These are auto-generated by W&B — not our code. Safe to ignore.

### Eval metrics (end-of-training)

Logged under `stage1/eval/*` prefix. Train metrics have no extra prefix, val metrics have `val__` prefix:

| Example key | Split |
|---|---|
| `stage1/eval/direction_accuracy` | Train |
| `stage1/eval/r2` | Train |
| `stage1/eval/val__direction_accuracy` | Val |
| `stage1/eval/val__r2` | Val |

Same metrics as the CLI eval — see [Metrics Legend](#metrics-legend) above.

---

## Stage 2: Training vs Evaluation

Stage 2 has two fundamentally different modes. They use different data strategies, different metric semantics, and should never be confused.

### Training mode (`stage2_train`)

**Data strategy:** All N envs share the **same full data range** with `random_start=True`. Each env picks a random starting point. Multiple envs may overlap on the same bars — this is intentional for exploration diversity, not coverage.

**Purpose:** maximize learning signal. Metrics are noisy per-epoch signals, not quality measurements.

| Key | Description |
|---|---|
| `stage2/train_reward` | Mean reward per env during rollout. Noisy — varies with random starts |
| `stage2/train_best_reward` | Best reward seen so far (triggers checkpoint save) |
| `stage2/train_pg_loss` | PPO policy gradient loss |
| `stage2/train_v_loss` | Value function loss |
| `stage2/train_entropy` | Policy entropy (type + params). 0.3–0.8 is healthy exploration |
| `stage2/train_type_entropy` | Action-type entropy only |
| `stage2/train_params_entropy` | Continuous params entropy only |

### Evaluation mode (`evaluate_rollout`, sequential)

**Data strategy:** Single env runs the **full data range** sequentially — one $10k account, one position at a time. No data splitting, no parallel envs.

**Capital model:** `compounding=False` (default) — balance resets to initial after each step, so position sizing stays constant regardless of accumulated PnL. The equity curve tracks cumulative rewards additively. Set `compounding=True` for realistic compounding.

**Inference:** Runs on CPU (`device="cpu"`) for batch=1 — **2.4x faster than GPU** due to eliminated kernel launch/transfer overhead. Uses fused forward pass (`eval_action_fused`) that bypasses nn.Module dispatch and Distribution object creation. `deterministic=True` uses greedy actions (argmax type, mean params) for ~30% speedup.

**Speed benchmarks** (80k 1-min bars, MixedPolicy h=256 l=3, GTX 1050 Ti):

| Mode | Steps/sec | 80k bars | vs. original |
|------|-----------|----------|--------------|
| CPU deterministic | 4300 | 19s | **5.3x faster** |
| CPU stochastic | 3300 | 24s | **4x faster** |
| GPU deterministic | 1800 | 45s | 2.2x faster |
| GPU stochastic | 1300 | 60s | 1.6x faster |
| Old parallel (64 envs, GPU) | — | ~3s | fastest (but wrong capital model) |

---

## Stage 2 Eval Metrics

Logged under `stage2/eval/*` prefix after training completes. Source: `rollout_eval.py`.

### Per-episode metrics (single-account)

These are the primary quality metrics. They answer: "if I ran this strategy on the full data with $10k, what would happen?"

| Key | Source | Description |
|---|---|---|
| `pnl_per_episode` | `sum(rewards)` | **Total PnL** — primary metric |
| `equity_per_episode` | `balance + pnl` | **Ending wallet value** |
| `return_pct` | `pnl / balance * 100` | Return % |
| `n_trades_per_episode` | `n_trades` | Total number of closed trades |

### Step-level risk metrics (from per-bar rewards)

Computed from individual bar rewards (not trade PnLs). These capture the full mark-to-market risk profile including unrealized P&L swings.

| Key | Source | Description |
|---|---|---|
| `sharpe` | `rewards / balance` | Sharpe ratio of per-bar returns |
| `sortino` | `rewards / balance` | Sortino ratio (downside-only risk) |
| `max_drawdown` | equity curve | Max peak-to-trough drawdown |

### Trade-level metrics (prefixed `trade_`)

Computed from closed trades. Prefixed with `trade_` to avoid collision with step-level metrics above.

| Key | Source | Description |
|---|---|---|
| `n_trades` | `len(trades)` | Total closed trades |
| `trade_pnl_total` | `sum(trade.pnl)` | Total realized PnL |
| `trade_commission_total` | `sum(trade.commission)` | Total commissions paid |
| `trade_avg_pnl` | `trade_pnl_total / n_trades` | Average P&L per closed trade |
| `trade_win_rate` | from `risk_metrics` | Fraction of trades with positive PnL |
| `trade_profit_factor` | from `risk_metrics` | Gross profit / gross loss |
| `trade_sharpe` | from trade PnLs | Sharpe ratio of per-trade returns (not per-bar) |
| `trade_sortino` | from trade PnLs | Sortino ratio of per-trade returns |
| `trade_max_drawdown` | from trade equity curve | Max drawdown of cumulative trade PnLs |
| `trade_return_pct` | `trade_pnl_total / balance * 100` | Total return % |

### Baselines & comparisons

| Key | Source | Description |
|---|---|---|
| `baseline_buyhold_pnl` | from bar closes | Buy-and-hold PnL |
| `baseline_buyhold_return_pct` | | Return % for buy-and-hold |
| `baseline_perfect_pnl` | `sum(\|bar-to-bar returns\|)` | Theoretical ceiling — perfect timing, zero cost |
| `baseline_perfect_return_pct` | | Return % for perfect oracle |
| `log_ratio_vs_buyhold` | `log(equity / bh_equity)` | Negative = underperforming buy-and-hold |
| `log_ratio_vs_perfect` | `log(equity / perfect_equity)` | Always negative (ceiling). Closer to 0 = better |

### Other

| Key | Description |
|---|---|
| `total_steps` | Total bars stepped |
| `overnight_holds` | Positions held through swap hour |
| `action_{hold,limit_buy,...}_pct` | Fraction of steps each action type was chosen |

### Curve metrics (logged per-step via `log_metrics`)

Logged by `log_rollout_curves()` on a custom `eval_step` axis (separate from training steps). Subsampled to max 5000 points. Single continuous equity curve for the full data range:

| Key | Description |
|---|---|
| `equity` | Per-bar equity curve |
| `reward` | Per-bar reward |

### Config params (logged at start)

| Key | Description |
|---|---|
| `stage1/train_samples` | Number of training sequences |
| `stage1/val_samples` | Number of validation sequences |
| `stage1/model_n_params` | Total model parameters |
| `stage1/data_param_ratio` | `train_samples / n_params` — <1.0 = high overfitting risk |
| `stage1/baseline_*` | Naive baseline losses (persistence, zero, mean) |
| `stage1/target_std_*` | Target standard deviation per horizon |

---

## Interpreting Results

### Perfect memorization (expected for overfit runs)

```
r2                    1.000000
direction_accuracy    0.991     ← near-zero targets have noisy sign
dir_acc_p50+          1.000000  ← filtering removes noise
mae_p*                0.000000
pearson_corr          1.000000
profit_on_sign        positive
```

### Random / no learning

```
r2                    ≈ 0 or negative
direction_accuracy    ≈ 0.50
pearson_corr          ≈ 0
profit_on_sign        ≈ 0
```

### Good generalization (goal)

```
r2                    > 0 (any positive value is signal)
direction_accuracy    > 0.52 (even small edge compounds)
dir_acc_p90+          > 0.55 (predicting big moves)
profit_on_sign        positive
filtered_pnl_p75+    > profit_on_sign_per_bar (confident preds are better)
calibration_ratio     ≈ 1.0
```
