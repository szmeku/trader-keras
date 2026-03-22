# Stage 1 ML Pipeline — E2E Flow

## High-Level Pipeline

```mermaid
flowchart LR
    subgraph D["DATA"]
        direction TB
        D1("Parquet Files<br/>(trades)")
        D2("OHLCV Bars")
        D3("~35 Features<br/>+ Target")
        D1 --> D2 --> D3
    end

    subgraph P["PREP"]
        direction TB
        P1("Train/Val Split<br/>80/20 temporal")
        P2("Z-Score Norm<br/>stats from train only")
        P3("Sliding Windows<br/>+ StridedLoader")
        P1 --> P2 --> P3
    end

    subgraph T["TRAIN"]
        direction TB
        T1("GRU / Transformer<br/>forward pass")
        T2("Loss + Backprop<br/>+ early stopping")
        T3("Checkpoint<br/>weights + norm stats")
        T1 --> T2 --> T3
    end

    subgraph E["EVAL"]
        direction TB
        E1("Normalize inputs<br/>with saved stats")
        E2("Inference<br/>+ denorm preds")
        E3("40+ Metrics")
        E1 --> E2 --> E3
    end

    D --> P --> T --> E

    style D fill:#e1f0ff,stroke:#4a90d9
    style P fill:#fff3e0,stroke:#f5a623
    style T fill:#e8f5e9,stroke:#4caf50
    style E fill:#fce4ec,stroke:#e91e63
```

---

## Detailed Pipeline

### 1. Data Loading & Feature Engineering

```mermaid
flowchart TD
    RAW["<b>Raw Trades</b><br/><i>data/loader.py</i><br/>timestamp, price, amount, side"]
    RESAMP["<b>Resample to Bars</b><br/><i>data/resampler.py</i><br/>Numba-accelerated OHLCV binning"]
    OHLCV["open, high, low, close, volume, buy_ratio"]

    RAW --> RESAMP --> OHLCV

    subgraph FEAT["Feature Engineering — data/features.py"]
        direction LR
        subgraph COL1[" "]
            direction TB
            F1["<b>Base</b><br/>returns, log_returns<br/>volatility *, volume_ma *<br/>volume_ratio, close_pos"]
            F2["<b>Log Space</b><br/>log_volume<br/>log_volatility<br/>log_volume_ma"]
        end
        subgraph COL2[" "]
            direction TB
            F3["<b>Rolling</b><br/>dist_ma_5/10/20/50 *<br/>mom_5/10/20 *, mom_*_sign<br/>vol_regime *"]
            F4["<b>Lagged & Derived</b><br/>ret_lag1/2/3/5, log_ret_lag*<br/>log_tr *, price_accel *<br/>time_in_day_counter"]
        end
    end

    OHLCV --> FEAT
    LEAK["<b>* shifted by 1 bar</b> — no lookahead leakage"]
    FEAT -.-> LEAK

    TGT["<b>Target Creation</b><br/>target = log( close[t+h] / close[t] )<br/>multi-horizon: one column per h"]
    FEAT --> TGT

    style RAW fill:#e1f0ff,stroke:#4a90d9
    style RESAMP fill:#e1f0ff,stroke:#4a90d9
    style OHLCV fill:#e1f0ff,stroke:#4a90d9
    style FEAT fill:#fff8e1,stroke:#f5a623
    style LEAK fill:#fff,stroke:#999,stroke-dasharray:5 5
    style TGT fill:#e8f5e9,stroke:#4caf50
    style COL1 fill:#fff8e1,stroke:none
    style COL2 fill:#fff8e1,stroke:none
```

### 2. Normalization & Sequence Building

```mermaid
flowchart TD
    DF["DataFrame with ~35 features + target + segment_id"]

    SPLIT["<b>Time-Series Split</b><br/>train = first 80% | val = last 20%<br/>no shuffle, preserves temporal order"]

    STATS["<b>Compute Stats (train only)</b><br/><i>sequences.py — O(n) cumsum trick</i><br/>x_mean, x_std — per feature<br/>y_std — per horizon"]

    ZNORM["<b>Z-Score Normalize (both splits)</b><br/><i>standard.py — applied in-place</i><br/>features = (features - x_mean) / x_std<br/>targets = targets / y_std<br/><i>Note: no y_mean subtraction</i>"]

    SEQ["<b>SequenceDataset</b><br/><i>sequences.py</i><br/>Sliding windows of (lookback, n_features)<br/>Segment-aware: no cross-asset windows<br/>Stride for subsampling"]

    LOAD["<b>StridedLoader</b><br/><i>VRAM-aware batching</i><br/>Small data: all on GPU, as_strided() zero-copy<br/>Large data: superbatch streaming from pinned CPU"]

    DF --> SPLIT --> STATS --> ZNORM --> SEQ --> LOAD

    style DF fill:#e1f0ff,stroke:#4a90d9
    style SPLIT fill:#fff3e0,stroke:#f5a623
    style STATS fill:#fff3e0,stroke:#f5a623
    style ZNORM fill:#fff3e0,stroke:#f5a623
    style SEQ fill:#e8f5e9,stroke:#4caf50
    style LOAD fill:#e8f5e9,stroke:#4caf50
```

### 3. Training Loop

```mermaid
flowchart TD
    BATCH["Batch: x(B, lookback, n_feat) + y(B,)"]

    subgraph FWD["Forward Pass"]
        direction LR
        GRU["<b>GRU</b><br/>x → GRU layers<br/>→ last hidden state<br/>→ FC → pred"]
        TF["<b>Transformer</b><br/>x → Linear proj + pos enc<br/>→ TransformerEncoder<br/>→ mean pool → FC → pred"]
    end

    LOSS["<b>Loss</b><br/>Deterministic: MSE / L1 / Huber<br/>Probabilistic: gaussian_nll(mu, sigma, target)<br/>Optional: magnitude weighting (alpha)"]
    BACK["<b>Backward + Optimize</b><br/>clip_grad_norm -> Adam step<br/>LR scheduler: cosine / plateau / step / onecycle"]
    VAL["<b>Validation</b><br/>Same forward pass, no_grad<br/>Compute val_loss + overfit_ratio"]

    ESTOP{"val_loss<br/>improved?"}
    SAVE["<b>Save Best Checkpoint</b><br/>model + x_mean + x_std + y_std<br/>+ feature_cols + arch + config"]
    PATIENCE["Increment patience counter"]
    DONE["Training complete"]

    BATCH --> FWD --> LOSS --> BACK --> VAL --> ESTOP
    ESTOP -->|Yes| SAVE --> BATCH
    ESTOP -->|No| PATIENCE
    PATIENCE -->|"patience < max"| BATCH
    PATIENCE -->|"patience >= max"| DONE

    style BATCH fill:#e1f0ff,stroke:#4a90d9
    style FWD fill:#e8f5e9,stroke:#4caf50
    style LOSS fill:#fff3e0,stroke:#f5a623
    style BACK fill:#fff3e0,stroke:#f5a623
    style VAL fill:#e8f5e9,stroke:#4caf50
    style ESTOP fill:#fce4ec,stroke:#e91e63
    style SAVE fill:#e8f5e9,stroke:#4caf50
    style PATIENCE fill:#fce4ec,stroke:#e91e63
    style DONE fill:#f3e5f5,stroke:#9c27b0
    style GRU fill:#e8f5e9,stroke:none
    style TF fill:#e8f5e9,stroke:none
```

### 4. Evaluation & Metrics

```mermaid
flowchart TD
    CKPT["<b>Checkpoint .pt file</b><br/>model weights + x_mean, x_std, y_std"]
    DATA["<b>Reload Data</b><br/>Same load_data() pipeline<br/>-> train_df, val_df<br/><i>Must match training params<br/>or split boundary differs</i>"]

    REBUILD["Rebuild model from state_dict<br/>Set to eval mode"]

    NORM_IN["<b>Normalize inputs (from checkpoint)</b><br/>feats_norm = (feats - x_mean) / x_std<br/><i>Same stats as training, not recomputed</i>"]
    INFER["<b>Inference (no_grad)</b><br/>preds_norm = model(feats_norm)<br/>Probabilistic: extract mu, discard sigma"]
    DENORM["<b>Denormalize predictions</b><br/><i><b>preds = preds_norm * y_std</b></i>"]
    RAW_TGT["<b>Raw targets</b><br/>Never normalized at eval time<br/>targets = eval_targets[window_ends]"]

    CKPT --> REBUILD
    DATA --> NORM_IN
    REBUILD --> NORM_IN --> INFER --> DENORM

    subgraph M["Metrics — eval_metrics.py + simp_metrics.py"]
        direction LR
        subgraph M1[" "]
            direction TB
            MA["<b>Direction</b><br/>dir_accuracy<br/>dir_acc_p50..p99<br/>weighted_dir_acc"]
            MB["<b>Statistical</b><br/>pearson, spearman<br/>R-squared"]
            MC["<b>Magnitude</b><br/>MAE by percentile<br/>calibration ratio"]
        end
        subgraph M2[" "]
            direction TB
            MD["<b>Trading PnL</b><br/>profit_on_sign<br/>profit_with_filter"]
            ME["<b>Simplified</b><br/>simp_pred_both/long/short<br/>simp_ratio vs oracle<br/>per |pred| percentile"]
            MF["<b>Backtest</b><br/>leveraged, non-overlapping<br/>TP at entry*exp(pred)<br/>winrate, avg_pnl"]
        end
    end

    DENORM --> M
    RAW_TGT --> M

    LOG["Log to W&B<br/>train metrics + val__ prefixed metrics"]
    M --> LOG

    style CKPT fill:#e8f5e9,stroke:#4caf50
    style DATA fill:#e1f0ff,stroke:#4a90d9
    style REBUILD fill:#fff3e0,stroke:#f5a623
    style NORM_IN fill:#fff3e0,stroke:#f5a623
    style INFER fill:#e8f5e9,stroke:#4caf50
    style DENORM fill:#fce4ec,stroke:#e91e63
    style RAW_TGT fill:#e1f0ff,stroke:#4a90d9
    style M fill:#f3e5f5,stroke:#9c27b0
    style M1 fill:#f3e5f5,stroke:none
    style M2 fill:#f3e5f5,stroke:none
    style LOG fill:#fff3e0,stroke:#f5a623
```

---

## Normalization Summary

| Stage | What | How | Where |
|---|---|---|---|
| Feature creation | volatility, MAs, momentum, vol_regime | log1p(), ratios, log(close/MA) | `features.py` |
| Leakage prevention | volatility, volume_ma, MAs, mom, vol_regime, log_tr, price_accel | np.roll(x, 1) / shift(1) | `features.py:176-230` |
| Input normalization | All feature columns | z-score: (x - x_mean) / x_std | `standard.py:129` |
| Target normalization | log-return targets | **scale only: y / y_std** (no mean) | `standard.py:131` |
| Stats source | x_mean, x_std, y_std | Computed from **TRAIN split only** | `sequences.py:54-91` |
| Eval normalization | Eval features | Same z-score with **checkpoint stats** (not recomputed) | `stage1_eval.py:56` |
| Denormalization | Predictions at eval | **preds * y_std** | `stage1_eval.py:68` |

> **Caveat:** At eval time, data is reloaded fresh via `load_data()`. The normalization stats (x_mean, x_std, y_std) always come from the checkpoint (identical to training). However, the train/val **split boundary** depends on which data is loaded — if `data_pattern`, `load_limit`, or the parquet files differ from training, the val set will contain different rows.

## Data Shape at Each Stage

```
Raw trades:     (N_trades, 4)      timestamp, price, amount, side
OHLCV bars:     (N_bars, 7)        timestamp, open, high, low, close, volume, buy_ratio
After features: (N_bars', ~35)     OHLCV + indicators  (shorter: dropna on rolling windows)
After target:   (N_bars'', ~36)    + target col         (shorter: forward shift drops tail)
After split:    train 80% / val 20%  (temporal, no shuffle)
Sequences:      (B, lookback, n_features) + (B,) target
Model output:   (B,) or (B, n_horizons) or ((B,n_h), (B,n_h)) if probabilistic
```

## Key Files

| Component | File |
|---|---|
| Data loading | `crypto_trader/data/loader.py` |
| Bar resampling | `crypto_trader/data/resampler.py` |
| Feature engineering | `crypto_trader/data/features.py` |
| Feature column defs | `crypto_trader/constants.py` |
| Sequence dataset + norm | `crypto_trader/models/sequences.py` |
| Training orchestration | `crypto_trader/models/standard.py` |
| Training loop | `crypto_trader/models/training.py` |
| Loss functions | `crypto_trader/models/losses.py` |
| GRU / Transformer | `crypto_trader/models/gru.py` |
| Stage 1 dispatcher | `crypto_trader/trainer/stage1.py` |
| Evaluation | `crypto_trader/eval/stage1_eval.py` |
| Metrics | `crypto_trader/eval/eval_metrics.py` |
| Simplified metrics | `crypto_trader/eval/simp_metrics.py` |
| SNR / risk-coverage | `crypto_trader/eval/evaluation.py` (unused) |
| CLI entry point | `run.py` |
