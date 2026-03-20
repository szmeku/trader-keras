# Sanity Check (2025-01-29) — HISTORICAL

> **Note:** This document is from the pre-refactoring codebase. Module paths, feature counts, and commands below are outdated. Kept for reference only — see [[architecture]] for current system details.

**Dataset:** binance_aaveusdt_20250105_0000_to_20250106_0000.parquet
**Data Size:** 75,905 trades → 22,074 bars (1s resolution)

---

## Stage 1: GRU Predictor ✓

| Detail | Value |
|---|---|
| Train sequences | 17,599 (60-step lookback, 19 features) |
| Val sequences | 4,355 |
| Epochs | 30 |
| Best val loss | 0.000003 (MSE) |
| Overfitting | Yes (epoch 27: train 0.000020 vs val 0.000045) |
| Output | `gru_predictor.pt` (497KB) |

**Validation sanity:**
- MSE on 1000 val samples: 0.000002
- Predictions: mean=-0.000167, std=0.000459
- Actual returns: mean=-0.000002, std=0.001383
- Correlation: -0.0570 (weak, expected for short training)

**Conclusion:** GRU trains successfully and can overfit as expected.

---

## Stage 2: RL with Frozen GRU ✓

| Detail | Value |
|---|---|
| Frozen GRU params | 41,345 |
| Trainable policy params | 4,813 (10.4% of total) |
| Total params | 46,158 |
| Output | `gru_rl_model.pt` (187KB) |

**Quick test (10 episodes on subset):**
- Data: 1,098 steps × 13 features
- Exploration: 30% epsilon-greedy
- Trades per episode: ~750
- Fee impact: 0.1% × 750 = 75% in fees alone

**Backtest on validation:**

| Metric | Value |
|---|---|
| Model return | -50.31% |
| Buy & Hold | -0.45% |
| Alpha | -49.86% |
| Trades | 692 |
| Total fees | $4,977.52 |
| Final equity | $4,968.96 |

**Analysis:** Underperforms due to insufficient training (10 episodes), high exploration (30%), excessive trading (692 trades), and fee erosion (49.8% of equity).

---

## Key Findings

### What Works
1. Data loading with no future leakage
2. Feature engineering
3. GRU predictor training (Stage 1)
4. Frozen GRU + RL policy (Stage 2)
5. Realistic 0.1% taker fees
6. Backtesting with metrics
7. Model save/load

### Limitations at Time of Check
1. High trading frequency (needs position holding training)
2. Insufficient training episodes (10 vs recommended 100+)
3. Weak predictive correlation (-0.057, needs more Stage 1 epochs)

---

## Commands Used (outdated — see [[README]] for current CLI)

```bash
# These commands are from the old codebase and no longer work:
# python -c "from crypto_trader.gru_predictor import main; main()"
# python -c "from crypto_trader.gru_policy import sanity_check; sanity_check()"
# python run.py gru-train --quick-test

# Current equivalents:
python run.py train config.yml                  # Full pipeline
pytest tests/test_rolemodel.py -s              # Reproducibility sanity check (~30s on GPU)
# backtest command removed (2026-03-16) — needs rewrite for icmarkets_env
```

**Conclusion:** All stages verified working. System ready for extended training runs.
