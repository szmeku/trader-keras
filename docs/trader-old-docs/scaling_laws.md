# Scaling Laws: Data-to-Parameter Ratios

How much data do you need per model parameter? Depends on architecture.

## Our metric: `data_param_ratio`

```
data_param_ratio = n_sequences * lookback / n_params
```

Counts total **row-observations** the model sees during training. A sequence of lookback=100 carries 10x more information than lookback=10.

## Ratios by Architecture

| Architecture | Min ratio | Recommended | Source |
|---|---|---|---|
| **GRU/LSTM** | 10:1 | **30-50:1** | MLP heuristics; no formal RNN scaling law |
| **MLP** | 10:1 | **30-50:1** | Baum & Haussler generalization bounds |
| **Transformer (compute-optimal)** | ~20 tokens/param | **20:1** | Chinchilla (Hoffmann 2022) |
| **Transformer (inference-optimal)** | 100:1 | **200-2000:1** | LLaMA-3, post-Chinchilla |

## Key Papers

- **Kaplan et al. 2020** ([arxiv](https://arxiv.org/abs/2001.08361)): Original LLM scaling laws. ~3 tokens/param — "train big, stop early". Larger models are more sample-efficient.
- **Chinchilla / Hoffmann 2022** ([arxiv](https://arxiv.org/abs/2203.15556)): Revised to **~20 tokens/param**. Model size and data should scale equally — doubling compute means doubling both.
- **Post-Chinchilla overtraining**: When optimizing for inference cost (smaller model, more data):
  - LLaMA-1 65B: ~20 tokens/param
  - LLaMA-2 70B: ~30 tokens/param
  - LLaMA-3 8B: ~1,875 tokens/param (15T tokens)

## Notes for Our GRU Predictor

- GRUs have ~25-33% fewer params than LSTMs (no output gate) — more data-efficient
- With overlapping windows (stride < lookback), samples are correlated — effective independent count is lower than raw count
- **Target: 30-50x params** for generalization. During overfit phase, we intentionally go below this.
- Current configs (lookback=10, stride=10, hidden=30, 2 layers): ~9.7k params, ~47.5k row-observations → ratio ~4.9 (deep overfit territory)

## Practical Implications

| Ratio | Regime | Expected behavior |
|---|---|---|
| < 1:1 | Extreme overfit | Model memorizes; train loss → 0, no generalization |
| 1-10:1 | Overfit | Good for debugging signal; model should fit train perfectly |
| 10-30:1 | Transition | Some generalization, still risks overfitting |
| 30-50:1 | Sweet spot | Good generalization for RNNs/MLPs |
| > 50:1 | Data-rich | Diminishing returns, consider scaling model up |
