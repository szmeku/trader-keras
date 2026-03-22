[[docs/overfit_rl]]
# Overfit Plan

## Core Vision (from user)

**Goal**: predict stuff super useful during trading — not just "where price ends up" but actionable levels for limit orders.

Key ideas:
- **Volume-weighted max/min** — not absolute max/min but slightly lower/higher levels with enough volume to actually execute. Tells us realistic tradable bounds for limit orders.
- also regular max/min so we know to be careful with momental max min that can take out our limit orders
- **Buy/sell pressure** — we have `is_buyer_maker` in raw trades, giving literal ratio of buys vs  sells. Currently only used as trade-count ratio (`buy_ratio`). Should weight by volume instead.
- **Predict volume itself** — useful signal for confidence / liquidity.
- **All prices in log-return form**
- **One network, multiple horizon heads** — shared backbone is efficient, heads are cheap linear projections. Multi-task across horizons acts as regularization.
- **Use richer trade data** — now we have `price`, `quantity`, `is_buyer_maker` per trade but collapse to just OHLCV + buy_ratio. Much information lost

---

## Phase 1: Prove memorization works (NOW)

Simplify everything to debug why the model can't memorize:

- **Single horizon** (e.g. 60s) — one output head, one target value
- **Single value** — just μ, no σ (MSE loss, not Gaussian NLL)
- **Tiny sample** — 20-50 sequences, model should get loss → 0
- **No regularization** — dropout=0, no weight decay
- If this fails → architecture bug. If it works → scale up gradually.

Scaling ladder:
1. 20 samples, MSE, single horizon → must memorize
2. 20 samples, probabilistic (μ,σ), single horizon → must memorize
3. 20 samples, probabilistic, multi-horizon → must memorize
4. Full dataset, probabilistic, multi-horizon → overfit (train loss < baseline)

---

## Phase 2: Richer features from raw trade data

Current resampler produces: OHLCV + `buy_ratio` (trade count ratio).
Raw data has: `price`, `quantity`, `is_buyer_maker` per trade.

New features to add per bar:
- **VWAP** — volume-weighted average price (more representative than close)
- **Buy volume / sell volume** — weight by quantity, not just trade count
- **Order flow imbalance** — `(buy_vol - sell_vol) / total_vol`
- **Trade count** — liquidity proxy
- **Large trade indicator** — trades above 95th percentile quantity
- **Volume quantiles** — price levels where X% of volume traded

---

## Phase 3: Richer targets

Current target: `log(close[t+h] / close[t])` — where price ends up.

New targets for limit order utility:
- **Volume-weighted high** within `[t, t+h]` — realistic ask level (not absolute max, but max weighted by volume = executable level)
- **Volume-weighted low** within `[t, t+h]` — realistic bid level
- **Max/min reachable price** — `log(high[t:t+h] / close[t])`, `log(low[t:t+h] / close[t])`
- **Volume prediction** — `log(sum(volume[t:t+h]))` as auxiliary head (confidence/liquidity signal)
- **Price quantiles** — price at which X% of future volume trades (order book depth proxy)

All targets in log-return form relative to current close.

---

## Phase 4: Multi-horizon + probabilistic

Once single-head memorization is proven:
- Re-enable multi-horizon heads (shared backbone, per-horizon linear projections)
- Re-enable probabilistic output (μ, σ) with Gaussian NLL
- σ via softplus (maps ℝ→ℝ⁺, smooth, stable) + floor 0.01
- Magnitude weighting (`|return|^alpha`) to focus on big moves

---

## Phase 5: Architecture refinements

- Compare GRU vs Transformer on memorization speed and final loss
- Tune: hidden_size, num_layers, n_heads, feedforward dim
- Positional encoding: learnable vs sinusoidal vs RoPE
- Pooling: mean pool vs last-token vs attention pooling
- Consider: patch embedding (group N consecutive bars into one token) to reduce sequence length

---

## Notes

- `buy_ratio` already exists as feature (trade count based) — enhance to volume-weighted
- Input normalization (x_mean, x_std from train set) is critical for Transformer
- Target normalization (y_std per horizon) keeps loss scale uniform across horizons
- Checkpoint saves normalization stats for inference consistency
- softplus for σ: network output is unbounded ℝ, softplus maps to ℝ⁺ (σ must be positive). It's σ not σ² — the loss uses σ² explicitly as `sigma**2`.
