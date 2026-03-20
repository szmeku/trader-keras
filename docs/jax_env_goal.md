# The Goal

Build a **high-performance trading environment** for RL training — targeting millions of steps per second.

## Core Problem

RL training is typically bottlenecked by the environment, not the model. Trading environments compound this because:

- **Sequential nature**: one position at a time means episodes are inherently serial. Naive parallelization (multiple envs with separate balances) is unrealistic — in reality there is one shared balance. Our batching approach (see Parallelization Strategy) avoids this: each batch is a genuinely independent episode (own starting balance, own asset, own time window), not parallel envs pretending to share one account.
- **Data transfer overhead**: inference may or may not run on GPU. If env is on CPU and model on GPU (or vice versa), moving data between them every step can dominate step time. A fully GPU-native approach eliminates this.

## Implementation Approach

**JAX/GPU-native environment** — the entire sim runs on GPU as JIT-compiled pure functions. No CPU↔GPU transfer during training. Follows the gymnax `Environment` pattern (pure functional, `jit`-able `step`/`reset`) for compatibility with PureJaxRL/Stoix.

No existing framework covers our needs. All JAX-based trading envs are LOB/HFT simulators (wrong domain). All OHLCV-based trading envs are Python/PyTorch at 1-200k steps/sec (too slow). Custom build is the only path.

> **Future opportunity:** The existing JAX LOB/HFT simulators (JAX-LOB, JaxMARL-HFT) could become relevant if we move to sub-minute resolution (tick or order-book data). Worth revisiting when we outgrow 1-minute bars.

## Parallelization Strategy

The sequential bottleneck is solved by batching across *naturally independent* dimensions — each batch is genuinely serial internally, but batches run in parallel via `vmap`:

1. **Multi-asset** — train on ~20 assets simultaneously. The agent is asset-agnostic — it always sees one position, one price series, no asset identifier in features. It doesn't know (or care) which asset it's trading. The pipeline assigns each batch a different asset's data, but from the agent's perspective every batch looks the same. Each has its own price series, own balance, own sequential history.
2. **WFO (Walk-Forward Optimization) windows** — split each asset's history into e.g. 4-month chunks. Each chunk is an independent episode with its own 10k starting balance.
3. **Data augmentation** — random perturbations to prices/spreads create synthetic variants of each window, multiplying the batch count further.

**Example:** 20 assets × 6 WFO windows × 10 augmentations = **1,200 parallel batches** — enough to saturate a GPU while every individual batch remains strictly sequential.

## Spec-Agnostic Agent Interface

The goal of training on multiple assets is to learn **universal trading patterns** — patterns that generalize across markets, not asset-specific quirks. For this to work, the agent must be completely blind to which asset it's trading. It should never be able to distinguish BTC from EURUSD from gold — not from the features, not from the observation scale, not from the action semantics.

Asset specs (contract_size, tick_value, leverage) differ across assets and would act as fingerprints if exposed. Raw balance of $10k with 1:500 forex leverage behaves very differently from $10k with 1:20 crypto leverage — the agent could learn to detect this and develop asset-specific strategies instead of general ones.

The solution: the env abstracts away all spec details. The agent operates entirely in a **normalized, spec-free "fraction of account" space**. The env translates to/from real units internally. The agent never receives the spec.

| Layer | Agent sees | Env handles internally |
|---|---|---|
| **Observations** | Precomputed features (self-relative, e.g. price ratios) + dynamic state as fractions of balance (equity/balance, unrealized_pnl/balance, margin_used/balance) | Converting position state to normalized fractions using spec + leverage |
| **Actions** | Order type + risk/margin fraction (e.g. "risk 2% of balance") | Converting fraction to correct lot size using contract_size, leverage, current price |
| **Rewards** | PnL as fraction of balance | Computing raw PnL from spec, then dividing by balance |

The agent never sees lots, contract sizes, leverage, or absolute dollar values. Every batch looks identical regardless of which asset is underneath.

## Architecture Decisions (settled)

| Decision | Value |
|---|---|
| Implementation | JAX/GPU-native, pure functional |
| Asset scope | Single asset per env; multi-asset via parallel batches |
| Position model | Netting mode — one position at a time, close-and-reverse supported |
| Action space | HOLD, LIMIT_BUY, LIMIT_SELL, STOP_BUY, STOP_SELL, CANCEL (volume expressed as balance fraction, not lots) |
| Reward function | Switchable (profit, Sharpe ratio, others) as fraction of balance — pluggable, not a speed concern |
| Episode boundaries | Fixed number of bars |
| Bar simulation | 4-tick model (OHLC order based on bar direction) — MT5-compatible |
| Framework compatibility | TBD — depends on which RL framework we choose |

## Observation Design: Minimal Inputs, Let the NN Learn

Rather than hand-crafting features (RSI, SMA, Bollinger, etc.) which would all need identical reimplementation in MQL5 for production and validation — we keep inputs minimal and self-relative, letting the NN's hidden layers discover whatever patterns matter.

**Minimal input set:**
- **Lookback window of N bars** as price ratios: close_t/close_{t-1}, high_t/close_t, low_t/close_t, open_t/close_t — inherently asset-agnostic
- **Spread** normalized by price: spread_t/close_t
- **Time encoding**: sin/cos of hour
- **Dynamic state**: equity/balance, unrealized_pnl/balance, margin_used/balance, has_position, position_side, position_duration

No hand-crafted indicators. The lookback window gives the NN raw material to learn whatever momentum/mean-reversion/volatility patterns exist. And it's trivially reproducible in MQL5 — just price ratios.

All observation values are computed automatically by the env from raw bar data + sim state. Single source of truth for feature definitions.

## Architecture-Agnostic Pipeline

The pipeline treats the model as a black box between two fixed-shape tensors:

```
env → [obs tensor] → ONNX model → [action tensor] → env
```

The env doesn't care what's inside the model. The MQL5 EA doesn't care either — it calls `OnnxRun()`. Architecture is swappable (MLP, GRU, Transformer) as long as:
- ONNX exportable
- Matching input/output shapes

For sequential models (GRU, Transformer), hidden state is just an extra input/output tensor in the ONNX graph — passed in, updated state returned. The ONNX runtime guarantees identical inference on both sides, so recurrence adds no validation complexity.

## Validation: NN-Based MT5 Parity Test

The ultimate correctness test: run the **same deterministic ONNX model** through our env/sim and through MT5 Strategy Tester, then compare trade registers.

This is stronger than the existing SMA-based validation (which only tests order mechanics) because it validates the **entire loop**:

| What's tested | SMA test | NN test |
|---|---|---|
| Observation construction | No | Yes |
| Action decoding (volume/price from model output) | No | Yes |
| Sim mechanics (fills, P&L, commissions) | Yes | Yes |

**The test model:** simple MLP with fixed-seed random weights, argmax action selection (no sampling), no dropout. Dumb but deterministic — same input always produces same output.

**Whole pipeline must be validated:** the feature computation (price ratios, normalization) must produce identical values in Python and MQL5. By keeping features minimal and self-relative (just price ratios + balance fractions), this parity is straightforward to achieve and maintain.

Proof of concept already exists: `../trader/mql5/OracleTrader.mq5` — runs ONNX inference in MT5 Strategy Tester with the same action types and parametric decoding.

## Requirements

### Speed
- Target: **millions of steps/sec** (M/s)
- Profile and benchmark everything — real measurements drive architecture decisions
- Minimize CPU↔GPU data transfer; be mindful of where inference runs

### Data
- Use our parquet files as data source (prices, spreads, timestamps)
- New data and instrument specs (commissions, contract sizes) via `../trader/tools/mt5_collect.py`

### Correctness
- **MT5 Strategy Tester parity is the ultimate validation** — zero-tolerance match on trade register (prices, P&L, commissions, timing)
- Reference implementation and tests: `../trader/tests/test_sim_vs_mt5.py` (1138 trades, exact match)
- Existing sim features to preserve: commission, swap, stop-out, margin checks, rollover window, delay bars

### Design
- Use existing libraries and tools where they solve the problem — don't reinvent
- Well-tested — tests should exercise the interface that production will use
- Clean, modular design — no hacks, no shortcuts

## Non-goals (for now)
- Portfolio management (agent choosing between assets or holding multiple positions across assets)
- Specific RL framework coupling
