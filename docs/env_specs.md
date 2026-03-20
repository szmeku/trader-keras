# Trading Simulation Environment — Specification

**Broker model:** IC Markets Raw Spread account
**Starting asset:** XAUUSD (Gold)
**Data:** 1-minute bars from `~/projects/data/icmarkets_*.parquet`

---

## 1. Data Format

Columns: `timestamp, open, high, low, close, volume, spread`
(`real_volume` — always 0, ignored)

- Timestamps are in **IC Markets server time (GMT+2 / GMT+3 DST)**, timezone-naive
- Daily break: ~23:58 → 01:02 server time (= 5PM–6PM ET, CME gold maintenance)
- Weekend: Friday ~23:56 → Monday ~01:02
- Spread column is in **points** (1 point = `point_size` in price units)

## 2. Instrument Config

Per-instrument specification. For v1 hardcoded for XAUUSD, later loaded from JSON (see [fetching script](#instrument-spec-fetching)).

| Property                | XAUUSD                      | Notes                      |
| ----------------------- | --------------------------- | -------------------------- |
| Contract size           | 100 oz                      | 1 lot = 100 troy ounces    |
| Point size              | 0.01                        | Smallest price increment   |
| Point value / lot       | $1.00                       | contract_size × point_size |
| Commission (Raw Spread) | $3.50 / side / lot          | $7 round turn              |
| Leverage                | configurable (default 1:20) | Max 1:1000 per IC Markets  |
| Min lot                 | 0.01                        | = 1 oz                     |
| Max lot                 | 100                         | = 10,000 oz                |
| Lot step                | 0.01                        |                            |

**Point size derivation:** auto-derive from data as GCD of price differences.
<!-- TODO: consider setting point_size manually per instrument instead of deriving —
     derivation may fail on sparse/gapped data or instruments with irregular tick sizes -->

### Instrument Spec Fetching

Uses the existing MT5 data bridge (see [metatrader.md](metatrader.md)):
```bash
python tools/mt5_collect.py --specs XAUUSD EURUSD GBPUSD
```
Exports `SymbolInfo` (contract_size, point, tick_size, tick_value, volumes, etc.) via the DataExporter EA → saves to `~/projects/data/instrument_specs.json`.
Specs are stable (rarely change), so the cached JSON is fine.

## 3. Costs

| Cost | Source | When applied |
|---|---|---|
| **Spread** | Actual per-bar `spread` column from data | At fill — bid/ask derived from OHLC + spread |
| **Commission** | $3.50/side/lot (from instrument config) | At fill |
| **Swaps** | **Punitive flat rate** (e.g., −$50/lot/night) | If position held past daily break |

**Swap avoidance strategy:** Set swap cost high enough that the model always learns to close before the daily break (~23:58 server time / 5PM ET). For MT5 validation: add a hard rule "close all positions at 23:50 server time" in both Python and MQL5 — this eliminates swaps from comparison entirely.

**Bid/Ask from OHLC + spread:**
MT5 OHLC prices are **bid** prices ([confirmed](https://www.mql5.com/en/forum/388842)). So:
- `bid = close` (the raw OHLC value)
- `ask = bid + spread * point_size`
- Same logic for open/high/low when checking intra-bar fills
- Buys fill at ask, sells fill at bid

## 4. Margin & Liquidation

- **Required margin** = (price × volume × contract_size) / leverage
- **Free margin** = equity − used_margin
- **Margin level** = (equity / used_margin) × 100%
- **Margin call** at 100% — warning only
- **Stop-out** at 50% — auto-close most unprofitable position, loop until margin level > 50%
- Reject new orders if required_margin > free_margin

## 5. Agent Actions

| Action        | Parameters          | Description                                                                             |
| ------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `LIMIT_ORDER` | side, volume, price | Place limit order (buy below ask, sell above bid)                                       |
| `STOP_ORDER`  | side, volume, price | Place stop order (buy above ask, sell below bid) — for stop-losses and breakout entries |
| `CANCEL`      | order_id            | Cancel pending order                                                                    |
| `HOLD`        | —                   | Do nothing                                                                              |

- **Side:** `BUY` or `SELL` (long / short)
- Orders are **good till cancelled** (persist until filled, cancelled, or stop-out)
- Limit orders fill when price reaches target; stop orders trigger when price crosses target
- Fill price = order price (no additional slippage beyond spread already in data)

## 6. Environment Auto-Actions

| Event       | Trigger                                 | Behavior                                  |
| ----------- | --------------------------------------- | ----------------------------------------- |
| Order fill  | Price touches limit/stop level on a bar | Execute at order price, deduct commission |
| Stop-out    | Margin level ≤ 50%                      | Close most unprofitable position          |
| Swap charge | Position held past daily break          | Deduct punitive swap                      |

## 7. Account State (tracked each step)

`balance, equity, used_margin, free_margin, margin_level`
Plus: list of open positions, list of pending orders.

## 8. Fill Model — MT5 "1 minute OHLC" Tick Generation

Each M1 bar is processed as **4 ticks** in a deterministic order matching MT5 Strategy Tester:
- **Bullish bar** (close >= open): `Open → Low → High → Close`
- **Bearish bar** (close < open): `Open → High → Low → Close`

Processing per bar:
1. **tick[0] (Open):** check old pending orders, check stop-out
2. **Agent action placed** (matches MT5 EA `OnTick` on new bar)
3. **ticks[0-3]:** check current pending + stop-out at each tick (new order can fill at Open if price matches — matches MT5 EA placing order at OnTick and filling immediately)
4. **Swap check** at bar level
5. **Reward** = equity change at bar close

**Netting** (MT5 netting account model): when a pending order triggers while an opposite-side position exists, the position is closed and any volume remainder opens a new position in the order's direction. Same volume = flatten, excess volume = reverse.

This means:
- **Limit orders at open price** fill at tick[0] — matches MT5 market order at OnTick
- **Stop-out** is checked at worst-price ticks (low for longs, high for shorts), not just close
- **Pending orders** can fill intra-bar at the tick that triggers them
- **Order of events within a bar matters** — stop-out and fills can race
- **Reversals** work in one step via netting (e.g., LIMIT_SELL with 2x volume closes long + opens short)

## 9. Design Goals

- **Correctness first**, speed second — but design for speed (numpy arrays, avoid per-tick Python loops)
- Minimal code (~200 lines target)
- Gymnasium `step()/reset()` interface
- Single instrument per env instance (v1)
- Replaces existing `crypto_trader/envs/`

## 10. Validation Strategy

**Goal:** Verify Python sim matches MT5 Strategy Tester trade-for-trade.

### Validation framework

Strategy-agnostic: register any `Strategy` class in `tools/strategies/__init__.py` and validate it.

```bash
# Full auto: Python sim + MT5 Strategy Tester + comparison
python tools/validate.py sma --symbol BTCUSD --auto

# Step-by-step
python tools/validate.py sma --symbol BTCUSD                  # 1. Run Python sim
python tools/validate.py sma --symbol BTCUSD --compare         # 2. Compare vs MT5

# Oracle baseline (no MT5 counterpart)
python tools/validate.py oracle --symbol XAUUSD --from 2025-12-01 --to 2025-12-15

# All runs log to W&B by default (--no-wandb to skip)
```

### Files

| File | Role |
|---|---|
| `tools/validate.py` | Validation orchestrator (dispatches to registered strategies) |
| `tools/strategies/base.py` | Strategy protocol + generic `run_strategy()` runner |
| `tools/strategies/sma_crossover.py` | `SMAStrategy` class (Python SMA crossover) |
| `tools/strategies/oracle.py` | `OracleStrategy` — DP oracle, theoretical profit ceiling |
| `tools/strategies/SMAStrategy.mq5` | MQL5 SMA EA (same logic, co-located with Python) |
| `tools/strategies/OracleStrategy.mq5` | MQL5 DP oracle EA — native port of Python DP solver |
| `tools/mt5_tester.py` | MT5 Strategy Tester config generation |
| `mql5/deploy.sh` | Deploys MQL5 files to MT5 Advisors folder |

---

## Implementation Status (2026-03-13)

### Implemented as specified
- [x] 4-tick OHLC model matching MT5 "1 minute OHLC" mode
- [x] Stop-out at 50% margin level, checked at each tick
- [x] Netting account model (close + reverse in one order)
- [x] Spread from data (per-bar, not constant)
- [x] Commission from `instrument_specs.json` (per-lot-side)
- [x] Punitive swap charge at hour 23
- [x] Rollover window 23:58-00:05 (broker rejects orders)
- [x] Margin calculation with configurable leverage
- [x] Limit and stop orders with correct fill semantics
- [x] P&L rounding via `Decimal.quantize(ROUND_HALF_UP)` matching MT5

### Implemented differently from original spec
- **Commission source**: Spec said "from MT5 deal history" — implemented as git-tracked `data/broker_commissions.json` with symbol→class→rate mapping instead (no trading required to obtain)
- **Gymnasium interface**: Not implemented on `TradingSim` — it has its own `step(bar, action)` API. The RL envs (`envs_legacy/`) wrap it with Gym interface separately
- **Multi-instrument**: Spec said "single instrument per env (v1)" — still single, no multi-asset env yet
- **Point size**: Spec said "auto-derive from GCD of price diffs" — `derive_point_size()` exists but we use `instrument_specs.json` values in practice (more reliable)

### Validated
- **BTCUSD**: 1138/1138 trades match (100%), entry=0.0000, exit=0.0000, P&L=0.0000
- **XAUUSD**: validated via SMA crossover strategy

### Not implemented (deferred)
- [ ] Tick-level validation (real tick data via `CopyTicks`) — OHLC model is sufficient for training
- [ ] `CopyRates` replay mode (alternative to Strategy Tester)
- [ ] Multi-asset strategies
- [ ] Strategy EAs with MQL_TESTER guard have been added — live trading not possible by accident

---

*See also: [env_requirements_default.md](env_requirements_default.md) for the full unfiltered IC Markets spec, [[oracle_baseline]] for the DP oracle comparison.*
